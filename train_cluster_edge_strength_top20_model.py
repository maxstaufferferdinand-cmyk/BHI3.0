from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")
CLUSTER_DIR = BASE_DIR / "surgical_time_normalized_cluster_edges"
GRAPH_DIR = BASE_DIR / "surgical_training_graph_outputs"

OUT_DIR = BASE_DIR / "cluster_edge_strength_model_outputs"
OUT_DIR.mkdir(exist_ok=True)

CLUSTER_EDGES_CSV = CLUSTER_DIR / "cluster_edges_time_normalized.csv"
PROBLEM_CLUSTERS_CSV = CLUSTER_DIR / "problem_node_clusters.csv"
SOLUTION_CLUSTERS_CSV = CLUSTER_DIR / "solution_node_clusters.csv"

PROBLEM_EMB_NPY = GRAPH_DIR / "surgical_problem_embeddings.npy"
SOLUTION_EMB_NPY = GRAPH_DIR / "surgical_solution_embeddings.npy"

RANDOM_STATE = 42

# Endpoint definition
TOP_POSITIVE_FRACTION = 0.20
BOTTOM_NEGATIVE_FRACTION = 0.50

# Score composition
WEIGHT_WITHIN_YEAR = 0.70
WEIGHT_TIME_ADJUSTED = 0.30

OUT_MODEL = OUT_DIR / "cluster_edge_strength_top20_model.joblib"
OUT_SUMMARY = OUT_DIR / "cluster_edge_strength_top20_summary.txt"
OUT_METRICS = OUT_DIR / "cluster_edge_strength_top20_metrics.csv"
OUT_DATASET = OUT_DIR / "cluster_edge_strength_top20_training_dataset.csv"
OUT_FEATURES = OUT_DIR / "cluster_edge_strength_top20_feature_names.json"
OUT_IMPORTANCE = OUT_DIR / "cluster_edge_strength_top20_feature_importance.csv"
OUT_ROC = OUT_DIR / "cluster_edge_strength_top20_roc.png"
OUT_PR = OUT_DIR / "cluster_edge_strength_top20_precision_recall.png"


# ============================================================
# HELPERS
# ============================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def normalize_matrix(x):
    x = x.astype(np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


def log1p_series(x):
    return np.log1p(pd.to_numeric(x, errors="coerce").fillna(0).clip(lower=0))


def rank_percentile(x):
    return pd.to_numeric(x, errors="coerce").rank(method="average", pct=True)


def cosine_matrix_rows(a, b):
    return np.sum(a * b, axis=1)


def build_cluster_centroids(node_df, emb, cluster_col):
    """
    Embeddings are assumed to align row-wise with node_df.
    """
    clusters = node_df[cluster_col].astype(int).values
    unique_clusters = np.sort(np.unique(clusters))

    centroids = {}
    sizes = {}

    for c in unique_clusters:
        idx = np.where(clusters == c)[0]
        vec = emb[idx].mean(axis=0)
        vec = vec / max(np.linalg.norm(vec), 1e-12)
        centroids[int(c)] = vec.astype(np.float32)
        sizes[int(c)] = int(len(idx))

    return centroids, sizes


def get_centroid_array(cluster_ids, centroid_dict, dim):
    arr = np.zeros((len(cluster_ids), dim), dtype=np.float32)
    for i, cid in enumerate(cluster_ids):
        arr[i] = centroid_dict.get(int(cid), np.zeros(dim, dtype=np.float32))
    return arr


def plot_roc(y_test, pred):
    fpr, tpr, _ = roc_curve(y_test, pred)
    auc = roc_auc_score(y_test, pred)

    plt.figure(figsize=(6.5, 5.7), dpi=300)
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black", alpha=0.7)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Cluster-edge strength model ROC")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_ROC, dpi=300)
    plt.close()


def plot_pr(y_test, pred):
    precision, recall, _ = precision_recall_curve(y_test, pred)
    ap = average_precision_score(y_test, pred)

    plt.figure(figsize=(6.5, 5.7), dpi=300)
    plt.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Cluster-edge strength model precision-recall")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PR, dpi=300)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("Checking files...")
    for p in [
        CLUSTER_EDGES_CSV,
        PROBLEM_CLUSTERS_CSV,
        SOLUTION_CLUSTERS_CSV,
        PROBLEM_EMB_NPY,
        SOLUTION_EMB_NPY,
    ]:
        require_file(p)

    print("Loading data...")
    edges = pd.read_csv(CLUSTER_EDGES_CSV, encoding="utf-8-sig", low_memory=False)
    problem_nodes = pd.read_csv(PROBLEM_CLUSTERS_CSV, encoding="utf-8-sig", low_memory=False)
    solution_nodes = pd.read_csv(SOLUTION_CLUSTERS_CSV, encoding="utf-8-sig", low_memory=False)

    problem_emb = normalize_matrix(np.load(PROBLEM_EMB_NPY))
    solution_emb = normalize_matrix(np.load(SOLUTION_EMB_NPY))

    print("Cluster edges:", edges.shape)
    print("Problem nodes:", problem_nodes.shape)
    print("Solution nodes:", solution_nodes.shape)
    print("Problem embeddings:", problem_emb.shape)
    print("Solution embeddings:", solution_emb.shape)

    required_cols = {
        "problem_cluster_id",
        "solution_cluster_id",
        "cluster_weight",
        "n_exact_edges",
        "n_unique_problem_nodes",
        "n_unique_solution_nodes",
        "first_year",
        "last_year",
        "cluster_age_since_first_years",
        "cluster_active_span_years",
        "cluster_time_adjusted_strength",
        "cluster_within_year_percentile",
        "cluster_edge_density_proxy",
    }

    missing = required_cols - set(edges.columns)
    if missing:
        raise ValueError(f"Missing required columns in cluster edges: {missing}")

    if len(problem_nodes) != problem_emb.shape[0]:
        raise ValueError("Problem node table and embedding matrix have different row counts.")
    if len(solution_nodes) != solution_emb.shape[0]:
        raise ValueError("Solution node table and embedding matrix have different row counts.")

    # ------------------------------------------------------------
    # 1. Build one efficient final edge-strength score
    # ------------------------------------------------------------

    edges = edges.copy()
    edges["problem_cluster_id"] = edges["problem_cluster_id"].astype(int)
    edges["solution_cluster_id"] = edges["solution_cluster_id"].astype(int)

    # Global percentile of the existing time-adjusted strength.
    edges["time_adjusted_percentile_global"] = rank_percentile(edges["cluster_time_adjusted_strength"])

    # Use existing within-year percentile from previous script.
    edges["cluster_within_year_percentile"] = pd.to_numeric(
        edges["cluster_within_year_percentile"], errors="coerce"
    ).fillna(0.0)

    edges["final_edge_strength"] = (
        WEIGHT_WITHIN_YEAR * edges["cluster_within_year_percentile"]
        + WEIGHT_TIME_ADJUSTED * edges["time_adjusted_percentile_global"]
    )

    top_cut = edges["final_edge_strength"].quantile(1 - TOP_POSITIVE_FRACTION)
    bottom_cut = edges["final_edge_strength"].quantile(BOTTOM_NEGATIVE_FRACTION)

    edges["label"] = np.nan
    edges.loc[edges["final_edge_strength"] >= top_cut, "label"] = 1
    edges.loc[edges["final_edge_strength"] <= bottom_cut, "label"] = 0

    model_df = edges.dropna(subset=["label"]).copy()
    model_df["label"] = model_df["label"].astype(int)

    print("\nEndpoint:")
    print(f"Top positive fraction: {TOP_POSITIVE_FRACTION}")
    print(f"Bottom negative fraction: {BOTTOM_NEGATIVE_FRACTION}")
    print(f"Top cutoff final_edge_strength: {top_cut:.4f}")
    print(f"Bottom cutoff final_edge_strength: {bottom_cut:.4f}")
    print(model_df["label"].value_counts().sort_index())

    # ------------------------------------------------------------
    # 2. Build non-leaky features
    # ------------------------------------------------------------
    # Direct target-defining values like cluster_weight, n_exact_edges,
    # cluster_time_adjusted_strength, final_edge_strength are NOT used as features.

    print("\nBuilding cluster centroids...")
    problem_centroids, problem_cluster_sizes = build_cluster_centroids(
        problem_nodes, problem_emb, "problem_cluster_id"
    )
    solution_centroids, solution_cluster_sizes = build_cluster_centroids(
        solution_nodes, solution_emb, "solution_cluster_id"
    )

    dim = problem_emb.shape[1]
    p_cent = get_centroid_array(model_df["problem_cluster_id"].values, problem_centroids, dim)
    s_cent = get_centroid_array(model_df["solution_cluster_id"].values, solution_centroids, dim)
    centroid_cosine = cosine_matrix_rows(p_cent, s_cent)

    # Cluster-level graph degrees and totals, computed across all cluster edges.
    # Then subtract current edge contribution where appropriate to reduce leakage.
    p_group = edges.groupby("problem_cluster_id").agg(
        problem_cluster_degree_all=("solution_cluster_id", "nunique"),
        problem_cluster_total_weight_all=("cluster_weight", "sum"),
        problem_cluster_total_edges_all=("n_exact_edges", "sum"),
    )

    s_group = edges.groupby("solution_cluster_id").agg(
        solution_cluster_degree_all=("problem_cluster_id", "nunique"),
        solution_cluster_total_weight_all=("cluster_weight", "sum"),
        solution_cluster_total_edges_all=("n_exact_edges", "sum"),
    )

    model_df = model_df.merge(p_group, on="problem_cluster_id", how="left")
    model_df = model_df.merge(s_group, on="solution_cluster_id", how="left")

    model_df["problem_cluster_size"] = model_df["problem_cluster_id"].map(problem_cluster_sizes).fillna(0)
    model_df["solution_cluster_size"] = model_df["solution_cluster_id"].map(solution_cluster_sizes).fillna(0)

    # Subtract the current edge from cluster totals.
    model_df["problem_degree_without_edge"] = (
        model_df["problem_cluster_degree_all"] - 1
    ).clip(lower=0)

    model_df["solution_degree_without_edge"] = (
        model_df["solution_cluster_degree_all"] - 1
    ).clip(lower=0)

    model_df["problem_total_weight_without_edge"] = (
        model_df["problem_cluster_total_weight_all"] - model_df["cluster_weight"]
    ).clip(lower=0)

    model_df["solution_total_weight_without_edge"] = (
        model_df["solution_cluster_total_weight_all"] - model_df["cluster_weight"]
    ).clip(lower=0)

    model_df["problem_total_exact_edges_without_edge"] = (
        model_df["problem_cluster_total_edges_all"] - model_df["n_exact_edges"]
    ).clip(lower=0)

    model_df["solution_total_exact_edges_without_edge"] = (
        model_df["solution_cluster_total_edges_all"] - model_df["n_exact_edges"]
    ).clip(lower=0)

    model_df["centroid_cosine"] = centroid_cosine

    # Keep feature set compact.
    feature_names = [
        "centroid_cosine",
        "problem_cluster_size_log",
        "solution_cluster_size_log",
        "problem_degree_without_edge_log",
        "solution_degree_without_edge_log",
        "problem_total_weight_without_edge_log",
        "solution_total_weight_without_edge_log",
        "problem_total_exact_edges_without_edge_log",
        "solution_total_exact_edges_without_edge_log",
        "cluster_edge_density_proxy",
    ]

    model_df["problem_cluster_size_log"] = log1p_series(model_df["problem_cluster_size"])
    model_df["solution_cluster_size_log"] = log1p_series(model_df["solution_cluster_size"])
    model_df["problem_degree_without_edge_log"] = log1p_series(model_df["problem_degree_without_edge"])
    model_df["solution_degree_without_edge_log"] = log1p_series(model_df["solution_degree_without_edge"])
    model_df["problem_total_weight_without_edge_log"] = log1p_series(model_df["problem_total_weight_without_edge"])
    model_df["solution_total_weight_without_edge_log"] = log1p_series(model_df["solution_total_weight_without_edge"])
    model_df["problem_total_exact_edges_without_edge_log"] = log1p_series(model_df["problem_total_exact_edges_without_edge"])
    model_df["solution_total_exact_edges_without_edge_log"] = log1p_series(model_df["solution_total_exact_edges_without_edge"])
    model_df["cluster_edge_density_proxy"] = pd.to_numeric(
        model_df["cluster_edge_density_proxy"], errors="coerce"
    ).fillna(0)

    X = model_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    y = model_df["label"].values.astype(int)

    # ------------------------------------------------------------
    # 3. Train/test split and model
    # ------------------------------------------------------------

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        np.arange(len(model_df)),
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining model...")
    model = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.03,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, pred)
    ap = average_precision_score(y_test, pred)

    print("\nMetrics:")
    print("ROC-AUC:", roc_auc)
    print("Average precision:", ap)

    # ------------------------------------------------------------
    # 4. Feature importance
    # ------------------------------------------------------------

    print("\nCalculating permutation importance...")
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean_auc_drop": perm.importances_mean,
        "importance_sd": perm.importances_std,
    }).sort_values("importance_mean_auc_drop", ascending=False)

    # ------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------

    joblib.dump(model, OUT_MODEL)

    OUT_FEATURES.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame([{
        "roc_auc": roc_auc,
        "average_precision": ap,
        "n_total_used": len(model_df),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "top_cut_final_edge_strength": float(top_cut),
        "bottom_cut_final_edge_strength": float(bottom_cut),
        "positive_fraction_definition": TOP_POSITIVE_FRACTION,
        "negative_fraction_definition": BOTTOM_NEGATIVE_FRACTION,
    }])

    metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    imp_df.to_csv(OUT_IMPORTANCE, index=False, encoding="utf-8-sig")

    plot_roc(y_test, pred)
    plot_pr(y_test, pred)

    output_df = model_df.copy()
    output_df["used_for_model"] = True
    output_df["ml_predicted_probability_strong"] = np.nan
    output_df.iloc[idx_test, output_df.columns.get_loc("ml_predicted_probability_strong")] = pred

    keep_cols = [
        "problem_cluster_id",
        "solution_cluster_id",
        "problem_cluster_name",
        "solution_cluster_name",
        "cluster_weight",
        "n_exact_edges",
        "first_year",
        "last_year",
        "cluster_time_adjusted_strength",
        "cluster_within_year_percentile",
        "time_adjusted_percentile_global",
        "final_edge_strength",
        "label",
        "ml_predicted_probability_strong",
        "centroid_cosine",
        "example_surgical_problems",
        "example_technical_solutions",
    ]
    keep_cols = [c for c in keep_cols if c in output_df.columns]
    output_df[keep_cols].to_csv(OUT_DATASET, index=False, encoding="utf-8-sig")

    summary = []
    summary.append("Cluster-edge strength top-20 model")
    summary.append("=" * 90)
    summary.append("")
    summary.append("Endpoint:")
    summary.append("Positive class: top 20% of final_edge_strength.")
    summary.append("Negative class: bottom 50% of final_edge_strength.")
    summary.append("Middle 30% excluded.")
    summary.append("")
    summary.append("final_edge_strength:")
    summary.append(f"{WEIGHT_WITHIN_YEAR} * cluster_within_year_percentile + {WEIGHT_TIME_ADJUSTED} * global percentile of cluster_time_adjusted_strength")
    summary.append("")
    summary.append("Important:")
    summary.append("Direct target-defining variables such as cluster_weight, n_exact_edges, and cluster_time_adjusted_strength were not used as ML features.")
    summary.append("")
    summary.append(f"Total cluster edges: {len(edges):,}")
    summary.append(f"Model dataset rows: {len(model_df):,}")
    summary.append(f"Positive rows: {int((y == 1).sum()):,}")
    summary.append(f"Negative rows: {int((y == 0).sum()):,}")
    summary.append("")
    summary.append(f"Top cutoff final_edge_strength: {top_cut:.6f}")
    summary.append(f"Bottom cutoff final_edge_strength: {bottom_cut:.6f}")
    summary.append("")
    summary.append("Metrics:")
    summary.append(f"ROC-AUC: {roc_auc:.6f}")
    summary.append(f"Average precision: {ap:.6f}")
    summary.append("")
    summary.append("Features:")
    for f in feature_names:
        summary.append(f"- {f}")
    summary.append("")
    summary.append("Permutation feature importance:")
    summary.append(imp_df.to_string(index=False))
    summary.append("")
    summary.append("Output files:")
    for p in [
        OUT_MODEL,
        OUT_METRICS,
        OUT_IMPORTANCE,
        OUT_DATASET,
        OUT_FEATURES,
        OUT_ROC,
        OUT_PR,
    ]:
        summary.append(str(p))

    OUT_SUMMARY.write_text("\n".join(summary), encoding="utf-8")

    print("\nSaved:")
    print(OUT_SUMMARY)
    print(OUT_METRICS)
    print(OUT_IMPORTANCE)
    print(OUT_DATASET)
    print(OUT_ROC)
    print(OUT_PR)
    print("\nDONE")


if __name__ == "__main__":
    main()