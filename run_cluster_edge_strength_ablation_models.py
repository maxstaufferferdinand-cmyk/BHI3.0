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

OUT_DIR = BASE_DIR / "cluster_edge_strength_ablation_outputs"
OUT_DIR.mkdir(exist_ok=True)

CLUSTER_EDGES_CSV = CLUSTER_DIR / "cluster_edges_time_normalized.csv"
PROBLEM_CLUSTERS_CSV = CLUSTER_DIR / "problem_node_clusters.csv"
SOLUTION_CLUSTERS_CSV = CLUSTER_DIR / "solution_node_clusters.csv"

PROBLEM_EMB_NPY = GRAPH_DIR / "surgical_problem_embeddings.npy"
SOLUTION_EMB_NPY = GRAPH_DIR / "surgical_solution_embeddings.npy"

RANDOM_STATE = 42

TOP_POSITIVE_FRACTION = 0.20
BOTTOM_NEGATIVE_FRACTION = 0.50

WEIGHT_WITHIN_YEAR = 0.70
WEIGHT_TIME_ADJUSTED = 0.30

OUT_SUMMARY = OUT_DIR / "cluster_edge_strength_ablation_summary.txt"
OUT_METRICS = OUT_DIR / "cluster_edge_strength_ablation_metrics.csv"
OUT_IMPORTANCE = OUT_DIR / "cluster_edge_strength_ablation_feature_importance.csv"
OUT_ROC_PNG = OUT_DIR / "cluster_edge_strength_ablation_roc.png"
OUT_PR_PNG = OUT_DIR / "cluster_edge_strength_ablation_precision_recall.png"


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path):
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


def train_eval_model(X_train, X_test, y_train, y_test, feature_names, model_name):
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

    # permutation importance; safe for compact feature sets
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )

    imp = pd.DataFrame({
        "model": model_name,
        "feature": feature_names,
        "importance_mean_auc_drop": perm.importances_mean,
        "importance_sd": perm.importances_std,
    }).sort_values(["model", "importance_mean_auc_drop"], ascending=[True, False])

    return model, pred, roc_auc, ap, imp


def plot_roc_curves(curves):
    plt.figure(figsize=(7.4, 6.4), dpi=300)

    plt.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle="--",
        linewidth=1.1,
        alpha=0.6,
        label="Chance (AUC = 0.500)",
    )

    for c in curves:
        fpr, tpr, _ = roc_curve(c["y_test"], c["pred"])
        plt.plot(
            fpr,
            tpr,
            linewidth=2.2,
            label=f"{c['model']} (AUC = {c['roc_auc']:.3f})",
        )

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Ablation ROC curves")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8.5, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_ROC_PNG, dpi=300)
    plt.close()


def plot_pr_curves(curves):
    plt.figure(figsize=(7.4, 6.4), dpi=300)

    for c in curves:
        precision, recall, _ = precision_recall_curve(c["y_test"], c["pred"])
        plt.plot(
            recall,
            precision,
            linewidth=2.2,
            label=f"{c['model']} (AP = {c['ap']:.3f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Ablation precision-recall curves")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8.5, loc="lower left")
    plt.tight_layout()
    plt.savefig(OUT_PR_PNG, dpi=300)
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
        raise ValueError("Problem node table and problem embedding matrix have different row counts.")
    if len(solution_nodes) != solution_emb.shape[0]:
        raise ValueError("Solution node table and solution embedding matrix have different row counts.")

    edges = edges.copy()
    edges["problem_cluster_id"] = edges["problem_cluster_id"].astype(int)
    edges["solution_cluster_id"] = edges["solution_cluster_id"].astype(int)

    # ========================================================
    # 1. Rebuild same endpoint as current model
    # ========================================================

    edges["time_adjusted_percentile_global"] = rank_percentile(edges["cluster_time_adjusted_strength"])
    edges["cluster_within_year_percentile"] = pd.to_numeric(
        edges["cluster_within_year_percentile"], errors="coerce"
    ).fillna(0)

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
    print(f"Top cutoff: {top_cut:.6f}")
    print(f"Bottom cutoff: {bottom_cut:.6f}")
    print(model_df["label"].value_counts().sort_index())

    # ========================================================
    # 2. Build compact non-leaky feature matrix
    # ========================================================

    print("\nBuilding centroids...")
    problem_centroids, problem_cluster_sizes = build_cluster_centroids(
        problem_nodes, problem_emb, "problem_cluster_id"
    )
    solution_centroids, solution_cluster_sizes = build_cluster_centroids(
        solution_nodes, solution_emb, "solution_cluster_id"
    )

    dim = problem_emb.shape[1]
    p_cent = get_centroid_array(model_df["problem_cluster_id"].values, problem_centroids, dim)
    s_cent = get_centroid_array(model_df["solution_cluster_id"].values, solution_centroids, dim)

    model_df["centroid_cosine"] = cosine_matrix_rows(p_cent, s_cent)

    # Graph neighborhood features across all cluster edges.
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

    # Edge-excluded graph context to reduce direct leakage.
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

    # Log transforms
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

    # ========================================================
    # 3. Feature groups
    # ========================================================

    feature_sets = {
        "combined_graph_embedding": [
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
        ],

        "embedding_only": [
            "centroid_cosine",
            "problem_cluster_size_log",
            "solution_cluster_size_log",
        ],

        "graph_topology_only": [
            "problem_degree_without_edge_log",
            "solution_degree_without_edge_log",
            "problem_total_weight_without_edge_log",
            "solution_total_weight_without_edge_log",
            "problem_total_exact_edges_without_edge_log",
            "solution_total_exact_edges_without_edge_log",
            "cluster_edge_density_proxy",
        ],

        "degree_only": [
            "problem_degree_without_edge_log",
            "solution_degree_without_edge_log",
        ],

        "hub_weight_only": [
            "problem_total_weight_without_edge_log",
            "solution_total_weight_without_edge_log",
            "problem_total_exact_edges_without_edge_log",
            "solution_total_exact_edges_without_edge_log",
        ],

        "no_hub_weight_features": [
            "centroid_cosine",
            "problem_cluster_size_log",
            "solution_cluster_size_log",
            "problem_degree_without_edge_log",
            "solution_degree_without_edge_log",
            "cluster_edge_density_proxy",
        ],
    }

    # ========================================================
    # 4. Same train/test split for all ablations
    # ========================================================

    y = model_df["label"].values.astype(int)
    row_index = np.arange(len(model_df))

    train_idx, test_idx = train_test_split(
        row_index,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    curves = []
    metrics_rows = []
    importance_rows = []

    print("\nTraining ablation models...")

    for model_name, feature_names in feature_sets.items():
        print(f"\nModel: {model_name}")
        print("Features:", feature_names)

        X_all = (
            model_df[feature_names]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .values
            .astype(np.float32)
        )

        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model, pred, roc_auc, ap, imp = train_eval_model(
            X_train,
            X_test,
            y_train,
            y_test,
            feature_names,
            model_name,
        )

        model_path = OUT_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        print(f"ROC-AUC: {roc_auc:.6f}")
        print(f"Average precision: {ap:.6f}")

        metrics_rows.append({
            "model": model_name,
            "roc_auc": roc_auc,
            "average_precision": ap,
            "n_features": len(feature_names),
            "features": "; ".join(feature_names),
            "model_path": str(model_path),
        })

        importance_rows.append(imp)

        curves.append({
            "model": model_name,
            "pred": pred,
            "y_test": y_test,
            "roc_auc": roc_auc,
            "ap": ap,
        })

    # ========================================================
    # 5. Save results
    # ========================================================

    metrics_df = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
    importance_df = pd.concat(importance_rows, ignore_index=True)

    metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    importance_df.to_csv(OUT_IMPORTANCE, index=False, encoding="utf-8-sig")

    plot_roc_curves(curves)
    plot_pr_curves(curves)

    summary = []
    summary.append("Cluster-edge strength ablation models")
    summary.append("=" * 90)
    summary.append("")
    summary.append("Endpoint:")
    summary.append("Positive class: top 20% of final cluster-edge strength.")
    summary.append("Negative class: bottom 50% of final cluster-edge strength.")
    summary.append("Middle 30% excluded.")
    summary.append("")
    summary.append("final_edge_strength:")
    summary.append(f"{WEIGHT_WITHIN_YEAR} * cluster_within_year_percentile + {WEIGHT_TIME_ADJUSTED} * global percentile of cluster_time_adjusted_strength")
    summary.append("")
    summary.append(f"Total cluster edges: {len(edges):,}")
    summary.append(f"Model dataset rows: {len(model_df):,}")
    summary.append(f"Positive rows: {int((y == 1).sum()):,}")
    summary.append(f"Negative rows: {int((y == 0).sum()):,}")
    summary.append("")
    summary.append(f"Top cutoff: {top_cut:.6f}")
    summary.append(f"Bottom cutoff: {bottom_cut:.6f}")
    summary.append("")
    summary.append("Ablation metrics:")
    summary.append(metrics_df[["model", "roc_auc", "average_precision", "n_features"]].to_string(index=False))
    summary.append("")
    summary.append("Interpretation guide:")
    summary.append("- If combined_graph_embedding is clearly best, graph structure and embeddings complement each other.")
    summary.append("- If graph_topology_only is almost identical to combined, the model is mostly graph/hub-driven.")
    summary.append("- If embedding_only is substantially above random, semantic cluster alignment contributes independently.")
    summary.append("- If no_hub_weight_features remains strong, the result is not only driven by high-frequency hub clusters.")
    summary.append("")
    summary.append("Permutation feature importance:")
    for model_name in metrics_df["model"]:
        sub = importance_df[importance_df["model"] == model_name].sort_values(
            "importance_mean_auc_drop", ascending=False
        )
        summary.append("")
        summary.append(f"[{model_name}]")
        summary.append(sub.to_string(index=False))
    summary.append("")
    summary.append("Output files:")
    for p in [OUT_METRICS, OUT_IMPORTANCE, OUT_ROC_PNG, OUT_PR_PNG]:
        summary.append(str(p))

    OUT_SUMMARY.write_text("\n".join(summary), encoding="utf-8")

    print("\nSaved:")
    print(OUT_SUMMARY)
    print(OUT_METRICS)
    print(OUT_IMPORTANCE)
    print(OUT_ROC_PNG)
    print(OUT_PR_PNG)
    print("\nDONE")


if __name__ == "__main__":
    main()