from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

CLUSTER_DIR = BASE_DIR / "surgical_time_normalized_cluster_edges"
GRAPH_DIR = BASE_DIR / "surgical_training_graph_outputs"
ABLATION_DIR = BASE_DIR / "cluster_edge_strength_ablation_outputs"

CLUSTER_EDGES_CSV = CLUSTER_DIR / "cluster_edges_time_normalized.csv"
PROBLEM_CLUSTERS_CSV = CLUSTER_DIR / "problem_node_clusters.csv"
SOLUTION_CLUSTERS_CSV = CLUSTER_DIR / "solution_node_clusters.csv"

PROBLEM_EMB_NPY = GRAPH_DIR / "surgical_problem_embeddings.npy"
SOLUTION_EMB_NPY = GRAPH_DIR / "surgical_solution_embeddings.npy"

OUT_DIR = BASE_DIR / "roc_reference_style_outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_PNG = OUT_DIR / "ablation_roc_curves_reference_style.png"
OUT_PDF = OUT_DIR / "ablation_roc_curves_reference_style.pdf"
OUT_CSV = OUT_DIR / "ablation_roc_auc_summary_reference_style.csv"

RANDOM_STATE = 42

TOP_POSITIVE_FRACTION = 0.20
BOTTOM_NEGATIVE_FRACTION = 0.50

WEIGHT_WITHIN_YEAR = 0.70
WEIGHT_TIME_ADJUSTED = 0.30


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


# ============================================================
# MAIN
# ============================================================

def main():
    for p in [
        CLUSTER_EDGES_CSV,
        PROBLEM_CLUSTERS_CSV,
        SOLUTION_CLUSTERS_CSV,
        PROBLEM_EMB_NPY,
        SOLUTION_EMB_NPY,
    ]:
        require_file(p)

    edges = pd.read_csv(CLUSTER_EDGES_CSV, encoding="utf-8-sig", low_memory=False)
    problem_nodes = pd.read_csv(PROBLEM_CLUSTERS_CSV, encoding="utf-8-sig", low_memory=False)
    solution_nodes = pd.read_csv(SOLUTION_CLUSTERS_CSV, encoding="utf-8-sig", low_memory=False)

    problem_emb = normalize_matrix(np.load(PROBLEM_EMB_NPY))
    solution_emb = normalize_matrix(np.load(SOLUTION_EMB_NPY))

    edges = edges.copy()
    edges["problem_cluster_id"] = edges["problem_cluster_id"].astype(int)
    edges["solution_cluster_id"] = edges["solution_cluster_id"].astype(int)

    # ========================================================
    # Rebuild endpoint
    # ========================================================

    edges["time_adjusted_percentile_global"] = rank_percentile(
        edges["cluster_time_adjusted_strength"]
    )

    edges["cluster_within_year_percentile"] = pd.to_numeric(
        edges["cluster_within_year_percentile"],
        errors="coerce"
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

    # ========================================================
    # Rebuild features exactly as before
    # ========================================================

    problem_centroids, problem_cluster_sizes = build_cluster_centroids(
        problem_nodes, problem_emb, "problem_cluster_id"
    )

    solution_centroids, solution_cluster_sizes = build_cluster_centroids(
        solution_nodes, solution_emb, "solution_cluster_id"
    )

    dim = problem_emb.shape[1]

    p_cent = get_centroid_array(
        model_df["problem_cluster_id"].values,
        problem_centroids,
        dim
    )

    s_cent = get_centroid_array(
        model_df["solution_cluster_id"].values,
        solution_centroids,
        dim
    )

    model_df["centroid_cosine"] = cosine_matrix_rows(p_cent, s_cent)

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

    model_df["problem_cluster_size_log"] = log1p_series(model_df["problem_cluster_size"])
    model_df["solution_cluster_size_log"] = log1p_series(model_df["solution_cluster_size"])
    model_df["problem_degree_without_edge_log"] = log1p_series(model_df["problem_degree_without_edge"])
    model_df["solution_degree_without_edge_log"] = log1p_series(model_df["solution_degree_without_edge"])
    model_df["problem_total_weight_without_edge_log"] = log1p_series(model_df["problem_total_weight_without_edge"])
    model_df["solution_total_weight_without_edge_log"] = log1p_series(model_df["solution_total_weight_without_edge"])
    model_df["problem_total_exact_edges_without_edge_log"] = log1p_series(model_df["problem_total_exact_edges_without_edge"])
    model_df["solution_total_exact_edges_without_edge_log"] = log1p_series(model_df["solution_total_exact_edges_without_edge"])

    model_df["cluster_edge_density_proxy"] = pd.to_numeric(
        model_df["cluster_edge_density_proxy"],
        errors="coerce"
    ).fillna(0)

    # Same feature sets as ablation script
    feature_sets = {
        "Combined graph + embedding": {
            "model_file": "combined_graph_embedding.joblib",
            "features": [
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
            "color": "#1f77b4",
        },

        "Embedding only": {
            "model_file": "embedding_only.joblib",
            "features": [
                "centroid_cosine",
                "problem_cluster_size_log",
                "solution_cluster_size_log",
            ],
            "color": "#ff7f0e",
        },

        "Graph topology only": {
            "model_file": "graph_topology_only.joblib",
            "features": [
                "problem_degree_without_edge_log",
                "solution_degree_without_edge_log",
                "problem_total_weight_without_edge_log",
                "solution_total_weight_without_edge_log",
                "problem_total_exact_edges_without_edge_log",
                "solution_total_exact_edges_without_edge_log",
                "cluster_edge_density_proxy",
            ],
            "color": "#2ca02c",
        },

        "Degree only": {
            "model_file": "degree_only.joblib",
            "features": [
                "problem_degree_without_edge_log",
                "solution_degree_without_edge_log",
            ],
            "color": "#d62728",
        },

        "Hub-weight only": {
            "model_file": "hub_weight_only.joblib",
            "features": [
                "problem_total_weight_without_edge_log",
                "solution_total_weight_without_edge_log",
                "problem_total_exact_edges_without_edge_log",
                "solution_total_exact_edges_without_edge_log",
            ],
            "color": "#9467bd",
        },

        "No hub-weight features": {
            "model_file": "no_hub_weight_features.joblib",
            "features": [
                "centroid_cosine",
                "problem_cluster_size_log",
                "solution_cluster_size_log",
                "problem_degree_without_edge_log",
                "solution_degree_without_edge_log",
                "cluster_edge_density_proxy",
            ],
            "color": "#8c564b",
        },
    }

    y = model_df["label"].values.astype(int)
    row_index = np.arange(len(model_df))

    _, test_idx = train_test_split(
        row_index,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    y_test = y[test_idx]

    curves = []
    summary_rows = []

    for model_label, cfg in feature_sets.items():
        model_path = ABLATION_DIR / cfg["model_file"]
        require_file(model_path)

        model = joblib.load(model_path)

        X_all = (
            model_df[cfg["features"]]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .values
            .astype(np.float32)
        )

        X_test = X_all[test_idx]

        pred = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, pred)
        auc = roc_auc_score(y_test, pred)

        curves.append({
            "label": model_label,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
            "color": cfg["color"],
        })

        summary_rows.append({
            "model": model_label,
            "auc": auc,
            "n_test": len(y_test),
            "n_test_positive": int((y_test == 1).sum()),
            "n_test_negative": int((y_test == 0).sum()),
        })

    # ========================================================
    # Plot like the reference image
    # ========================================================

    plt.figure(figsize=(8.2, 6.2), dpi=300)

    for c in curves:
        plt.plot(
            c["fpr"],
            c["tpr"],
            color=c["color"],
            linewidth=2.0,
            label=f"{c['label']} (AUC = {c['auc']:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        color="#b084cc",
        linestyle="--",
        linewidth=1.5,
        label="Random"
    )

    plt.title("ROC curves for link prediction models", fontsize=14)
    plt.xlabel("False positive rate", fontsize=12)
    plt.ylabel("True positive rate", fontsize=12)

    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.05)

    plt.legend(
        loc="lower right",
        fontsize=8,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.savefig(OUT_PDF)
    plt.close()

    summary_df = pd.DataFrame(summary_rows).sort_values("auc", ascending=False)
    summary_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("Saved:")
    print(OUT_PNG)
    print(OUT_PDF)
    print(OUT_CSV)
    print("")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()