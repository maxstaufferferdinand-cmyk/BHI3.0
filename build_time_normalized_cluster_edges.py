from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import joblib


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")
GRAPH_DIR = BASE_DIR / "surgical_training_graph_outputs"
OUT_DIR = BASE_DIR / "surgical_time_normalized_cluster_edges"
OUT_DIR.mkdir(exist_ok=True)

EDGE_FILE = GRAPH_DIR / "surgical_graph_edges_aggregated_with_years.csv"

PROBLEM_INDEX_CSV = GRAPH_DIR / "surgical_problem_embedding_index.csv"
SOLUTION_INDEX_CSV = GRAPH_DIR / "surgical_solution_embedding_index.csv"

PROBLEM_EMB_NPY = GRAPH_DIR / "surgical_problem_embeddings.npy"
SOLUTION_EMB_NPY = GRAPH_DIR / "surgical_solution_embeddings.npy"

# Adjust later if needed.
N_PROBLEM_CLUSTERS = 1200
N_SOLUTION_CLUSTERS = 1800

RANDOM_STATE = 42
BATCH_SIZE = 4096
MAX_ITER = 150

# If None, use max(last_year / first_year) in edge file.
ANALYSIS_END_YEAR = None

# Main outputs
OUT_PROBLEM_CLUSTERS = OUT_DIR / "problem_node_clusters.csv"
OUT_SOLUTION_CLUSTERS = OUT_DIR / "solution_node_clusters.csv"
OUT_EXACT_EDGES = OUT_DIR / "exact_edges_time_normalized.csv"
OUT_CLUSTER_EDGES = OUT_DIR / "cluster_edges_time_normalized.csv"

OUT_PROBLEM_KMEANS = OUT_DIR / "problem_kmeans_model.joblib"
OUT_SOLUTION_KMEANS = OUT_DIR / "solution_kmeans_model.joblib"

OUT_SUMMARY = OUT_DIR / "time_normalized_cluster_edge_summary.txt"
OUT_CONFIG = OUT_DIR / "time_normalized_cluster_edge_config.json"

OUT_EXACT_CUTOFFS = OUT_DIR / "exact_edge_cutoff_counts.csv"
OUT_CLUSTER_CUTOFFS = OUT_DIR / "cluster_edge_cutoff_counts.csv"

OUT_EXACT_HIST = OUT_DIR / "exact_edge_weight_distribution.png"
OUT_CLUSTER_HIST = OUT_DIR / "cluster_edge_weight_distribution.png"
OUT_CLUSTER_TIME_NORM_HIST = OUT_DIR / "cluster_time_normalized_strength_distribution.png"

OUT_TOP_EXACT = OUT_DIR / "top_exact_edges_by_time_normalized_strength.csv"
OUT_TOP_CLUSTER = OUT_DIR / "top_cluster_edges_by_time_normalized_strength.csv"


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


def safe_year(x):
    try:
        if pd.isna(x):
            return np.nan
        y = int(float(x))
        if 1900 <= y <= 2035:
            return y
        return np.nan
    except Exception:
        return np.nan


def log1p_safe(x):
    try:
        return math.log1p(max(float(x), 0.0))
    except Exception:
        return 0.0


def detect_text_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_node_text_from_edges(edges, id_col, text_col):
    tmp = (
        edges[[id_col, text_col]]
        .dropna()
        .drop_duplicates()
        .groupby(id_col)[text_col]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    return dict(zip(tmp[id_col].astype(str), tmp[text_col].astype(str)))


def add_fallback_text(index_df, edges, id_col, edge_text_col, preferred_cols):
    text_col = detect_text_column(index_df, preferred_cols)

    if text_col is not None:
        out = index_df.copy()
        out["node_text"] = out[text_col].astype(str)
        return out

    print(f"No text column found in index for {id_col}; using edge file text fallback.")
    id_to_text = make_node_text_from_edges(edges, id_col, edge_text_col)

    out = index_df.copy()
    out["node_text"] = out[id_col].astype(str).map(id_to_text)
    out["node_text"] = out["node_text"].fillna(out[id_col].astype(str))
    return out


def fit_clusters(emb, n_clusters, label):
    n_clusters = min(n_clusters, max(2, emb.shape[0] - 1))

    print(f"\nFitting {label} MiniBatchKMeans")
    print(f"Nodes: {emb.shape[0]:,} | dims: {emb.shape[1]:,} | clusters: {n_clusters:,}")

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        max_iter=MAX_ITER,
        n_init=3,
        verbose=0,
        reassignment_ratio=0.01,
    )

    labels = model.fit_predict(emb)

    print(f"{label} clustering done.")
    print(f"Inertia: {model.inertia_:,.2f}")

    return model, labels


def make_cluster_names(node_df, id_col, cluster_col, text_col="node_text", top_n=5):
    names = {}

    for cluster_id, g in node_df.groupby(cluster_col):
        texts = (
            g[text_col]
            .astype(str)
            .replace("nan", np.nan)
            .dropna()
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )

        if texts:
            names[cluster_id] = "; ".join(texts)
        else:
            names[cluster_id] = f"cluster_{cluster_id}"

    return names


def add_time_normalization(df, weight_col, first_year_col, last_year_col, end_year, prefix):
    out = df.copy()

    out[first_year_col] = pd.to_numeric(out[first_year_col], errors="coerce")
    out[last_year_col] = pd.to_numeric(out[last_year_col], errors="coerce")
    out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce").fillna(1.0)

    out[f"{prefix}_age_since_first_years"] = end_year - out[first_year_col] + 1
    out.loc[out[f"{prefix}_age_since_first_years"] < 1, f"{prefix}_age_since_first_years"] = np.nan

    out[f"{prefix}_active_span_years"] = out[last_year_col] - out[first_year_col] + 1
    out.loc[out[f"{prefix}_active_span_years"] < 1, f"{prefix}_active_span_years"] = 1

    out[f"{prefix}_log_weight"] = np.log1p(out[weight_col])
    out[f"{prefix}_annualized_since_first"] = out[weight_col] / out[f"{prefix}_age_since_first_years"]
    out[f"{prefix}_annualized_active_span"] = out[weight_col] / out[f"{prefix}_active_span_years"]

    # Conservative time-adjusted strength:
    # rewards recurrence, penalizes simply being old.
    out[f"{prefix}_time_adjusted_strength"] = (
        np.log1p(out[weight_col]) / np.log1p(out[f"{prefix}_age_since_first_years"])
    )

    # Year-cohort normalization:
    # compare an edge only against other edges first appearing in the same year.
    cohort = (
        out.dropna(subset=[first_year_col])
        .groupby(first_year_col)[weight_col]
        .agg(
            cohort_n="count",
            cohort_mean="mean",
            cohort_median="median",
            cohort_p75=lambda x: np.percentile(x, 75),
            cohort_p90=lambda x: np.percentile(x, 90),
            cohort_p95=lambda x: np.percentile(x, 95),
        )
        .reset_index()
    )

    out = out.merge(cohort, on=first_year_col, how="left")

    out[f"{prefix}_weight_over_cohort_mean"] = out[weight_col] / out["cohort_mean"].replace(0, np.nan)
    out[f"{prefix}_weight_over_cohort_p90"] = out[weight_col] / out["cohort_p90"].replace(0, np.nan)

    # Percentile rank within first-year cohort.
    out[f"{prefix}_within_year_percentile"] = (
        out.groupby(first_year_col)[weight_col]
        .rank(method="average", pct=True)
    )

    return out


def cutoff_table(df, weight_col, strength_col, cutoffs_weight, cutoffs_strength):
    rows = []

    n = len(df)

    for c in cutoffs_weight:
        rows.append({
            "metric": weight_col,
            "cutoff": c,
            "n_ge_cutoff": int((df[weight_col] >= c).sum()),
            "fraction_ge_cutoff": float((df[weight_col] >= c).mean()) if n else np.nan,
        })

    for c in cutoffs_strength:
        rows.append({
            "metric": strength_col,
            "cutoff": c,
            "n_ge_cutoff": int((df[strength_col] >= c).sum()),
            "fraction_ge_cutoff": float((df[strength_col] >= c).mean()) if n else np.nan,
        })

    return pd.DataFrame(rows)


def plot_hist(series, out_path, title, xlabel, log_x=False):
    s = pd.to_numeric(series, errors="coerce").dropna()

    plt.figure(figsize=(7, 5), dpi=300)

    if log_x:
        plt.hist(np.log1p(s), bins=60)
        plt.xlabel(f"log(1 + {xlabel})")
    else:
        plt.hist(s, bins=60)
        plt.xlabel(xlabel)

    plt.ylabel("Number of edges")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def top_examples_text(df, cols, n=30):
    cols = [c for c in cols if c in df.columns]
    return df[cols].head(n).to_string(index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Checking files...")
    for p in [EDGE_FILE, PROBLEM_INDEX_CSV, SOLUTION_INDEX_CSV, PROBLEM_EMB_NPY, SOLUTION_EMB_NPY]:
        require_file(p)

    print("Loading inputs...")
    edges = pd.read_csv(EDGE_FILE, encoding="utf-8-sig", low_memory=False)
    problem_index = pd.read_csv(PROBLEM_INDEX_CSV, encoding="utf-8-sig", low_memory=False)
    solution_index = pd.read_csv(SOLUTION_INDEX_CSV, encoding="utf-8-sig", low_memory=False)

    problem_emb = normalize_matrix(np.load(PROBLEM_EMB_NPY))
    solution_emb = normalize_matrix(np.load(SOLUTION_EMB_NPY))

    print("Edges:", edges.shape)
    print("Problem index:", problem_index.shape)
    print("Solution index:", solution_index.shape)
    print("Problem embeddings:", problem_emb.shape)
    print("Solution embeddings:", solution_emb.shape)

    required_edge_cols = {"problem_id", "solution_id", "surgical_problem", "technical_solution", "weight", "first_year", "last_year"}
    missing = required_edge_cols - set(edges.columns)
    if missing:
        raise ValueError(f"Edge file is missing columns: {missing}")

    edges["problem_id"] = edges["problem_id"].astype(str)
    edges["solution_id"] = edges["solution_id"].astype(str)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(1).astype(int)
    edges["first_year"] = edges["first_year"].apply(safe_year)
    edges["last_year"] = edges["last_year"].apply(safe_year)

    edges = edges.dropna(subset=["first_year", "last_year"]).copy()
    edges["first_year"] = edges["first_year"].astype(int)
    edges["last_year"] = edges["last_year"].astype(int)

    if ANALYSIS_END_YEAR is None:
        end_year = int(max(edges["last_year"].max(), edges["first_year"].max()))
    else:
        end_year = int(ANALYSIS_END_YEAR)

    print(f"Analysis end year: {end_year}")

    # Add node text fallback
    problem_index["problem_id"] = problem_index["problem_id"].astype(str)
    solution_index["solution_id"] = solution_index["solution_id"].astype(str)

    problem_index = add_fallback_text(
        problem_index,
        edges,
        id_col="problem_id",
        edge_text_col="surgical_problem",
        preferred_cols=["surgical_problem", "problem", "problem_text", "problem_key", "node_text", "text"],
    )

    solution_index = add_fallback_text(
        solution_index,
        edges,
        id_col="solution_id",
        edge_text_col="technical_solution",
        preferred_cols=["technical_solution", "solution", "solution_text", "solution_key", "node_text", "text"],
    )

    # Safety: align index lengths and embeddings
    if len(problem_index) != problem_emb.shape[0]:
        raise ValueError(f"Problem index rows {len(problem_index)} != problem embedding rows {problem_emb.shape[0]}")
    if len(solution_index) != solution_emb.shape[0]:
        raise ValueError(f"Solution index rows {len(solution_index)} != solution embedding rows {solution_emb.shape[0]}")

    # --------------------------------------------------------
    # 1. Cluster problem and solution nodes
    # --------------------------------------------------------

    problem_model, problem_cluster = fit_clusters(problem_emb, N_PROBLEM_CLUSTERS, "problem")
    solution_model, solution_cluster = fit_clusters(solution_emb, N_SOLUTION_CLUSTERS, "solution")

    problem_index["problem_cluster_id"] = problem_cluster
    solution_index["solution_cluster_id"] = solution_cluster

    problem_cluster_names = make_cluster_names(problem_index, "problem_id", "problem_cluster_id")
    solution_cluster_names = make_cluster_names(solution_index, "solution_id", "solution_cluster_id")

    problem_index["problem_cluster_name"] = problem_index["problem_cluster_id"].map(problem_cluster_names)
    solution_index["solution_cluster_name"] = solution_index["solution_cluster_id"].map(solution_cluster_names)

    problem_index.to_csv(OUT_PROBLEM_CLUSTERS, index=False, encoding="utf-8-sig")
    solution_index.to_csv(OUT_SOLUTION_CLUSTERS, index=False, encoding="utf-8-sig")

    joblib.dump(problem_model, OUT_PROBLEM_KMEANS)
    joblib.dump(solution_model, OUT_SOLUTION_KMEANS)

    print("Saved node clusters.")

    # --------------------------------------------------------
    # 2. Exact edge time normalization
    # --------------------------------------------------------

    print("\nComputing exact edge time normalization...")

    problem_map = problem_index[["problem_id", "problem_cluster_id", "problem_cluster_name"]]
    solution_map = solution_index[["solution_id", "solution_cluster_id", "solution_cluster_name"]]

    exact_edges = edges.merge(problem_map, on="problem_id", how="left")
    exact_edges = exact_edges.merge(solution_map, on="solution_id", how="left")

    before = len(exact_edges)
    exact_edges = exact_edges.dropna(subset=["problem_cluster_id", "solution_cluster_id"]).copy()
    print(f"Exact edges with cluster assignment: {len(exact_edges):,}/{before:,}")

    exact_edges["problem_cluster_id"] = exact_edges["problem_cluster_id"].astype(int)
    exact_edges["solution_cluster_id"] = exact_edges["solution_cluster_id"].astype(int)

    exact_edges = add_time_normalization(
        exact_edges,
        weight_col="weight",
        first_year_col="first_year",
        last_year_col="last_year",
        end_year=end_year,
        prefix="exact",
    )

    exact_edges.to_csv(OUT_EXACT_EDGES, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 3. Cluster-edge aggregation
    # --------------------------------------------------------

    print("\nAggregating cluster edges...")

    # Representative names
    def join_top_texts(x, top_n=8):
        vals = pd.Series(x.dropna().astype(str)).value_counts().head(top_n).index.tolist()
        return " | ".join(vals)

    cluster_edges = (
        exact_edges
        .groupby(["problem_cluster_id", "solution_cluster_id"], as_index=False)
        .agg(
            cluster_weight=("weight", "sum"),
            n_exact_edges=("pair_id", "count"),
            n_unique_problem_nodes=("problem_id", "nunique"),
            n_unique_solution_nodes=("solution_id", "nunique"),
            first_year=("first_year", "min"),
            last_year=("last_year", "max"),
            problem_cluster_name=("problem_cluster_name", "first"),
            solution_cluster_name=("solution_cluster_name", "first"),
            example_surgical_problems=("surgical_problem", join_top_texts),
            example_technical_solutions=("technical_solution", join_top_texts),
        )
    )

    cluster_edges = add_time_normalization(
        cluster_edges,
        weight_col="cluster_weight",
        first_year_col="first_year",
        last_year_col="last_year",
        end_year=end_year,
        prefix="cluster",
    )

    # Additional specificity measures
    cluster_edges["cluster_edge_density_proxy"] = (
        cluster_edges["n_exact_edges"]
        / (cluster_edges["n_unique_problem_nodes"] * cluster_edges["n_unique_solution_nodes"]).replace(0, np.nan)
    )

    cluster_edges["cluster_weight_per_exact_edge"] = (
        cluster_edges["cluster_weight"] / cluster_edges["n_exact_edges"].replace(0, np.nan)
    )

    cluster_edges = cluster_edges.sort_values(
        ["cluster_time_adjusted_strength", "cluster_weight", "n_exact_edges"],
        ascending=[False, False, False],
    )

    cluster_edges.to_csv(OUT_CLUSTER_EDGES, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 4. Cutoff tables and plots
    # --------------------------------------------------------

    print("\nGenerating cutoff tables and plots...")

    exact_cutoffs = cutoff_table(
        exact_edges,
        weight_col="weight",
        strength_col="exact_time_adjusted_strength",
        cutoffs_weight=[1, 2, 3, 4, 5, 10, 20, 50, 100],
        cutoffs_strength=[0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0],
    )

    cluster_cutoffs = cutoff_table(
        cluster_edges,
        weight_col="cluster_weight",
        strength_col="cluster_time_adjusted_strength",
        cutoffs_weight=[1, 2, 3, 4, 5, 10, 20, 50, 100, 250, 500],
        cutoffs_strength=[0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    )

    exact_cutoffs.to_csv(OUT_EXACT_CUTOFFS, index=False, encoding="utf-8-sig")
    cluster_cutoffs.to_csv(OUT_CLUSTER_CUTOFFS, index=False, encoding="utf-8-sig")

    plot_hist(
        exact_edges["weight"],
        OUT_EXACT_HIST,
        "Exact problem-solution edge weight distribution",
        "exact edge weight",
        log_x=True,
    )

    plot_hist(
        cluster_edges["cluster_weight"],
        OUT_CLUSTER_HIST,
        "Cluster-edge weight distribution",
        "cluster edge weight",
        log_x=True,
    )

    plot_hist(
        cluster_edges["cluster_time_adjusted_strength"],
        OUT_CLUSTER_TIME_NORM_HIST,
        "Time-normalized cluster-edge strength distribution",
        "cluster time-adjusted strength",
        log_x=False,
    )

    # Top edges
    exact_top_cols = [
        "pair_id",
        "surgical_problem",
        "technical_solution",
        "weight",
        "first_year",
        "last_year",
        "exact_age_since_first_years",
        "exact_annualized_since_first",
        "exact_time_adjusted_strength",
        "exact_within_year_percentile",
        "problem_cluster_name",
        "solution_cluster_name",
        "pmids",
    ]

    cluster_top_cols = [
        "problem_cluster_id",
        "solution_cluster_id",
        "problem_cluster_name",
        "solution_cluster_name",
        "cluster_weight",
        "n_exact_edges",
        "n_unique_problem_nodes",
        "n_unique_solution_nodes",
        "first_year",
        "last_year",
        "cluster_age_since_first_years",
        "cluster_annualized_since_first",
        "cluster_time_adjusted_strength",
        "cluster_within_year_percentile",
        "cluster_edge_density_proxy",
        "example_surgical_problems",
        "example_technical_solutions",
    ]

    exact_edges.sort_values(
        ["exact_time_adjusted_strength", "weight"],
        ascending=[False, False],
    )[exact_top_cols].head(500).to_csv(OUT_TOP_EXACT, index=False, encoding="utf-8-sig")

    cluster_edges.sort_values(
        ["cluster_time_adjusted_strength", "cluster_weight"],
        ascending=[False, False],
    )[cluster_top_cols].head(500).to_csv(OUT_TOP_CLUSTER, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------

    summary = []
    summary.append("Time-normalized semantic cluster edge analysis")
    summary.append("=" * 90)
    summary.append("")
    summary.append(f"Input edge file: {EDGE_FILE}")
    summary.append(f"Analysis end year: {end_year}")
    summary.append("")
    summary.append(f"Exact edges after year filter: {len(exact_edges):,}")
    summary.append(f"Problem nodes: {len(problem_index):,}")
    summary.append(f"Solution nodes: {len(solution_index):,}")
    summary.append(f"Problem clusters: {problem_index['problem_cluster_id'].nunique():,}")
    summary.append(f"Solution clusters: {solution_index['solution_cluster_id'].nunique():,}")
    summary.append(f"Cluster edges: {len(cluster_edges):,}")
    summary.append("")
    summary.append("Exact edge raw weight summary:")
    summary.append(str(exact_edges["weight"].describe()))
    summary.append("")
    summary.append("Exact edge time-adjusted strength summary:")
    summary.append(str(exact_edges["exact_time_adjusted_strength"].describe()))
    summary.append("")
    summary.append("Cluster edge raw weight summary:")
    summary.append(str(cluster_edges["cluster_weight"].describe()))
    summary.append("")
    summary.append("Cluster edge time-adjusted strength summary:")
    summary.append(str(cluster_edges["cluster_time_adjusted_strength"].describe()))
    summary.append("")
    summary.append("Exact edge cutoff counts:")
    summary.append(exact_cutoffs.to_string(index=False))
    summary.append("")
    summary.append("Cluster edge cutoff counts:")
    summary.append(cluster_cutoffs.to_string(index=False))
    summary.append("")
    summary.append("Top 30 exact edges by time-adjusted strength:")
    summary.append(
        top_examples_text(
            exact_edges.sort_values(["exact_time_adjusted_strength", "weight"], ascending=[False, False]),
            exact_top_cols,
            n=30,
        )
    )
    summary.append("")
    summary.append("Top 30 cluster edges by time-adjusted strength:")
    summary.append(
        top_examples_text(
            cluster_edges.sort_values(["cluster_time_adjusted_strength", "cluster_weight"], ascending=[False, False]),
            cluster_top_cols,
            n=30,
        )
    )
    summary.append("")
    summary.append("Important interpretation:")
    summary.append("- raw weight is not time-normalized.")
    summary.append("- time_adjusted_strength penalizes edges that had many years to accumulate occurrences.")
    summary.append("- cluster edges aggregate semantically similar problem and solution nodes.")
    summary.append("- exact edges remain available for traceability.")
    summary.append("")
    summary.append("Output files:")
    for p in [
        OUT_PROBLEM_CLUSTERS,
        OUT_SOLUTION_CLUSTERS,
        OUT_EXACT_EDGES,
        OUT_CLUSTER_EDGES,
        OUT_EXACT_CUTOFFS,
        OUT_CLUSTER_CUTOFFS,
        OUT_TOP_EXACT,
        OUT_TOP_CLUSTER,
        OUT_EXACT_HIST,
        OUT_CLUSTER_HIST,
        OUT_CLUSTER_TIME_NORM_HIST,
    ]:
        summary.append(str(p))

    OUT_SUMMARY.write_text("\n".join(summary), encoding="utf-8")

    config = {
        "N_PROBLEM_CLUSTERS": N_PROBLEM_CLUSTERS,
        "N_SOLUTION_CLUSTERS": N_SOLUTION_CLUSTERS,
        "RANDOM_STATE": RANDOM_STATE,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_ITER": MAX_ITER,
        "ANALYSIS_END_YEAR": end_year,
        "input_edge_file": str(EDGE_FILE),
    }
    OUT_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(OUT_SUMMARY)
    print(OUT_PROBLEM_CLUSTERS)
    print(OUT_SOLUTION_CLUSTERS)
    print(OUT_EXACT_EDGES)
    print(OUT_CLUSTER_EDGES)
    print(OUT_TOP_CLUSTER)

    print("\nPreview cluster cutoff table:")
    print(cluster_cutoffs.to_string(index=False))

    print("\nDONE")


if __name__ == "__main__":
    main()