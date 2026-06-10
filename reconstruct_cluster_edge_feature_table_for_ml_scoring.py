from pathlib import Path
import json
import re
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

CLUSTER_EDGES_FILE = (
    BASE_DIR
    / "surgical_time_normalized_cluster_edges"
    / "cluster_edges_time_normalized.csv"
)

PROBLEM_NODE_CLUSTERS_FILE = (
    BASE_DIR
    / "surgical_time_normalized_cluster_edges"
    / "problem_node_clusters.csv"
)

SOLUTION_NODE_CLUSTERS_FILE = (
    BASE_DIR
    / "surgical_time_normalized_cluster_edges"
    / "solution_node_clusters.csv"
)

PROBLEM_INDEX_FILE = (
    BASE_DIR
    / "surgical_training_graph_outputs"
    / "surgical_problem_embedding_index.csv"
)

SOLUTION_INDEX_FILE = (
    BASE_DIR
    / "surgical_training_graph_outputs"
    / "surgical_solution_embedding_index.csv"
)

PROBLEM_EMB_FILE = (
    BASE_DIR
    / "surgical_training_graph_outputs"
    / "surgical_problem_embeddings.npy"
)

SOLUTION_EMB_FILE = (
    BASE_DIR
    / "surgical_training_graph_outputs"
    / "surgical_solution_embeddings.npy"
)

OUT_DIR = BASE_DIR / "cluster_edge_strength_model_outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_FEATURE_TABLE = OUT_DIR / "reconstructed_cluster_edge_feature_table_for_ml_scoring.csv"
OUT_SUMMARY = OUT_DIR / "reconstructed_cluster_edge_feature_table_summary.json"


FEATURE_COLS = [
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


# ============================================================
# HELPERS
# ============================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def find_col(df, candidates, required=False):
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c

    if required:
        raise ValueError(
            f"Could not find required column among {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms


def build_cluster_centroids(index_df, cluster_df, emb, side_name):
    """
    Robustly match embedding index to node clusters and compute cluster centroids.
    Returns:
      cluster_id_to_centroid_norm
      cluster_id_to_size
    """
    print(f"\nBuilding {side_name} cluster centroids...")

    print(f"{side_name} index rows: {len(index_df):,}")
    print(f"{side_name} cluster rows: {len(cluster_df):,}")
    print(f"{side_name} embedding shape: {emb.shape}")
    print(f"{side_name} index columns: {list(index_df.columns)}")
    print(f"{side_name} cluster columns: {list(cluster_df.columns)}")

    if len(index_df) != emb.shape[0]:
        raise ValueError(f"{side_name} index rows and embedding rows do not match.")

    cluster_col = find_col(
        cluster_df,
        [
            f"{side_name}_cluster_id",
            "cluster_id",
            f"{side_name}_cluster",
        ],
        required=True,
    )

    id_col_index = find_col(
        index_df,
        [
            f"{side_name}_id",
            "node_id",
            "id",
        ],
        required=False,
    )

    id_col_cluster = find_col(
        cluster_df,
        [
            f"{side_name}_id",
            "node_id",
            "id",
        ],
        required=False,
    )

    text_col_index = find_col(
        index_df,
        [
            f"{side_name}_text",
            f"{side_name}",
            "node_text",
            "concept_text",
            "text",
        ],
        required=False,
    )

    text_col_cluster = find_col(
        cluster_df,
        [
            f"{side_name}_text",
            f"{side_name}",
            "node_text",
            "concept_text",
            "text",
        ],
        required=False,
    )

    print(f"Detected {side_name} columns:")
    print("  cluster_col:", cluster_col)
    print("  id_col_index:", id_col_index)
    print("  id_col_cluster:", id_col_cluster)
    print("  text_col_index:", text_col_index)
    print("  text_col_cluster:", text_col_cluster)

    idx = index_df.copy()
    idx["_row_idx"] = np.arange(len(idx))

    cl = cluster_df.copy()
    cl["_cluster_id"] = cl[cluster_col].astype(int)

    merged = None

    # Prefer ID merge
    if id_col_index and id_col_cluster:
        idx["_join_id"] = idx[id_col_index].astype(str)
        cl["_join_id"] = cl[id_col_cluster].astype(str)

        merged = idx.merge(
            cl[["_join_id", "_cluster_id"]],
            on="_join_id",
            how="inner",
        )

        print(f"{side_name}: merged by ID rows: {len(merged):,}")

    # Fallback text merge
    if (merged is None or len(merged) == 0) and text_col_index and text_col_cluster:
        idx["_join_text"] = idx[text_col_index].apply(clean_text).str.lower()
        cl["_join_text"] = cl[text_col_cluster].apply(clean_text).str.lower()

        merged = idx.merge(
            cl[["_join_text", "_cluster_id"]],
            on="_join_text",
            how="inner",
        )

        print(f"{side_name}: merged by text rows: {len(merged):,}")

    # Fallback row order
    if (merged is None or len(merged) == 0):
        if len(index_df) != len(cluster_df):
            raise ValueError(
                f"Could not merge {side_name} index to clusters by ID/text, "
                f"and row counts differ: index {len(index_df)}, clusters {len(cluster_df)}"
            )

        merged = idx.copy()
        merged["_cluster_id"] = cl["_cluster_id"].values

        print(f"{side_name}: using row-order fallback rows: {len(merged):,}")

    if len(merged) == 0:
        raise ValueError(f"No {side_name} rows matched to clusters.")

    cluster_to_rows = (
        merged
        .groupby("_cluster_id")["_row_idx"]
        .apply(list)
        .to_dict()
    )

    cluster_ids = sorted(cluster_to_rows.keys())

    centroid_dict = {}
    size_dict = {}

    for cid in cluster_ids:
        rows = cluster_to_rows[cid]
        mat = np.array(emb[rows], dtype=np.float32)
        centroid = mat.mean(axis=0)
        centroid_norm = normalize_rows(centroid.reshape(1, -1))[0]

        centroid_dict[int(cid)] = centroid_norm
        size_dict[int(cid)] = int(len(rows))

    print(f"{side_name}: cluster centroids built: {len(centroid_dict):,}")

    return centroid_dict, size_dict


def get_weight_col(df):
    return find_col(
        df,
        [
            "cluster_weight",
            "weight",
            "edge_weight",
            "n_edges",
            "count",
        ],
        required=False,
    )


def get_exact_col(df):
    return find_col(
        df,
        [
            "n_exact_edges",
            "total_exact_edges",
            "exact_edges",
            "n_pairs",
        ],
        required=False,
    )


# ============================================================
# MAIN
# ============================================================

print("============================================================")
print("Reconstruct cluster-edge feature table for ML scoring")
print("============================================================")

# Load files
for f in [
    CLUSTER_EDGES_FILE,
    PROBLEM_NODE_CLUSTERS_FILE,
    SOLUTION_NODE_CLUSTERS_FILE,
    PROBLEM_INDEX_FILE,
    SOLUTION_INDEX_FILE,
    PROBLEM_EMB_FILE,
    SOLUTION_EMB_FILE,
]:
    if not f.exists():
        raise FileNotFoundError(f)

cluster_edges = pd.read_csv(CLUSTER_EDGES_FILE, encoding="utf-8-sig", low_memory=False)
problem_clusters = pd.read_csv(PROBLEM_NODE_CLUSTERS_FILE, encoding="utf-8-sig", low_memory=False)
solution_clusters = pd.read_csv(SOLUTION_NODE_CLUSTERS_FILE, encoding="utf-8-sig", low_memory=False)

problem_index = pd.read_csv(PROBLEM_INDEX_FILE, encoding="utf-8-sig", low_memory=False)
solution_index = pd.read_csv(SOLUTION_INDEX_FILE, encoding="utf-8-sig", low_memory=False)

problem_emb = np.load(PROBLEM_EMB_FILE, mmap_mode="r")
solution_emb = np.load(SOLUTION_EMB_FILE, mmap_mode="r")

print(f"Cluster edges rows: {len(cluster_edges):,}")
print("Cluster edge columns:", list(cluster_edges.columns))

problem_cluster_col = find_col(
    cluster_edges,
    ["problem_cluster_id", "source_problem_cluster_id", "problem_cluster"],
    required=True,
)

solution_cluster_col = find_col(
    cluster_edges,
    ["solution_cluster_id", "target_solution_cluster_id", "solution_cluster"],
    required=True,
)

weight_col = get_weight_col(cluster_edges)
exact_col = get_exact_col(cluster_edges)

density_col_existing = find_col(
    cluster_edges,
    ["cluster_edge_density_proxy", "edge_density_proxy", "density_proxy"],
    required=False,
)

print("Detected cluster edge columns:")
print("  problem_cluster_col:", problem_cluster_col)
print("  solution_cluster_col:", solution_cluster_col)
print("  weight_col:", weight_col)
print("  exact_col:", exact_col)
print("  density_col_existing:", density_col_existing)

df = cluster_edges.copy()
df["problem_cluster_id"] = df[problem_cluster_col].astype(int)
df["solution_cluster_id"] = df[solution_cluster_col].astype(int)

if weight_col:
    df["_cluster_weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
else:
    df["_cluster_weight"] = 1.0

if exact_col:
    df["_n_exact_edges"] = pd.to_numeric(df[exact_col], errors="coerce").fillna(1.0)
else:
    df["_n_exact_edges"] = 1.0

# Build centroids and sizes
problem_centroids, problem_sizes = build_cluster_centroids(
    problem_index,
    problem_clusters,
    problem_emb,
    side_name="problem",
)

solution_centroids, solution_sizes = build_cluster_centroids(
    solution_index,
    solution_clusters,
    solution_emb,
    side_name="solution",
)

# Graph topology summaries
print("\nComputing graph topology features...")

problem_degree = df.groupby("problem_cluster_id")["solution_cluster_id"].nunique().to_dict()
solution_degree = df.groupby("solution_cluster_id")["problem_cluster_id"].nunique().to_dict()

problem_total_weight = df.groupby("problem_cluster_id")["_cluster_weight"].sum().to_dict()
solution_total_weight = df.groupby("solution_cluster_id")["_cluster_weight"].sum().to_dict()

problem_total_exact = df.groupby("problem_cluster_id")["_n_exact_edges"].sum().to_dict()
solution_total_exact = df.groupby("solution_cluster_id")["_n_exact_edges"].sum().to_dict()

# Feature reconstruction
print("\nReconstructing feature columns...")

centroid_cosines = []
problem_size_logs = []
solution_size_logs = []

problem_degree_wo_logs = []
solution_degree_wo_logs = []

problem_weight_wo_logs = []
solution_weight_wo_logs = []

problem_exact_wo_logs = []
solution_exact_wo_logs = []

density_values = []

missing_centroids = 0

for _, row in df.iterrows():
    pc = int(row["problem_cluster_id"])
    sc = int(row["solution_cluster_id"])

    p_cent = problem_centroids.get(pc)
    s_cent = solution_centroids.get(sc)

    if p_cent is None or s_cent is None:
        centroid_cosines.append(0.0)
        missing_centroids += 1
    else:
        centroid_cosines.append(float(np.dot(p_cent, s_cent)))

    p_size = int(problem_sizes.get(pc, 1))
    s_size = int(solution_sizes.get(sc, 1))

    problem_size_logs.append(float(np.log1p(p_size)))
    solution_size_logs.append(float(np.log1p(s_size)))

    p_deg_wo = max(float(problem_degree.get(pc, 1)) - 1.0, 0.0)
    s_deg_wo = max(float(solution_degree.get(sc, 1)) - 1.0, 0.0)

    problem_degree_wo_logs.append(float(np.log1p(p_deg_wo)))
    solution_degree_wo_logs.append(float(np.log1p(s_deg_wo)))

    w = float(row["_cluster_weight"])
    ex = float(row["_n_exact_edges"])

    p_w_wo = max(float(problem_total_weight.get(pc, w)) - w, 0.0)
    s_w_wo = max(float(solution_total_weight.get(sc, w)) - w, 0.0)

    p_ex_wo = max(float(problem_total_exact.get(pc, ex)) - ex, 0.0)
    s_ex_wo = max(float(solution_total_exact.get(sc, ex)) - ex, 0.0)

    problem_weight_wo_logs.append(float(np.log1p(p_w_wo)))
    solution_weight_wo_logs.append(float(np.log1p(s_w_wo)))

    problem_exact_wo_logs.append(float(np.log1p(p_ex_wo)))
    solution_exact_wo_logs.append(float(np.log1p(s_ex_wo)))

    if density_col_existing:
        density = pd.to_numeric(row[density_col_existing], errors="coerce")
        if pd.isna(density):
            density = 0.0
        density_values.append(float(density))
    else:
        # Fallback density proxy:
        # How dense is this edge relative to possible node combinations in both clusters?
        density = ex / max(p_size * s_size, 1)
        density_values.append(float(density))

df["centroid_cosine"] = centroid_cosines
df["problem_cluster_size_log"] = problem_size_logs
df["solution_cluster_size_log"] = solution_size_logs
df["problem_degree_without_edge_log"] = problem_degree_wo_logs
df["solution_degree_without_edge_log"] = solution_degree_wo_logs
df["problem_total_weight_without_edge_log"] = problem_weight_wo_logs
df["solution_total_weight_without_edge_log"] = solution_weight_wo_logs
df["problem_total_exact_edges_without_edge_log"] = problem_exact_wo_logs
df["solution_total_exact_edges_without_edge_log"] = solution_exact_wo_logs
df["cluster_edge_density_proxy"] = density_values

# Sanity checks
for c in FEATURE_COLS:
    if c not in df.columns:
        raise ValueError(f"Missing reconstructed feature: {c}")

    df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Keep useful columns + all original columns if wanted
out_cols = [
    "problem_cluster_id",
    "solution_cluster_id",
] + FEATURE_COLS

# Optional metadata columns
for c in [
    "cluster_weight",
    "weight",
    "n_exact_edges",
    "first_year",
    "last_year",
    "cluster_time_adjusted_strength",
    "cluster_within_year_percentile",
    "final_edge_strength",
]:
    if c in df.columns and c not in out_cols:
        out_cols.append(c)

out = df[out_cols].copy()

out.to_csv(OUT_FEATURE_TABLE, index=False, encoding="utf-8-sig")

summary = {
    "cluster_edges_file": str(CLUSTER_EDGES_FILE),
    "problem_node_clusters_file": str(PROBLEM_NODE_CLUSTERS_FILE),
    "solution_node_clusters_file": str(SOLUTION_NODE_CLUSTERS_FILE),
    "problem_index_file": str(PROBLEM_INDEX_FILE),
    "solution_index_file": str(SOLUTION_INDEX_FILE),
    "problem_embedding_file": str(PROBLEM_EMB_FILE),
    "solution_embedding_file": str(SOLUTION_EMB_FILE),
    "n_cluster_edges": int(len(df)),
    "n_output_rows": int(len(out)),
    "missing_centroids": int(missing_centroids),
    "density_col_existing": density_col_existing,
    "features": FEATURE_COLS,
    "output_feature_table": str(OUT_FEATURE_TABLE),
}

with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("============================================================")
print("DONE")
print("Feature table saved:")
print(OUT_FEATURE_TABLE)
print("Summary:")
print(OUT_SUMMARY)
print("Missing centroids:", missing_centroids)
print("============================================================")