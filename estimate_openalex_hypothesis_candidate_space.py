from pathlib import Path
import json
import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

# Surgical cluster files
CLUSTER_EDGES_FILE = (
    BASE_DIR
    / "surgical_time_normalized_cluster_edges"
    / "cluster_edges_time_normalized.csv"
)

SOLUTION_NODE_CLUSTERS_FILE = (
    BASE_DIR
    / "surgical_time_normalized_cluster_edges"
    / "solution_node_clusters.csv"
)

# Surgical solution embeddings
SURGICAL_SOLUTION_INDEX_FILE = (
    BASE_DIR
    / "surgical_training_graph_outputs"
    / "surgical_solution_embedding_index.csv"
)

SURGICAL_SOLUTION_EMB_FILE = (
    BASE_DIR
    / "surgical_training_graph_outputs"
    / "surgical_solution_embeddings.npy"
)

# OpenAlex concepts
OPENALEX_INDEX_FILE = (
    BASE_DIR
    / "openalex_engineering_mechanism_embeddings_cached"
    / "openalex_engineering_mechanism_index.csv"
)

OPENALEX_EMB_FILE = (
    BASE_DIR
    / "openalex_engineering_mechanism_embeddings_cached"
    / "openalex_engineering_mechanism_embeddings.npy"
)

# Outputs
OUT_DIR = BASE_DIR / "hypothesis_candidate_space_outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_SUMMARY_JSON = OUT_DIR / "candidate_space_summary.json"
OUT_FUNNEL_CSV = OUT_DIR / "candidate_space_funnel.csv"
OUT_CONCEPT_COUNTS_CSV = OUT_DIR / "openalex_concept_candidate_counts.csv"
OUT_SOLUTION_CLUSTER_MAP_CSV = OUT_DIR / "openalex_to_surgical_solution_cluster_topk.csv"
OUT_PROBLEM_CLUSTER_COUNTS_CSV = OUT_DIR / "problem_cluster_candidate_counts.csv"

# Projection parameters
TOP_K_SOLUTION_CLUSTERS = 10

# Chunk size for OpenAlex embeddings.
# 2000 is usually OK. If RAM is tight, set 500 or 1000.
OPENALEX_CHUNK_SIZE = 2000

# Save full OpenAlex -> solution_cluster top-k mapping?
# For 479,686 concepts and k=10, this creates ~4.8 million rows.
SAVE_FULL_TOPK_MAPPING = True

# Similarity thresholds only for counting sensitivity.
# None means no similarity filter.
SIMILARITY_THRESHOLDS = [None, 0.20, 0.25, 0.30, 0.35]


# ============================================================
# HELPERS
# ============================================================

def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = False):
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
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms


def topk_cosine_to_centroids(query_emb_norm, centroid_emb_norm, top_k: int):
    """
    Chunk-local top-k cosine against solution-cluster centroids.
    Returns indices and similarities.
    """
    sims = query_emb_norm @ centroid_emb_norm.T

    if top_k >= sims.shape[1]:
        idx = np.argsort(-sims, axis=1)
    else:
        idx_part = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
        row_indices = np.arange(sims.shape[0])[:, None]
        idx_sorted_local = np.argsort(-sims[row_indices, idx_part], axis=1)
        idx = idx_part[row_indices, idx_sorted_local]

    row_indices = np.arange(sims.shape[0])[:, None]
    vals = sims[row_indices, idx]

    return idx, vals


# ============================================================
# LOAD SURGICAL GRAPH
# ============================================================

print("============================================================")
print("Estimating OpenAlex → surgical problem candidate space")
print("============================================================")

print("Loading surgical cluster edges...")

if not CLUSTER_EDGES_FILE.exists():
    raise FileNotFoundError(CLUSTER_EDGES_FILE)

cluster_edges = pd.read_csv(CLUSTER_EDGES_FILE, encoding="utf-8-sig", low_memory=False)

print(f"Cluster edge rows: {len(cluster_edges):,}")
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

cluster_weight_col = find_col(
    cluster_edges,
    ["final_edge_strength", "cluster_weight", "weight", "n_exact_edges"],
    required=False,
)

cluster_edges["_problem_cluster_id"] = cluster_edges[problem_cluster_col].astype(int)
cluster_edges["_solution_cluster_id"] = cluster_edges[solution_cluster_col].astype(int)

if cluster_weight_col:
    cluster_edges["_edge_weight"] = pd.to_numeric(cluster_edges[cluster_weight_col], errors="coerce").fillna(0)
else:
    cluster_edges["_edge_weight"] = 1.0

n_problem_clusters_all = cluster_edges["_problem_cluster_id"].nunique()
n_solution_clusters_in_edges = cluster_edges["_solution_cluster_id"].nunique()
n_historical_cluster_edges = len(cluster_edges)

print(f"Problem clusters in historical graph: {n_problem_clusters_all:,}")
print(f"Solution clusters in historical graph: {n_solution_clusters_in_edges:,}")
print(f"Historical cluster edges: {n_historical_cluster_edges:,}")

# solution_cluster -> connected problem_clusters
solution_to_problem_clusters = defaultdict(set)
problem_counter = Counter()

for _, row in cluster_edges.iterrows():
    pc = int(row["_problem_cluster_id"])
    sc = int(row["_solution_cluster_id"])
    solution_to_problem_clusters[sc].add(pc)
    problem_counter[pc] += 1

solution_cluster_problem_counts = {
    sc: len(pcs)
    for sc, pcs in solution_to_problem_clusters.items()
}


# ============================================================
# LOAD SOLUTION CLUSTERS + SOLUTION EMBEDDINGS
# ============================================================

print("\nLoading surgical solution node clusters...")

if not SOLUTION_NODE_CLUSTERS_FILE.exists():
    raise FileNotFoundError(SOLUTION_NODE_CLUSTERS_FILE)

solution_clusters = pd.read_csv(SOLUTION_NODE_CLUSTERS_FILE, encoding="utf-8-sig", low_memory=False)

print(f"Solution node cluster rows: {len(solution_clusters):,}")
print("Solution node cluster columns:", list(solution_clusters.columns))

print("\nLoading surgical solution index and embeddings...")

if not SURGICAL_SOLUTION_INDEX_FILE.exists():
    raise FileNotFoundError(SURGICAL_SOLUTION_INDEX_FILE)

if not SURGICAL_SOLUTION_EMB_FILE.exists():
    raise FileNotFoundError(SURGICAL_SOLUTION_EMB_FILE)

solution_index = pd.read_csv(SURGICAL_SOLUTION_INDEX_FILE, encoding="utf-8-sig", low_memory=False)
solution_emb = np.load(SURGICAL_SOLUTION_EMB_FILE, mmap_mode="r")

print(f"Solution index rows: {len(solution_index):,}")
print(f"Solution embedding shape: {solution_emb.shape}")
print("Solution index columns:", list(solution_index.columns))

if len(solution_index) != solution_emb.shape[0]:
    raise ValueError("solution_index rows and solution_embeddings rows do not match.")

# Detect IDs/text
solution_index_id_col = find_col(
    solution_index,
    ["solution_id", "id", "node_id", "technical_solution_id"],
    required=True,
)

solution_index_text_col = find_col(
    solution_index,
    ["solution_text", "technical_solution", "solution", "node_text", "concept_text", "text"],
    required=False,
)

solution_cluster_id_col = find_col(
    solution_clusters,
    ["solution_cluster_id", "cluster_id", "solution_cluster"],
    required=True,
)

solution_cluster_node_id_col = find_col(
    solution_clusters,
    ["solution_id", "node_id", "id", "technical_solution_id"],
    required=False,
)

solution_cluster_text_col = find_col(
    solution_clusters,
    ["solution_text", "technical_solution", "solution", "node_text", "concept_text", "text"],
    required=False,
)

print("Detected columns:")
print("  solution_index_id_col:", solution_index_id_col)
print("  solution_index_text_col:", solution_index_text_col)
print("  solution_cluster_id_col:", solution_cluster_id_col)
print("  solution_cluster_node_id_col:", solution_cluster_node_id_col)
print("  solution_cluster_text_col:", solution_cluster_text_col)

solution_index["_solution_id"] = solution_index[solution_index_id_col].astype(str)
solution_index["_row_idx"] = np.arange(len(solution_index))

if solution_index_text_col:
    solution_index["_solution_text"] = solution_index[solution_index_text_col].apply(clean_text)
else:
    solution_index["_solution_text"] = solution_index["_solution_id"]

solution_clusters["_solution_cluster_id"] = solution_clusters[solution_cluster_id_col].astype(int)

# Merge cluster assignment to solution index
if solution_cluster_node_id_col:
    solution_clusters["_solution_id"] = solution_clusters[solution_cluster_node_id_col].astype(str)

    solution_index_with_cluster = solution_index.merge(
        solution_clusters[["_solution_id", "_solution_cluster_id"]],
        on="_solution_id",
        how="inner",
    )
else:
    # Fallback: assume same row order
    if len(solution_clusters) != len(solution_index):
        raise ValueError(
            "Could not find solution_id in solution_node_clusters.csv and row counts differ. "
            "Need to inspect solution_node_clusters columns."
        )

    solution_index_with_cluster = solution_index.copy()
    solution_index_with_cluster["_solution_cluster_id"] = solution_clusters["_solution_cluster_id"].values

print(f"Solution rows with cluster assignment: {len(solution_index_with_cluster):,}")

if len(solution_index_with_cluster) == 0:
    raise ValueError("No solution embeddings matched to solution clusters.")

# ============================================================
# BUILD SOLUTION-CLUSTER CENTROIDS
# ============================================================

print("\nBuilding surgical solution-cluster centroids...")

cluster_to_rows = (
    solution_index_with_cluster
    .groupby("_solution_cluster_id")["_row_idx"]
    .apply(list)
    .to_dict()
)

solution_cluster_ids = sorted(cluster_to_rows.keys())
centroid_rows = []

for sc in solution_cluster_ids:
    rows = cluster_to_rows[sc]
    mat = np.array(solution_emb[rows], dtype=np.float32)
    centroid = mat.mean(axis=0)
    centroid_rows.append(centroid)

solution_cluster_centroids = np.vstack(centroid_rows).astype(np.float32)
solution_cluster_centroids_norm = normalize_rows(solution_cluster_centroids)

solution_cluster_id_to_centroid_idx = {
    sc: i for i, sc in enumerate(solution_cluster_ids)
}

centroid_idx_to_solution_cluster_id = {
    i: sc for i, sc in enumerate(solution_cluster_ids)
}

n_solution_clusters_with_centroids = len(solution_cluster_ids)

print(f"Solution clusters with centroids: {n_solution_clusters_with_centroids:,}")


# ============================================================
# LOAD OPENALEX
# ============================================================

print("\nLoading OpenAlex concepts and embeddings...")

if not OPENALEX_INDEX_FILE.exists():
    raise FileNotFoundError(OPENALEX_INDEX_FILE)

if not OPENALEX_EMB_FILE.exists():
    raise FileNotFoundError(OPENALEX_EMB_FILE)

openalex_index = pd.read_csv(OPENALEX_INDEX_FILE, encoding="utf-8-sig")
openalex_emb = np.load(OPENALEX_EMB_FILE, mmap_mode="r")

print(f"OpenAlex concepts: {len(openalex_index):,}")
print(f"OpenAlex embeddings: {openalex_emb.shape}")
print("OpenAlex columns:", list(openalex_index.columns))

if len(openalex_index) != openalex_emb.shape[0]:
    raise ValueError("OpenAlex index and embeddings do not match.")

required = {"concept_id", "concept_text"}
missing = required - set(openalex_index.columns)

if missing:
    raise ValueError(f"OpenAlex index missing required columns: {missing}")

if "n_works" not in openalex_index.columns:
    openalex_index["n_works"] = np.nan

if "n_mentions" not in openalex_index.columns:
    openalex_index["n_mentions"] = np.nan

n_openalex_concepts_all = len(openalex_index)

print(f"All OpenAlex concepts included: {n_openalex_concepts_all:,}")


# ============================================================
# COUNT NAIVE UNIVERSE
# ============================================================

naive_problem_cluster_x_openalex = int(n_problem_clusters_all) * int(n_openalex_concepts_all)

print("\nNaive candidate universe:")
print(f"problem_clusters × OpenAlex concepts = {naive_problem_cluster_x_openalex:,}")


# ============================================================
# PROJECT OPENALEX TO SOLUTION CLUSTERS AND COUNT CANDIDATES
# ============================================================

print("\nProjecting all OpenAlex concepts to surgical solution-cluster centroids...")
print(f"TOP_K_SOLUTION_CLUSTERS = {TOP_K_SOLUTION_CLUSTERS}")
print(f"OPENALEX_CHUNK_SIZE = {OPENALEX_CHUNK_SIZE}")

# Prepare output mapping file
if SAVE_FULL_TOPK_MAPPING and OUT_SOLUTION_CLUSTER_MAP_CSV.exists():
    OUT_SOLUTION_CLUSTER_MAP_CSV.unlink()

concept_count_rows = []
problem_candidate_counter = Counter()

# Threshold-level total counters
threshold_summaries = {
    str(thr): {
        "threshold": thr,
        "n_openalex_with_any_solution_cluster": 0,
        "n_openalex_with_any_problem_candidate": 0,
        "total_solution_cluster_links": 0,
        "total_candidate_problem_concept_pairs": 0,
        "mean_candidate_problems_per_openalex": 0.0,
        "median_candidate_problems_per_openalex": 0.0,
        "max_candidate_problems_per_openalex": 0,
    }
    for thr in SIMILARITY_THRESHOLDS
}

threshold_candidate_counts_per_concept = {
    str(thr): []
    for thr in SIMILARITY_THRESHOLDS
}

n = n_openalex_concepts_all

for start in range(0, n, OPENALEX_CHUNK_SIZE):
    end = min(start + OPENALEX_CHUNK_SIZE, n)

    print(f"Processing OpenAlex concepts {start:,}–{end:,} / {n:,}")

    q = np.array(openalex_emb[start:end], dtype=np.float32)
    q_norm = normalize_rows(q)

    top_idx, top_sim = topk_cosine_to_centroids(
        q_norm,
        solution_cluster_centroids_norm,
        TOP_K_SOLUTION_CLUSTERS,
    )

    mapping_rows = []

    for local_i in range(end - start):
        global_i = start + local_i

        concept_id = int(openalex_index.iloc[global_i]["concept_id"])
        concept_text = openalex_index.iloc[global_i]["concept_text"]
        n_works = openalex_index.iloc[global_i].get("n_works", np.nan)
        n_mentions = openalex_index.iloc[global_i].get("n_mentions", np.nan)

        # full top-k mapping rows
        if SAVE_FULL_TOPK_MAPPING:
            for rank in range(TOP_K_SOLUTION_CLUSTERS):
                centroid_idx = int(top_idx[local_i, rank])
                sc = int(centroid_idx_to_solution_cluster_id[centroid_idx])
                sim = float(top_sim[local_i, rank])
                n_connected_problems = int(solution_cluster_problem_counts.get(sc, 0))

                mapping_rows.append({
                    "concept_id": concept_id,
                    "concept_text": concept_text,
                    "n_works": n_works,
                    "n_mentions": n_mentions,
                    "rank": rank + 1,
                    "solution_cluster_id": sc,
                    "cosine_to_solution_cluster_centroid": sim,
                    "n_connected_problem_clusters": n_connected_problems,
                })

        # candidate counts per threshold
        for thr in SIMILARITY_THRESHOLDS:
            key = str(thr)
            linked_solution_clusters = []

            for rank in range(TOP_K_SOLUTION_CLUSTERS):
                centroid_idx = int(top_idx[local_i, rank])
                sc = int(centroid_idx_to_solution_cluster_id[centroid_idx])
                sim = float(top_sim[local_i, rank])

                if thr is not None and sim < thr:
                    continue

                linked_solution_clusters.append(sc)

            linked_solution_clusters = list(dict.fromkeys(linked_solution_clusters))

            candidate_problem_clusters = set()

            for sc in linked_solution_clusters:
                candidate_problem_clusters.update(solution_to_problem_clusters.get(sc, set()))

            n_candidate_problems = len(candidate_problem_clusters)

            threshold_candidate_counts_per_concept[key].append(n_candidate_problems)

            threshold_summaries[key]["total_solution_cluster_links"] += len(linked_solution_clusters)
            threshold_summaries[key]["total_candidate_problem_concept_pairs"] += n_candidate_problems

            if len(linked_solution_clusters) > 0:
                threshold_summaries[key]["n_openalex_with_any_solution_cluster"] += 1

            if n_candidate_problems > 0:
                threshold_summaries[key]["n_openalex_with_any_problem_candidate"] += 1

            # For no-threshold main per-concept table
            if thr is None:
                for pc in candidate_problem_clusters:
                    problem_candidate_counter[int(pc)] += 1

                concept_count_rows.append({
                    "concept_id": concept_id,
                    "concept_text": concept_text,
                    "n_works": n_works,
                    "n_mentions": n_mentions,
                    "n_linked_solution_clusters_topk": len(linked_solution_clusters),
                    "n_candidate_problem_clusters": n_candidate_problems,
                    "best_solution_cluster_cosine": float(top_sim[local_i, 0]),
                    "mean_topk_solution_cluster_cosine": float(np.mean(top_sim[local_i, :])),
                    "top_solution_clusters": " | ".join(
                        str(int(centroid_idx_to_solution_cluster_id[int(top_idx[local_i, rank])]))
                        for rank in range(TOP_K_SOLUTION_CLUSTERS)
                    ),
                })

    if SAVE_FULL_TOPK_MAPPING and mapping_rows:
        mapping_df = pd.DataFrame(mapping_rows)
        write_header = not OUT_SOLUTION_CLUSTER_MAP_CSV.exists()
        mapping_df.to_csv(
            OUT_SOLUTION_CLUSTER_MAP_CSV,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8-sig",
        )

# finalize threshold summaries
for thr in SIMILARITY_THRESHOLDS:
    key = str(thr)
    counts = np.array(threshold_candidate_counts_per_concept[key], dtype=np.int64)

    if len(counts) > 0:
        threshold_summaries[key]["mean_candidate_problems_per_openalex"] = float(np.mean(counts))
        threshold_summaries[key]["median_candidate_problems_per_openalex"] = float(np.median(counts))
        threshold_summaries[key]["max_candidate_problems_per_openalex"] = int(np.max(counts))

# save per-concept counts
concept_counts_df = pd.DataFrame(concept_count_rows)
concept_counts_df.to_csv(OUT_CONCEPT_COUNTS_CSV, index=False, encoding="utf-8-sig")

problem_counts_df = pd.DataFrame([
    {
        "problem_cluster_id": pc,
        "n_openalex_concepts_candidate": n_cands,
    }
    for pc, n_cands in problem_candidate_counter.items()
]).sort_values("n_openalex_concepts_candidate", ascending=False)

problem_counts_df.to_csv(OUT_PROBLEM_CLUSTER_COUNTS_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# FUNNEL + SUMMARY
# ============================================================

funnel_rows = []

funnel_rows.append({
    "step": "raw_openalex_concepts",
    "count": int(n_openalex_concepts_all),
    "note": "All OpenAlex concepts included; no n_works frequency filter."
})

funnel_rows.append({
    "step": "historical_surgical_problem_clusters",
    "count": int(n_problem_clusters_all),
    "note": "Unique problem clusters in historical surgical cluster-edge graph."
})

funnel_rows.append({
    "step": "naive_problem_cluster_x_openalex",
    "count": int(naive_problem_cluster_x_openalex),
    "note": "Naive Cartesian product before framework reduction."
})

for thr in SIMILARITY_THRESHOLDS:
    key = str(thr)
    label = "no_similarity_threshold" if thr is None else f"cosine_ge_{thr}"

    funnel_rows.append({
        "step": f"framework_candidates_{label}",
        "count": int(threshold_summaries[key]["total_candidate_problem_concept_pairs"]),
        "note": (
            f"Candidate problem_cluster + OpenAlex concept pairs after projecting each OpenAlex concept "
            f"to top-{TOP_K_SOLUTION_CLUSTERS} surgical solution clusters and retaining historically connected problem clusters."
        )
    })

funnel_df = pd.DataFrame(funnel_rows)
funnel_df.to_csv(OUT_FUNNEL_CSV, index=False, encoding="utf-8-sig")

summary = {
    "method": (
        "All OpenAlex concepts are projected to top-k surgical solution-cluster centroids. "
        "A problem_cluster + OpenAlex concept candidate exists if the projected solution_cluster "
        "is historically connected to the problem_cluster in the surgical cluster-edge graph."
    ),
    "no_openalex_frequency_filter": True,
    "top_k_solution_clusters": int(TOP_K_SOLUTION_CLUSTERS),
    "openalex_chunk_size": int(OPENALEX_CHUNK_SIZE),
    "n_openalex_concepts_all": int(n_openalex_concepts_all),
    "n_problem_clusters_all": int(n_problem_clusters_all),
    "n_solution_clusters_in_historical_edges": int(n_solution_clusters_in_edges),
    "n_solution_clusters_with_centroids": int(n_solution_clusters_with_centroids),
    "n_historical_cluster_edges": int(n_historical_cluster_edges),
    "naive_problem_cluster_x_openalex": int(naive_problem_cluster_x_openalex),
    "threshold_summaries": threshold_summaries,
    "outputs": {
        "funnel_csv": str(OUT_FUNNEL_CSV),
        "concept_candidate_counts_csv": str(OUT_CONCEPT_COUNTS_CSV),
        "problem_cluster_candidate_counts_csv": str(OUT_PROBLEM_CLUSTER_COUNTS_CSV),
        "openalex_to_solution_cluster_topk_csv": str(OUT_SOLUTION_CLUSTER_MAP_CSV) if SAVE_FULL_TOPK_MAPPING else None,
        "summary_json": str(OUT_SUMMARY_JSON),
    }
}

with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n============================================================")
print("DONE")
print("Naive problem_cluster × OpenAlex concepts:")
print(f"{naive_problem_cluster_x_openalex:,}")
print("\nFramework candidate counts:")
for thr in SIMILARITY_THRESHOLDS:
    key = str(thr)
    label = "no threshold" if thr is None else f"cosine >= {thr}"
    print(
        f"{label}: "
        f"{threshold_summaries[key]['total_candidate_problem_concept_pairs']:,} "
        f"problem_cluster + OpenAlex concept pairs"
    )

print("\nOutputs:")
print("Funnel:", OUT_FUNNEL_CSV)
print("Concept counts:", OUT_CONCEPT_COUNTS_CSV)
print("Problem counts:", OUT_PROBLEM_CLUSTER_COUNTS_CSV)
if SAVE_FULL_TOPK_MAPPING:
    print("Top-k mapping:", OUT_SOLUTION_CLUSTER_MAP_CSV)
print("Summary:", OUT_SUMMARY_JSON)
print("============================================================")