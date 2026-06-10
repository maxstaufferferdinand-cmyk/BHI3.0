from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import umap


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

INDEX_FILE = BASE_DIR / "openalex_engineering_mechanism_embeddings_cached" / "openalex_engineering_mechanism_index.csv"
EMB_FILE = BASE_DIR / "openalex_engineering_mechanism_embeddings_cached" / "openalex_engineering_mechanism_embeddings.npy"

OUT_DIR = BASE_DIR / "openalex_umap_heatmap_outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_UMAP_PNG = OUT_DIR / "openalex_concepts_umap.png"
OUT_HEATMAP_PNG = OUT_DIR / "openalex_concepts_cluster_heatmap.png"
OUT_UMAP_CSV = OUT_DIR / "openalex_concepts_umap_coordinates.csv"
OUT_CLUSTER_CSV = OUT_DIR / "openalex_concepts_cluster_assignments.csv"
OUT_SUMMARY_JSON = OUT_DIR / "openalex_umap_heatmap_summary.json"

# UMAP / sampling
TOP_N_FOR_UMAP = 20000      # häufigste Concepts
PCA_COMPONENTS = 50
UMAP_RANDOM_STATE = 42

# Clustering for heatmap
N_CLUSTERS = 40


# ============================================================
# HELPERS
# ============================================================

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms


# ============================================================
# LOAD
# ============================================================

print("Loading files...")
if not INDEX_FILE.exists():
    raise FileNotFoundError(f"Missing index file: {INDEX_FILE}")
if not EMB_FILE.exists():
    raise FileNotFoundError(f"Missing embedding file: {EMB_FILE}")

index_df = pd.read_csv(INDEX_FILE, encoding="utf-8-sig")
emb = np.load(EMB_FILE)

print(f"Index rows: {len(index_df):,}")
print(f"Embedding shape: {emb.shape}")

if len(index_df) != emb.shape[0]:
    raise ValueError("Index length and embedding rows do not match.")

required_cols = {"concept_id", "concept_text"}
missing_cols = required_cols - set(index_df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Sort by n_works if available
if "n_works" in index_df.columns:
    plot_df = index_df.sort_values(["n_works", "n_mentions"], ascending=[False, False]).copy()
else:
    plot_df = index_df.copy()

# Select top N for UMAP
plot_df = plot_df.head(TOP_N_FOR_UMAP).copy()
plot_df = plot_df.reset_index(drop=True)

plot_emb = emb[plot_df["concept_id"].values]

print(f"Using {len(plot_df):,} concepts for UMAP and clustering.")


# ============================================================
# PCA -> UMAP
# ============================================================

print("Running PCA...")
pca = PCA(n_components=min(PCA_COMPONENTS, plot_emb.shape[1]), random_state=UMAP_RANDOM_STATE)
plot_emb_pca = pca.fit_transform(plot_emb)

print("Running UMAP...")
umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=UMAP_RANDOM_STATE,
)
umap_coords = umap_model.fit_transform(plot_emb_pca)

plot_df["umap_x"] = umap_coords[:, 0]
plot_df["umap_y"] = umap_coords[:, 1]


# ============================================================
# CLUSTERING
# ============================================================

print("Clustering concepts...")
kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=UMAP_RANDOM_STATE,
    batch_size=2048,
    n_init=10,
)

cluster_labels = kmeans.fit_predict(plot_emb_pca)
plot_df["cluster_id"] = cluster_labels

centroids = kmeans.cluster_centers_
centroid_sim = cosine_similarity(centroids)


# ============================================================
# UMAP PLOT
# ============================================================

print("Plotting UMAP...")
plt.figure(figsize=(12, 10))

# Coloring by cluster
scatter = plt.scatter(
    plot_df["umap_x"],
    plot_df["umap_y"],
    c=plot_df["cluster_id"],
    s=6,
    alpha=0.7,
)

plt.xlabel("UMAP 1", fontsize=12, fontweight="bold")
plt.ylabel("UMAP 2", fontsize=12, fontweight="bold")
plt.title("OpenAlex Engineering Concepts – UMAP", fontsize=14, fontweight="bold")

# cluster labels near cluster centers in UMAP space
cluster_centers_umap = (
    plot_df.groupby("cluster_id")[["umap_x", "umap_y"]]
    .mean()
    .reset_index()
)

for _, row in cluster_centers_umap.iterrows():
    plt.text(
        row["umap_x"],
        row["umap_y"],
        str(int(row["cluster_id"])),
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
    )

plt.tight_layout()
plt.savefig(OUT_UMAP_PNG, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# HEATMAP PLOT
# ============================================================

print("Plotting heatmap...")
plt.figure(figsize=(11, 9))

im = plt.imshow(centroid_sim, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04, label="Cosine similarity")

plt.xticks(range(N_CLUSTERS), [str(i) for i in range(N_CLUSTERS)], rotation=90, fontsize=8)
plt.yticks(range(N_CLUSTERS), [str(i) for i in range(N_CLUSTERS)], fontsize=8)

plt.xlabel("Cluster", fontsize=12, fontweight="bold")
plt.ylabel("Cluster", fontsize=12, fontweight="bold")
plt.title("OpenAlex Concept Cluster Centroid Similarity Heatmap", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_HEATMAP_PNG, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# EXPORT TABLES
# ============================================================

print("Saving tables...")

plot_df.to_csv(OUT_UMAP_CSV, index=False, encoding="utf-8-sig")

# representative concepts per cluster
cluster_summary_rows = []
for c in range(N_CLUSTERS):
    sub = plot_df[plot_df["cluster_id"] == c].copy()
    sub = sub.sort_values(["n_works", "n_mentions"], ascending=[False, False]) if "n_works" in sub.columns else sub
    top_examples = sub["concept_text"].head(10).tolist()

    cluster_summary_rows.append({
        "cluster_id": c,
        "n_concepts": len(sub),
        "top_examples": " | ".join(top_examples),
    })

cluster_summary_df = pd.DataFrame(cluster_summary_rows)
cluster_summary_df.to_csv(OUT_CLUSTER_CSV, index=False, encoding="utf-8-sig")

summary = {
    "index_file": str(INDEX_FILE),
    "embedding_file": str(EMB_FILE),
    "n_index_rows_total": int(len(index_df)),
    "embedding_shape_total": list(emb.shape),
    "n_used_for_umap_and_clustering": int(len(plot_df)),
    "pca_components": int(min(PCA_COMPONENTS, plot_emb.shape[1])),
    "n_clusters": int(N_CLUSTERS),
    "outputs": {
        "umap_png": str(OUT_UMAP_PNG),
        "heatmap_png": str(OUT_HEATMAP_PNG),
        "umap_coordinates_csv": str(OUT_UMAP_CSV),
        "cluster_assignments_csv": str(OUT_CLUSTER_CSV),
    }
}

with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nDONE")
print("UMAP PNG:", OUT_UMAP_PNG)
print("Heatmap PNG:", OUT_HEATMAP_PNG)
print("UMAP coordinates CSV:", OUT_UMAP_CSV)
print("Cluster assignments CSV:", OUT_CLUSTER_CSV)
print("Summary JSON:", OUT_SUMMARY_JSON)