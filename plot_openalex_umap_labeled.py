from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import gaussian_filter
import umap


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

INDEX_FILE = (
    BASE_DIR
    / "openalex_engineering_mechanism_embeddings_cached"
    / "openalex_engineering_mechanism_index.csv"
)

EMB_FILE = (
    BASE_DIR
    / "openalex_engineering_mechanism_embeddings_cached"
    / "openalex_engineering_mechanism_embeddings.npy"
)

OUT_DIR = BASE_DIR / "openalex_umap_labeled_outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_UMAP_PNG = OUT_DIR / "openalex_concepts_umap_labeled.png"
OUT_UMAP_PDF = OUT_DIR / "openalex_concepts_umap_labeled.pdf"
OUT_COORDS_CSV = OUT_DIR / "openalex_concepts_umap_labeled_coordinates.csv"
OUT_CLUSTER_SUMMARY = OUT_DIR / "openalex_concept_cluster_summary.csv"
OUT_SUMMARY_JSON = OUT_DIR / "openalex_umap_labeled_summary.json"

# Wie viele Concepts in die UMAP sollen.
# 20k ist meist guter Kompromiss. Für schöneres Bild: 10k–30k.
TOP_N_FOR_UMAP = 20000

# PCA vor UMAP beschleunigt und stabilisiert.
PCA_COMPONENTS = 50

# Cluster für Farbe.
N_CLUSTERS = 25

# Labels: nicht zu viele, sonst wird es hässlich.
MAX_LABELS_TOTAL = 35
MIN_CLUSTER_SIZE_FOR_LABEL = 100

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms


def short_label(text: str, max_len: int = 42) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def make_density_background(x, y, bins=350, sigma=4):
    """
    Creates smooth density grid for UMAP background.
    """
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap.T, sigma=sigma)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap, extent


# ============================================================
# LOAD
# ============================================================

print("Loading OpenAlex index and embeddings...")

if not INDEX_FILE.exists():
    raise FileNotFoundError(f"Missing index file: {INDEX_FILE}")

if not EMB_FILE.exists():
    raise FileNotFoundError(f"Missing embedding file: {EMB_FILE}")

index_df = pd.read_csv(INDEX_FILE, encoding="utf-8-sig")
emb = np.load(EMB_FILE, mmap_mode="r")

print(f"Index rows: {len(index_df):,}")
print(f"Embedding shape: {emb.shape}")

if len(index_df) != emb.shape[0]:
    raise ValueError("Index rows and embedding rows do not match.")

required_cols = {"concept_id", "concept_text"}
missing = required_cols - set(index_df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Häufigste Concepts auswählen
sort_cols = []
ascending = []

if "n_works" in index_df.columns:
    sort_cols.append("n_works")
    ascending.append(False)

if "n_mentions" in index_df.columns:
    sort_cols.append("n_mentions")
    ascending.append(False)

if sort_cols:
    plot_df = index_df.sort_values(sort_cols, ascending=ascending).head(TOP_N_FOR_UMAP).copy()
else:
    plot_df = index_df.head(TOP_N_FOR_UMAP).copy()

plot_df = plot_df.reset_index(drop=True)
concept_ids = plot_df["concept_id"].values

plot_emb = np.array(emb[concept_ids], dtype=np.float32)

print(f"Using {len(plot_df):,} concepts for UMAP.")


# ============================================================
# PCA + UMAP
# ============================================================

print("Running PCA...")

pca = PCA(
    n_components=min(PCA_COMPONENTS, plot_emb.shape[1]),
    random_state=RANDOM_STATE,
)

plot_emb_pca = pca.fit_transform(plot_emb)

print("Running UMAP...")

umap_model = umap.UMAP(
    n_neighbors=20,
    min_dist=0.08,
    n_components=2,
    metric="cosine",
    random_state=RANDOM_STATE,
    verbose=True,
)

coords = umap_model.fit_transform(plot_emb_pca)

plot_df["umap_x"] = coords[:, 0]
plot_df["umap_y"] = coords[:, 1]


# ============================================================
# CLUSTERING
# ============================================================

print("Clustering concepts...")

kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    batch_size=2048,
    n_init=10,
)

plot_df["cluster_id"] = kmeans.fit_predict(plot_emb_pca)

cluster_sizes = (
    plot_df.groupby("cluster_id")
    .size()
    .reset_index(name="n_concepts")
    .sort_values("n_concepts", ascending=False)
)

print("Cluster sizes:")
print(cluster_sizes.head(10))


# ============================================================
# SELECT TOP LABELS
# ============================================================

print("Selecting labels...")

label_rows = []

# pro Cluster der häufigste / wichtigste Begriff
for cluster_id in cluster_sizes["cluster_id"].tolist():
    sub = plot_df[plot_df["cluster_id"] == cluster_id].copy()

    if len(sub) < MIN_CLUSTER_SIZE_FOR_LABEL:
        continue

    if "n_works" in sub.columns:
        sub = sub.sort_values(["n_works", "n_mentions"], ascending=[False, False])
    elif "n_mentions" in sub.columns:
        sub = sub.sort_values("n_mentions", ascending=False)

    best = sub.iloc[0].copy()
    label_rows.append(best)

    if len(label_rows) >= MAX_LABELS_TOTAL:
        break

label_df = pd.DataFrame(label_rows)

print(f"Selected {len(label_df)} labels.")


# ============================================================
# DENSITY BACKGROUND
# ============================================================

print("Creating density background...")

density, extent = make_density_background(
    plot_df["umap_x"].values,
    plot_df["umap_y"].values,
    bins=400,
    sigma=4,
)


# ============================================================
# PLOT
# ============================================================

print("Plotting UMAP...")

plt.figure(figsize=(16, 12))

# Hintergrunddichte wie im Beispielbild
plt.imshow(
    density,
    extent=extent,
    origin="lower",
    cmap="viridis",
    alpha=0.85,
    aspect="auto",
)

# Alle Concepts als kleine Punkte mit Clusterfarben
scatter = plt.scatter(
    plot_df["umap_x"],
    plot_df["umap_y"],
    c=plot_df["cluster_id"],
    cmap="tab20",
    s=4,
    alpha=0.45,
    linewidths=0,
)

# Top-Labels als rote Punkte + Textboxen
if not label_df.empty:
    plt.scatter(
        label_df["umap_x"],
        label_df["umap_y"],
        c="red",
        s=28,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.4,
        label="Top cluster concepts",
        zorder=5,
    )

    for _, row in label_df.iterrows():
        label = short_label(row["concept_text"], max_len=45)

        plt.text(
            row["umap_x"] + 0.08,
            row["umap_y"] + 0.08,
            label,
            fontsize=7.5,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="gray",
                alpha=0.82,
                linewidth=0.4,
            ),
            zorder=6,
        )

plt.title(
    "OpenAlex engineering mechanism concept map",
    fontsize=16,
    fontweight="bold",
    pad=16,
)

plt.xlabel("UMAP component 1", fontsize=11)
plt.ylabel("UMAP component 2", fontsize=11)

cbar = plt.colorbar(scatter, fraction=0.035, pad=0.02)
cbar.set_label("Embedding cluster", fontsize=10)

plt.legend(loc="lower right", frameon=True, fontsize=9)

caption = (
    "Two-dimensional UMAP projection of OpenAlex engineering mechanism embeddings. "
    "Background density indicates local concentration of concepts; colored points indicate embedding clusters; "
    "red labels mark high-frequency representative concepts."
)

plt.figtext(
    0.12,
    0.015,
    caption,
    ha="left",
    fontsize=9,
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

plt.savefig(OUT_UMAP_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_UMAP_PDF, bbox_inches="tight")
plt.close()

print("Saved UMAP PNG:", OUT_UMAP_PNG)
print("Saved UMAP PDF:", OUT_UMAP_PDF)


# ============================================================
# EXPORT TABLES
# ============================================================

print("Saving CSV outputs...")

plot_df.to_csv(OUT_COORDS_CSV, index=False, encoding="utf-8-sig")

cluster_summary_rows = []

for cluster_id in sorted(plot_df["cluster_id"].unique()):
    sub = plot_df[plot_df["cluster_id"] == cluster_id].copy()

    if "n_works" in sub.columns:
        sub = sub.sort_values(["n_works", "n_mentions"], ascending=[False, False])
    elif "n_mentions" in sub.columns:
        sub = sub.sort_values("n_mentions", ascending=False)

    top_terms = sub["concept_text"].head(12).tolist()

    cluster_summary_rows.append(
        {
            "cluster_id": int(cluster_id),
            "n_concepts": int(len(sub)),
            "top_terms": " | ".join(top_terms),
        }
    )

cluster_summary = pd.DataFrame(cluster_summary_rows)
cluster_summary.to_csv(OUT_CLUSTER_SUMMARY, index=False, encoding="utf-8-sig")

summary = {
    "index_file": str(INDEX_FILE),
    "embedding_file": str(EMB_FILE),
    "n_total_concepts": int(len(index_df)),
    "embedding_shape": list(emb.shape),
    "top_n_for_umap": int(len(plot_df)),
    "pca_components": int(min(PCA_COMPONENTS, plot_emb.shape[1])),
    "n_clusters": int(N_CLUSTERS),
    "max_labels_total": int(MAX_LABELS_TOTAL),
    "outputs": {
        "umap_png": str(OUT_UMAP_PNG),
        "umap_pdf": str(OUT_UMAP_PDF),
        "coordinates_csv": str(OUT_COORDS_CSV),
        "cluster_summary_csv": str(OUT_CLUSTER_SUMMARY),
    },
}

with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Saved coordinates:", OUT_COORDS_CSV)
print("Saved cluster summary:", OUT_CLUSTER_SUMMARY)
print("Saved summary:", OUT_SUMMARY_JSON)

print("============================================================")
print("DONE")
print("PNG:")
print(OUT_UMAP_PNG)
print("============================================================")