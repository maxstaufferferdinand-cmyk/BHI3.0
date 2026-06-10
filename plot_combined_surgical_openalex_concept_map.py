from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from adjustText import adjust_text
import umap


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

# -----------------------------
# Surgical inputs
# -----------------------------
SURGICAL_EDGES_FILE = (
    BASE_DIR
    / "surgical_training_graph_outputs"
    / "surgical_graph_edges_aggregated_with_years.csv"
)

SURGICAL_PAIR_EMB_FILE = (
    BASE_DIR
    / "surgical_training_graph_outputs"
    / "surgical_pair_embeddings.npy"
)

# -----------------------------
# OpenAlex inputs
# -----------------------------
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

# -----------------------------
# Outputs
# -----------------------------
OUT_DIR = BASE_DIR / "combined_surgical_openalex_concept_map_outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_PNG = OUT_DIR / "figure_combined_surgical_openalex_concept_map.png"
OUT_PDF = OUT_DIR / "figure_combined_surgical_openalex_concept_map.pdf"
OUT_COORDS_CSV = OUT_DIR / "combined_surgical_openalex_umap_coordinates.csv"
OUT_LABELS_CSV = OUT_DIR / "combined_surgical_openalex_plotted_labels.csv"
OUT_SUMMARY_JSON = OUT_DIR / "combined_surgical_openalex_concept_map_summary.json"

# -----------------------------
# Sampling
# -----------------------------
# Für eine schöne Figure nicht alle 480k OpenAlex + 120k Surgical nehmen.
# Top nach Frequenz/Gewicht reicht für eine Paper-Figure.
TOP_N_SURGICAL = 15000
TOP_N_OPENALEX = 15000

# Labels
MAX_SURGICAL_LABELS = 18
MAX_OPENALEX_LABELS = 18

# PCA + UMAP
PCA_COMPONENTS = 50
RANDOM_STATE = 42

# UMAP style
UMAP_N_NEIGHBORS = 25
UMAP_MIN_DIST = 0.08

# Density
DENSITY_BINS = 420
DENSITY_SIGMA = 4


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


def short_label(text: str, max_len: int = 45) -> str:
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = False):
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    # fallback: contains-based search
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


def make_density_background(x, y, bins=400, sigma=4):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap.T, sigma=sigma)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap, extent


def load_surgical_data():
    print("Loading surgical pair embeddings and edge table...")

    if not SURGICAL_EDGES_FILE.exists():
        raise FileNotFoundError(f"Missing surgical edges file: {SURGICAL_EDGES_FILE}")

    if not SURGICAL_PAIR_EMB_FILE.exists():
        raise FileNotFoundError(f"Missing surgical pair embeddings: {SURGICAL_PAIR_EMB_FILE}")

    surg_df = pd.read_csv(SURGICAL_EDGES_FILE, encoding="utf-8-sig", low_memory=False)
    surg_emb = np.load(SURGICAL_PAIR_EMB_FILE, mmap_mode="r")

    print(f"Surgical edge rows: {len(surg_df):,}")
    print(f"Surgical pair embedding shape: {surg_emb.shape}")
    print("Surgical columns:", list(surg_df.columns))

    if len(surg_df) != surg_emb.shape[0]:
        raise ValueError(
            "Surgical edge table and surgical_pair_embeddings.npy do not have the same row count. "
            "If this happens, we need the exact pair index file used for surgical_pair_embeddings.npy."
        )

    # robust column detection
    problem_col = find_col(
        surg_df,
        [
            "surgical_problem",
            "problem_text",
            "problem",
            "problem_label",
            "problem_name",
            "source_problem",
            "problem_node_text",
            "problem_concept",
        ],
        required=False,
    )

    solution_col = find_col(
        surg_df,
        [
            "technical_solution",
            "solution_text",
            "solution",
            "solution_label",
            "solution_name",
            "target_solution",
            "solution_node_text",
            "solution_concept",
        ],
        required=False,
    )

    weight_col = find_col(
        surg_df,
        [
            "weight",
            "edge_weight",
            "cluster_weight",
            "n_pmids",
            "count",
            "n",
        ],
        required=False,
    )

    year_col = find_col(
        surg_df,
        [
            "first_year",
            "year",
            "publication_year",
        ],
        required=False,
    )

    print("Detected surgical columns:")
    print("  problem:", problem_col)
    print("  solution:", solution_col)
    print("  weight:", weight_col)
    print("  year:", year_col)

    if problem_col and solution_col:
        surg_df["_label_full"] = [
            f"{clean_text(p)} → {clean_text(s)}"
            for p, s in zip(surg_df[problem_col], surg_df[solution_col])
        ]
    else:
        # fallback label if text columns are absent
        surg_df["_label_full"] = [
            f"surgical pair {i}"
            for i in range(len(surg_df))
        ]

    if weight_col:
        surg_df["_rank_weight"] = pd.to_numeric(surg_df[weight_col], errors="coerce").fillna(0)
    else:
        surg_df["_rank_weight"] = 1

    if year_col:
        surg_df["_year"] = pd.to_numeric(surg_df[year_col], errors="coerce")
    else:
        surg_df["_year"] = np.nan

    # select top surgical pairs by weight
    surg_df["_source_row"] = np.arange(len(surg_df))

    surg_sel = (
        surg_df
        .sort_values(["_rank_weight"], ascending=False)
        .head(TOP_N_SURGICAL)
        .copy()
        .reset_index(drop=True)
    )

    surg_emb_sel = np.array(surg_emb[surg_sel["_source_row"].values], dtype=np.float32)

    out_meta = pd.DataFrame({
        "domain": "Surgical",
        "source_row": surg_sel["_source_row"].values,
        "concept_id": surg_sel["_source_row"].values,
        "label_full": surg_sel["_label_full"].values,
        "rank_weight": surg_sel["_rank_weight"].values,
        "n_works": np.nan,
        "n_mentions": np.nan,
        "year": surg_sel["_year"].values,
    })

    return out_meta, surg_emb_sel


def load_openalex_data():
    print("Loading OpenAlex concept index and embeddings...")

    if not OPENALEX_INDEX_FILE.exists():
        raise FileNotFoundError(f"Missing OpenAlex index: {OPENALEX_INDEX_FILE}")

    if not OPENALEX_EMB_FILE.exists():
        raise FileNotFoundError(f"Missing OpenAlex embeddings: {OPENALEX_EMB_FILE}")

    oa_df = pd.read_csv(OPENALEX_INDEX_FILE, encoding="utf-8-sig")
    oa_emb = np.load(OPENALEX_EMB_FILE, mmap_mode="r")

    print(f"OpenAlex index rows: {len(oa_df):,}")
    print(f"OpenAlex embedding shape: {oa_emb.shape}")
    print("OpenAlex columns:", list(oa_df.columns))

    if len(oa_df) != oa_emb.shape[0]:
        raise ValueError("OpenAlex index and embedding rows do not match.")

    required = {"concept_id", "concept_text"}
    missing = required - set(oa_df.columns)

    if missing:
        raise ValueError(f"OpenAlex index missing required columns: {missing}")

    if "n_works" not in oa_df.columns:
        oa_df["n_works"] = 1

    if "n_mentions" not in oa_df.columns:
        oa_df["n_mentions"] = 1

    oa_df["_rank_weight"] = (
        pd.to_numeric(oa_df["n_works"], errors="coerce").fillna(0) * 1.0
        + pd.to_numeric(oa_df["n_mentions"], errors="coerce").fillna(0) * 0.01
    )

    oa_df["_source_row"] = np.arange(len(oa_df))

    oa_sel = (
        oa_df
        .sort_values(["n_works", "n_mentions"], ascending=[False, False])
        .head(TOP_N_OPENALEX)
        .copy()
        .reset_index(drop=True)
    )

    oa_emb_sel = np.array(oa_emb[oa_sel["concept_id"].values], dtype=np.float32)

    out_meta = pd.DataFrame({
        "domain": "OpenAlex",
        "source_row": oa_sel["_source_row"].values,
        "concept_id": oa_sel["concept_id"].values,
        "label_full": oa_sel["concept_text"].apply(clean_text).values,
        "rank_weight": oa_sel["_rank_weight"].values,
        "n_works": oa_sel["n_works"].values,
        "n_mentions": oa_sel["n_mentions"].values,
        "year": np.nan,
    })

    return out_meta, oa_emb_sel


def select_labels(plot_df: pd.DataFrame):
    print("Selecting labels...")

    labels = []

    surg = plot_df[plot_df["domain"] == "Surgical"].copy()
    oa = plot_df[plot_df["domain"] == "OpenAlex"].copy()

    # Surgical top labels by rank weight
    surg = surg.sort_values("rank_weight", ascending=False).head(MAX_SURGICAL_LABELS)
    for _, row in surg.iterrows():
        labels.append(row)

    # OpenAlex top labels by n_works / rank_weight
    oa = oa.sort_values(["n_works", "n_mentions"], ascending=[False, False]).head(MAX_OPENALEX_LABELS)
    for _, row in oa.iterrows():
        labels.append(row)

    label_df = pd.DataFrame(labels)

    if label_df.empty:
        return label_df

    label_df["plot_label"] = label_df["label_full"].apply(lambda x: short_label(x, max_len=48))

    print(f"Labels selected: {len(label_df)}")
    print(label_df[["domain", "plot_label", "rank_weight"]].head(10))

    return label_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("============================================================")
    print("Combined surgical + OpenAlex concept map")
    print("============================================================")

    surgical_meta, surgical_emb = load_surgical_data()
    openalex_meta, openalex_emb = load_openalex_data()

    if surgical_emb.shape[1] != openalex_emb.shape[1]:
        raise ValueError(
            f"Embedding dimensions differ: surgical {surgical_emb.shape[1]}, OpenAlex {openalex_emb.shape[1]}"
        )

    print("Combining embeddings...")
    combined_meta = pd.concat([surgical_meta, openalex_meta], ignore_index=True)
    combined_emb = np.vstack([surgical_emb, openalex_emb]).astype(np.float32)

    print(f"Combined rows: {len(combined_meta):,}")
    print(f"Combined embedding shape: {combined_emb.shape}")

    # --------------------------------------------------------
    # PCA + joint UMAP
    # --------------------------------------------------------
    print("Running joint PCA...")

    pca = PCA(
        n_components=min(PCA_COMPONENTS, combined_emb.shape[1]),
        random_state=RANDOM_STATE,
    )

    combined_pca = pca.fit_transform(combined_emb)

    print("Running joint UMAP...")

    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="cosine",
        random_state=RANDOM_STATE,
        verbose=True,
    )

    coords = umap_model.fit_transform(combined_pca)

    combined_meta["umap_x"] = coords[:, 0]
    combined_meta["umap_y"] = coords[:, 1]

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    label_df = select_labels(combined_meta)

    # --------------------------------------------------------
    # Density background
    # --------------------------------------------------------
    print("Creating density background...")

    density, extent = make_density_background(
        combined_meta["umap_x"].values,
        combined_meta["umap_y"].values,
        bins=DENSITY_BINS,
        sigma=DENSITY_SIGMA,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    print("Plotting combined concept map...")

    plt.figure(figsize=(17, 12.5))

    # background density
    plt.imshow(
        density,
        extent=extent,
        origin="lower",
        cmap="Greys",
        alpha=0.35,
        aspect="auto",
    )

    surg_plot = combined_meta[combined_meta["domain"] == "Surgical"]
    oa_plot = combined_meta[combined_meta["domain"] == "OpenAlex"]

    # OpenAlex first, then surgical on top
    plt.scatter(
        oa_plot["umap_x"],
        oa_plot["umap_y"],
        s=5,
        c="#1f77b4",
        alpha=0.35,
        linewidths=0,
        label="OpenAlex engineering concepts",
        zorder=2,
    )

    plt.scatter(
        surg_plot["umap_x"],
        surg_plot["umap_y"],
        s=8,
        c="#d62728",
        alpha=0.45,
        linewidths=0,
        label="Surgical problem–solution pairs",
        zorder=3,
    )

    # label points
    texts = []

    if not label_df.empty:
        surg_labels = label_df[label_df["domain"] == "Surgical"]
        oa_labels = label_df[label_df["domain"] == "OpenAlex"]

        if len(oa_labels) > 0:
            plt.scatter(
                oa_labels["umap_x"],
                oa_labels["umap_y"],
                s=36,
                c="#00bcd4",
                edgecolors="white",
                linewidths=0.6,
                alpha=0.95,
                label="Labeled OpenAlex concepts",
                zorder=5,
            )

        if len(surg_labels) > 0:
            plt.scatter(
                surg_labels["umap_x"],
                surg_labels["umap_y"],
                s=42,
                c="#ff3b30",
                edgecolors="white",
                linewidths=0.6,
                alpha=0.95,
                label="Labeled surgical pairs",
                zorder=6,
            )

        for _, row in label_df.iterrows():
            if row["domain"] == "Surgical":
                edge_color = "#d62728"
                face_color = "#fff1f0"
            else:
                edge_color = "#1f77b4"
                face_color = "#eef6ff"

            txt = plt.text(
                row["umap_x"],
                row["umap_y"],
                row["plot_label"],
                fontsize=7.1,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor=face_color,
                    edgecolor=edge_color,
                    alpha=0.90,
                    linewidth=0.55,
                ),
                zorder=7,
            )
            texts.append(txt)

        print("Adjusting labels to avoid collision...")

        adjust_text(
            texts,
            x=label_df["umap_x"].values,
            y=label_df["umap_y"].values,
            expand_text=(1.25, 1.45),
            expand_points=(1.35, 1.55),
            force_text=(0.50, 0.90),
            force_points=(0.25, 0.45),
            lim=500,
            arrowprops=dict(
                arrowstyle="-",
                color="gray",
                lw=0.45,
                alpha=0.62,
            ),
        )

    plt.title(
        "Combined surgical and OpenAlex engineering concept map",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )

    plt.xlabel("Joint UMAP component 1", fontsize=11)
    plt.ylabel("Joint UMAP component 2", fontsize=11)

    plt.legend(
        loc="lower right",
        frameon=True,
        fontsize=8.5,
        markerscale=1.8,
    )

    caption = (
        "Joint UMAP projection of surgical problem–solution pair embeddings and OpenAlex engineering mechanism embeddings. "
        "Surgical concepts are shown in red/orange; OpenAlex engineering concepts are shown in blue/cyan. "
        "Labels mark high-frequency representative concepts and are automatically adjusted to reduce overlap."
    )

    plt.figtext(
        0.10,
        0.015,
        caption,
        ha="left",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.045, 1, 1])

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()

    print("Saved PNG:", OUT_PNG)
    print("Saved PDF:", OUT_PDF)

    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------
    print("Saving CSV outputs...")

    combined_meta.to_csv(OUT_COORDS_CSV, index=False, encoding="utf-8-sig")

    if not label_df.empty:
        label_df.to_csv(OUT_LABELS_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "surgical_edges_file": str(SURGICAL_EDGES_FILE),
        "surgical_pair_embedding_file": str(SURGICAL_PAIR_EMB_FILE),
        "openalex_index_file": str(OPENALEX_INDEX_FILE),
        "openalex_embedding_file": str(OPENALEX_EMB_FILE),
        "top_n_surgical": int(TOP_N_SURGICAL),
        "top_n_openalex": int(TOP_N_OPENALEX),
        "combined_rows": int(len(combined_meta)),
        "embedding_dim": int(combined_emb.shape[1]),
        "pca_components": int(min(PCA_COMPONENTS, combined_emb.shape[1])),
        "umap_n_neighbors": int(UMAP_N_NEIGHBORS),
        "umap_min_dist": float(UMAP_MIN_DIST),
        "max_surgical_labels": int(MAX_SURGICAL_LABELS),
        "max_openalex_labels": int(MAX_OPENALEX_LABELS),
        "outputs": {
            "png": str(OUT_PNG),
            "pdf": str(OUT_PDF),
            "coords_csv": str(OUT_COORDS_CSV),
            "labels_csv": str(OUT_LABELS_CSV),
            "summary_json": str(OUT_SUMMARY_JSON),
        },
    }

    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved coordinates:", OUT_COORDS_CSV)
    print("Saved labels:", OUT_LABELS_CSV)
    print("Saved summary:", OUT_SUMMARY_JSON)

    print("============================================================")
    print("DONE")
    print("Combined map PNG:")
    print(OUT_PNG)
    print("============================================================")


if __name__ == "__main__":
    main()