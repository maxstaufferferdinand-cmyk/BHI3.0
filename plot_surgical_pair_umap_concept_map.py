from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import umap

from scipy.ndimage import gaussian_filter

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except Exception:
    HAS_ADJUSTTEXT = False


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")
GRAPH_DIR = BASE_DIR / "surgical_training_graph_outputs"

PAIR_INDEX_CSV = GRAPH_DIR / "surgical_pair_embedding_index.csv"
PAIR_NODES_CSV = GRAPH_DIR / "surgical_pair_nodes.csv"
PAIR_EMB_NPY = GRAPH_DIR / "surgical_pair_embeddings.npy"

OUT_DIR = GRAPH_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

OUT_UMAP_COORDS = OUT_DIR / "surgical_pair_umap_coordinates.csv"
OUT_FIG_PNG = OUT_DIR / "surgical_pair_umap_concept_map.png"
OUT_FIG_PDF = OUT_DIR / "surgical_pair_umap_concept_map.pdf"
OUT_LABELS_CSV = OUT_DIR / "surgical_pair_umap_selected_labels.csv"

# Full UMAP on 120k pair embeddings may work, but can take time.
# For a first paper-quality map, 60k is usually enough.
MAX_POINTS_FOR_UMAP = 60000

# Always keep the most frequent pair edges, then sample the rest.
N_TOP_ALWAYS_KEEP = 8000
RANDOM_STATE = 42

# UMAP parameters
PCA_COMPONENTS = 50
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "cosine"

# Density background
GRID_SIZE = 450
DENSITY_SMOOTH_SIGMA = 7

# Labels
MAX_LABELS = 55
GRID_LABEL_BINS = 9
MIN_LABEL_SCORE = 2  # pair weight threshold; lower if too few labels

# Figure aesthetics
FIGSIZE = (12, 10)
DPI = 400

# Point aesthetics
POINT_SIZE = 2.2
POINT_ALPHA = 0.23

TOP_DOT_SIZE = 22
TOP_DOT_ALPHA = 0.95

TITLE = "Surgical problem–solution concept map"
SUBTITLE = (
    "Two-dimensional UMAP projection of embedded surgical problem–solution pairs. "
    "Background density indicates local concentration of concepts; red points label "
    "high-frequency historical problem–solution relations."
)


# ============================================================
# HELPERS
# ============================================================

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def short_label(x, max_chars=42):
    x = safe_str(x)
    x = re.sub(r"\s+", " ", x).strip()

    # Pair labels look like: problem -> solution
    if " -> " in x:
        left, right = x.split(" -> ", 1)
        left = left.strip()
        right = right.strip()

        left = shorten_phrase(left, 22)
        right = shorten_phrase(right, 22)
        return f"{left} → {right}"

    return shorten_phrase(x, max_chars)


def shorten_phrase(x, max_chars=38):
    x = safe_str(x)
    if len(x) <= max_chars:
        return x

    # Prefer meaningful truncation at word boundary
    words = x.split()
    out = []
    for w in words:
        candidate = " ".join(out + [w])
        if len(candidate) > max_chars:
            break
        out.append(w)

    if not out:
        return x[:max_chars - 1] + "…"
    return " ".join(out) + "…"


def load_inputs():
    print("Loading inputs...")
    print("Pair index:", PAIR_INDEX_CSV)
    print("Pair nodes:", PAIR_NODES_CSV)
    print("Pair embeddings:", PAIR_EMB_NPY)

    pair_index = pd.read_csv(PAIR_INDEX_CSV, encoding="utf-8-sig", low_memory=False)
    pair_nodes = pd.read_csv(PAIR_NODES_CSV, encoding="utf-8-sig", low_memory=False)
    emb = np.load(PAIR_EMB_NPY)

    print("pair_index:", pair_index.shape)
    print("pair_nodes:", pair_nodes.shape)
    print("embeddings:", emb.shape)

    if len(pair_index) != emb.shape[0]:
        raise ValueError(
            f"Length mismatch: pair_index={len(pair_index)} vs embeddings={emb.shape[0]}"
        )

    # pair_index has pair_id, label, pair_key, problem_id, solution_id, embedding_text
    # pair_nodes has pair_id, weight, years, etc.
    df = pair_index.merge(
        pair_nodes[
            [
                "pair_id",
                "surgical_problem",
                "technical_solution",
                "weight",
                "first_year",
                "last_year",
            ]
        ],
        on="pair_id",
        how="left",
    )

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1).astype(float)
    df["label"] = df["label"].fillna(
        df["surgical_problem"].astype(str) + " -> " + df["technical_solution"].astype(str)
    )

    return df, emb


def select_points(df, emb):
    n = len(df)

    if n <= MAX_POINTS_FOR_UMAP:
        print(f"Using all {n:,} points for UMAP.")
        selected_idx = np.arange(n)
        return df.copy(), emb.copy(), selected_idx

    print(f"Selecting {MAX_POINTS_FOR_UMAP:,} of {n:,} points for UMAP...")

    rng = np.random.default_rng(RANDOM_STATE)

    # keep top weighted pairs
    top_idx = (
        df.sort_values("weight", ascending=False)
        .head(min(N_TOP_ALWAYS_KEEP, n))
        .index
        .to_numpy()
    )

    remaining_idx = np.setdiff1d(np.arange(n), top_idx)

    n_random = MAX_POINTS_FOR_UMAP - len(top_idx)
    if n_random > 0:
        random_idx = rng.choice(remaining_idx, size=n_random, replace=False)
        selected_idx = np.concatenate([top_idx, random_idx])
    else:
        selected_idx = top_idx[:MAX_POINTS_FOR_UMAP]

    selected_idx = np.unique(selected_idx)
    selected_df = df.iloc[selected_idx].copy().reset_index(drop=True)
    selected_emb = emb[selected_idx]

    print("Selected:", len(selected_df))
    print("Top-weight pairs always kept:", len(top_idx))

    return selected_df, selected_emb, selected_idx


def compute_umap(emb):
    print("Normalizing embeddings...")
    emb_norm = normalize(emb, norm="l2")

    n_components = min(PCA_COMPONENTS, emb_norm.shape[1], emb_norm.shape[0] - 1)

    print(f"Running PCA to {n_components} components...")
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    emb_pca = pca.fit_transform(emb_norm)

    print("Explained variance by PCA:", round(float(pca.explained_variance_ratio_.sum()), 4))

    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
        low_memory=True,
        verbose=True,
    )

    coords = reducer.fit_transform(emb_pca)
    return coords


def density_grid(x, y):
    x_min, x_max = np.percentile(x, [0.3, 99.7])
    y_min, y_max = np.percentile(y, [0.3, 99.7])

    # Add margin
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    H, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=GRID_SIZE,
        range=[[x_min, x_max], [y_min, y_max]],
    )

    H = gaussian_filter(H.T, sigma=DENSITY_SMOOTH_SIGMA)
    H = H / (H.max() if H.max() > 0 else 1)

    return H, xedges, yedges, (x_min, x_max, y_min, y_max)


def select_labels(df):
    """
    Select high-weight labels spread over the map.
    Similar idea to the example figure: label the highest-degree concept per local tile.
    """
    label_df = df.copy()

    label_df = label_df[label_df["weight"] >= MIN_LABEL_SCORE].copy()
    if len(label_df) == 0:
        label_df = df.copy()

    x = label_df["umap_1"].to_numpy()
    y = label_df["umap_2"].to_numpy()

    x_bins = pd.cut(x, bins=GRID_LABEL_BINS, labels=False, duplicates="drop")
    y_bins = pd.cut(y, bins=GRID_LABEL_BINS, labels=False, duplicates="drop")

    label_df["_xbin"] = x_bins
    label_df["_ybin"] = y_bins

    # One strongest label per grid cell
    tile_best = (
        label_df
        .sort_values("weight", ascending=False)
        .groupby(["_xbin", "_ybin"], as_index=False)
        .head(1)
    )

    # Then global top from tile representatives
    tile_best = tile_best.sort_values("weight", ascending=False).head(MAX_LABELS).copy()

    tile_best["plot_label"] = tile_best["label"].apply(short_label)

    return tile_best


def plot_map(df, labels):
    x = df["umap_1"].to_numpy()
    y = df["umap_2"].to_numpy()

    H, xedges, yedges, extent = density_grid(x, y)
    x_min, x_max, y_min, y_max = extent

    plt.rcParams.update({
        "font.size": 9,
        "font.family": "Arial",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Density background
    im = ax.imshow(
        H,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        cmap="viridis",
        alpha=0.95,
        aspect="auto",
        zorder=1,
    )

    # All points
    ax.scatter(
        x,
        y,
        s=POINT_SIZE,
        c="#2d1e6f",
        alpha=POINT_ALPHA,
        linewidths=0,
        zorder=2,
        label="Problem–solution pairs",
    )

    # Top labels
    ax.scatter(
        labels["umap_1"],
        labels["umap_2"],
        s=TOP_DOT_SIZE,
        c="red",
        alpha=TOP_DOT_ALPHA,
        edgecolors="white",
        linewidths=0.35,
        zorder=5,
        label="High-frequency pairs",
    )

    texts = []
    for _, row in labels.iterrows():
        txt = ax.text(
            row["umap_1"],
            row["umap_2"],
            row["plot_label"],
            fontsize=7.2,
            color="black",
            ha="center",
            va="center",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor="white",
                edgecolor="0.65",
                linewidth=0.35,
                alpha=0.82,
            ),
        )
        texts.append(txt)

    if HAS_ADJUSTTEXT and len(texts) > 0:
        print("Adjusting labels with adjustText...")
        adjust_text(
            texts,
            ax=ax,
            expand_text=(1.05, 1.15),
            expand_points=(1.05, 1.15),
            force_text=(0.12, 0.18),
            force_points=(0.08, 0.12),
            lim=200,
            arrowprops=dict(
                arrowstyle="-",
                color="0.45",
                lw=0.35,
                alpha=0.6,
            ),
        )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Kernel density estimate", rotation=270, labelpad=14, fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("UMAP component 1")
    ax.set_ylabel("UMAP component 2")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_title(TITLE, fontsize=13, weight="bold", pad=12)

    # Subtitle as figure text
    fig.text(
        0.125,
        0.035,
        SUBTITLE,
        ha="left",
        va="bottom",
        fontsize=8.4,
        wrap=True,
    )

    # Legend
    leg = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.82,
        fontsize=8,
        markerscale=2.2,
        borderpad=0.6,
    )
    leg.get_frame().set_linewidth(0.4)

    # Clean frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(False)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    print("Saving:", OUT_FIG_PNG)
    fig.savefig(OUT_FIG_PNG, dpi=DPI, bbox_inches="tight")

    print("Saving:", OUT_FIG_PDF)
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")

    plt.close(fig)


def main():
    print("============================================================")
    print("Surgical pair UMAP concept map")
    print("============================================================")

    df, emb = load_inputs()
    df_sel, emb_sel, selected_idx = select_points(df, emb)

    coords = compute_umap(emb_sel)

    df_sel["umap_1"] = coords[:, 0]
    df_sel["umap_2"] = coords[:, 1]
    df_sel["original_index"] = selected_idx

    df_sel.to_csv(OUT_UMAP_COORDS, index=False, encoding="utf-8-sig")
    print("Saved coordinates:", OUT_UMAP_COORDS)

    labels = select_labels(df_sel)
    labels.to_csv(OUT_LABELS_CSV, index=False, encoding="utf-8-sig")
    print("Saved selected labels:", OUT_LABELS_CSV)
    print("Number of labels:", len(labels))

    plot_map(df_sel, labels)

    print("============================================================")
    print("DONE")
    print("Figure PNG:", OUT_FIG_PNG)
    print("Figure PDF:", OUT_FIG_PDF)
    print("Coordinates:", OUT_UMAP_COORDS)
    print("Labels:", OUT_LABELS_CSV)
    print("============================================================")


if __name__ == "__main__":
    main()