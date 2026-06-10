from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from adjustText import adjust_text
import umap


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

# -----------------------------
# Surgical pair map inputs
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
# Surgical solution-space inputs
# -----------------------------
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
OUT_DIR = BASE_DIR / "surgical_pair_map_projected_openalex_outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_PNG = OUT_DIR / "figure_surgical_pair_map_with_projected_openalex.png"
OUT_PDF = OUT_DIR / "figure_surgical_pair_map_with_projected_openalex.pdf"
OUT_SURGICAL_COORDS = OUT_DIR / "surgical_pair_umap_coordinates.csv"
OUT_OPENALEX_PROJECTION = OUT_DIR / "openalex_projected_to_surgical_pair_map.csv"
OUT_LABELS = OUT_DIR / "projected_openalex_map_labels.csv"
OUT_SUMMARY = OUT_DIR / "projection_summary.json"

# -----------------------------
# Sampling and projection
# -----------------------------
TOP_N_SURGICAL_PAIRS = 30000
TOP_N_OPENALEX = 3000

# For each OpenAlex concept: nearest surgical solution embeddings
TOP_K_SOLUTION_NEIGHBORS = 20

# For plotting: only show projected OpenAlex points above this similarity
MIN_BEST_COSINE_FOR_PLOT = 0.25

# Labels
MAX_SURGICAL_LABELS = 18
MAX_OPENALEX_LABELS = 22

# UMAP
PCA_COMPONENTS = 50
UMAP_N_NEIGHBORS = 25
UMAP_MIN_DIST = 0.08
RANDOM_STATE = 42

# Density background
DENSITY_BINS = 430
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


def short_label(text: str, max_len: int = 48) -> str:
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


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


def make_density_background(x, y, bins=400, sigma=4):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap.T, sigma=sigma)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap, extent


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms


def weighted_xy(coords: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=float)
    weights[weights < 0] = 0

    if weights.sum() <= 0:
        return float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1]))

    weights = weights / weights.sum()
    xy = coords.T @ weights
    return float(xy[0]), float(xy[1])


# ============================================================
# LOAD SURGICAL PAIR MAP
# ============================================================

def load_surgical_pairs():
    print("Loading surgical pair table and embeddings...")

    if not SURGICAL_EDGES_FILE.exists():
        raise FileNotFoundError(f"Missing surgical edges file: {SURGICAL_EDGES_FILE}")

    if not SURGICAL_PAIR_EMB_FILE.exists():
        raise FileNotFoundError(f"Missing surgical pair embeddings: {SURGICAL_PAIR_EMB_FILE}")

    edges = pd.read_csv(SURGICAL_EDGES_FILE, encoding="utf-8-sig", low_memory=False)
    pair_emb = np.load(SURGICAL_PAIR_EMB_FILE, mmap_mode="r")

    print(f"Surgical edge rows: {len(edges):,}")
    print(f"Surgical pair embeddings: {pair_emb.shape}")
    print("Surgical edge columns:", list(edges.columns))

    if len(edges) != pair_emb.shape[0]:
        raise ValueError(
            "surgical_graph_edges_aggregated_with_years.csv and surgical_pair_embeddings.npy "
            "do not have the same number of rows. Need exact pair embedding index."
        )

    problem_col = find_col(
        edges,
        [
            "surgical_problem",
            "problem_text",
            "problem",
            "problem_label",
            "problem_node_text",
            "problem_concept",
        ],
        required=False,
    )

    solution_col = find_col(
        edges,
        [
            "technical_solution",
            "solution_text",
            "solution",
            "solution_label",
            "solution_node_text",
            "solution_concept",
        ],
        required=False,
    )

    problem_id_col = find_col(
        edges,
        [
            "problem_id",
            "source_problem_id",
            "surgical_problem_id",
        ],
        required=False,
    )

    solution_id_col = find_col(
        edges,
        [
            "solution_id",
            "target_solution_id",
            "technical_solution_id",
        ],
        required=True,
    )

    weight_col = find_col(
        edges,
        [
            "weight",
            "edge_weight",
            "n_pmids",
            "count",
            "n",
        ],
        required=False,
    )

    first_year_col = find_col(
        edges,
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
    print("  problem_id:", problem_id_col)
    print("  solution_id:", solution_id_col)
    print("  weight:", weight_col)
    print("  first_year:", first_year_col)

    if problem_col and solution_col:
        edges["_pair_label"] = [
            f"{clean_text(p)} → {clean_text(s)}"
            for p, s in zip(edges[problem_col], edges[solution_col])
        ]
    else:
        edges["_pair_label"] = [f"surgical pair {i}" for i in range(len(edges))]

    if weight_col:
        edges["_weight"] = pd.to_numeric(edges[weight_col], errors="coerce").fillna(0)
    else:
        edges["_weight"] = 1.0

    if first_year_col:
        edges["_first_year"] = pd.to_numeric(edges[first_year_col], errors="coerce")
    else:
        edges["_first_year"] = np.nan

    edges["_source_row"] = np.arange(len(edges))
    edges["_solution_id"] = edges[solution_id_col].astype(str)

    selected = (
        edges
        .sort_values("_weight", ascending=False)
        .head(TOP_N_SURGICAL_PAIRS)
        .copy()
        .reset_index(drop=True)
    )

    selected_emb = np.array(pair_emb[selected["_source_row"].values], dtype=np.float32)

    print(f"Selected surgical pairs for UMAP: {len(selected):,}")

    return selected, selected_emb, solution_id_col


# ============================================================
# LOAD SURGICAL SOLUTION SPACE
# ============================================================

def load_surgical_solutions():
    print("Loading surgical solution index and embeddings...")

    if not SURGICAL_SOLUTION_INDEX_FILE.exists():
        raise FileNotFoundError(f"Missing surgical solution index: {SURGICAL_SOLUTION_INDEX_FILE}")

    if not SURGICAL_SOLUTION_EMB_FILE.exists():
        raise FileNotFoundError(f"Missing surgical solution embeddings: {SURGICAL_SOLUTION_EMB_FILE}")

    sol_index = pd.read_csv(SURGICAL_SOLUTION_INDEX_FILE, encoding="utf-8-sig", low_memory=False)
    sol_emb = np.load(SURGICAL_SOLUTION_EMB_FILE, mmap_mode="r")

    print(f"Surgical solution index rows: {len(sol_index):,}")
    print(f"Surgical solution embeddings: {sol_emb.shape}")
    print("Solution index columns:", list(sol_index.columns))

    if len(sol_index) != sol_emb.shape[0]:
        raise ValueError("Surgical solution index and embedding rows do not match.")

    solution_id_col = find_col(
        sol_index,
        [
            "solution_id",
            "id",
            "node_id",
            "technical_solution_id",
        ],
        required=True,
    )

    solution_text_col = find_col(
        sol_index,
        [
            "technical_solution",
            "solution_text",
            "solution",
            "node_text",
            "concept_text",
            "text",
        ],
        required=False,
    )

    print("Detected solution index columns:")
    print("  solution_id:", solution_id_col)
    print("  solution_text:", solution_text_col)

    sol_index["_solution_id"] = sol_index[solution_id_col].astype(str)

    if solution_text_col:
        sol_index["_solution_text"] = sol_index[solution_text_col].apply(clean_text)
    else:
        sol_index["_solution_text"] = sol_index["_solution_id"]

    return sol_index, sol_emb


# ============================================================
# LOAD OPENALEX
# ============================================================

def load_openalex():
    print("Loading OpenAlex concepts and embeddings...")

    if not OPENALEX_INDEX_FILE.exists():
        raise FileNotFoundError(f"Missing OpenAlex index: {OPENALEX_INDEX_FILE}")

    if not OPENALEX_EMB_FILE.exists():
        raise FileNotFoundError(f"Missing OpenAlex embeddings: {OPENALEX_EMB_FILE}")

    oa = pd.read_csv(OPENALEX_INDEX_FILE, encoding="utf-8-sig")
    oa_emb = np.load(OPENALEX_EMB_FILE, mmap_mode="r")

    print(f"OpenAlex index rows: {len(oa):,}")
    print(f"OpenAlex embeddings: {oa_emb.shape}")
    print("OpenAlex columns:", list(oa.columns))

    if len(oa) != oa_emb.shape[0]:
        raise ValueError("OpenAlex index and embeddings do not match.")

    required = {"concept_id", "concept_text"}
    missing = required - set(oa.columns)
    if missing:
        raise ValueError(f"OpenAlex index missing required columns: {missing}")

    if "n_works" not in oa.columns:
        oa["n_works"] = 1

    if "n_mentions" not in oa.columns:
        oa["n_mentions"] = 1

    oa = (
        oa
        .sort_values(["n_works", "n_mentions"], ascending=[False, False])
        .head(TOP_N_OPENALEX)
        .copy()
        .reset_index(drop=True)
    )

    oa_emb_sel = np.array(oa_emb[oa["concept_id"].values], dtype=np.float32)

    print(f"Selected OpenAlex concepts for projection: {len(oa):,}")

    return oa, oa_emb_sel


# ============================================================
# MAIN
# ============================================================

def main():
    print("============================================================")
    print("Surgical pair map with OpenAlex projected via surgical solution space")
    print("============================================================")

    surgical_pairs, surgical_pair_emb, pair_solution_id_col = load_surgical_pairs()
    solution_index, solution_emb = load_surgical_solutions()
    openalex_df, openalex_emb = load_openalex()

    if solution_emb.shape[1] != openalex_emb.shape[1]:
        raise ValueError(
            f"Dimension mismatch: solution embeddings {solution_emb.shape[1]} vs OpenAlex {openalex_emb.shape[1]}"
        )

    # --------------------------------------------------------
    # UMAP only on surgical pair embeddings
    # --------------------------------------------------------
    print("Running PCA on surgical pair embeddings...")

    pca = PCA(
        n_components=min(PCA_COMPONENTS, surgical_pair_emb.shape[1]),
        random_state=RANDOM_STATE,
    )

    surgical_pair_pca = pca.fit_transform(surgical_pair_emb)

    print("Running UMAP on surgical pair map...")

    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="cosine",
        random_state=RANDOM_STATE,
        verbose=True,
    )

    surgical_coords = umap_model.fit_transform(surgical_pair_pca)

    surgical_pairs["umap_x"] = surgical_coords[:, 0]
    surgical_pairs["umap_y"] = surgical_coords[:, 1]

    # --------------------------------------------------------
    # Build solution_id -> coordinates of surgical pairs
    # --------------------------------------------------------
    print("Building solution-to-pair-coordinate lookup...")

    solution_pair_lookup = {}

    for _, row in surgical_pairs.iterrows():
        sid = str(row["_solution_id"])
        solution_pair_lookup.setdefault(sid, []).append(
            {
                "x": float(row["umap_x"]),
                "y": float(row["umap_y"]),
                "weight": float(row["_weight"]),
                "pair_label": row["_pair_label"],
            }
        )

    print(f"Solutions represented in surgical pair UMAP: {len(solution_pair_lookup):,}")

    # --------------------------------------------------------
    # Nearest surgical solutions for OpenAlex concepts
    # --------------------------------------------------------
    print("Fitting nearest-neighbor model in surgical solution embedding space...")

    solution_emb_norm = normalize_rows(np.array(solution_emb, dtype=np.float32))
    openalex_emb_norm = normalize_rows(openalex_emb)

    nn = NearestNeighbors(
        n_neighbors=TOP_K_SOLUTION_NEIGHBORS,
        metric="cosine",
        algorithm="brute",
    )

    nn.fit(solution_emb_norm)

    print("Querying nearest surgical solutions for OpenAlex concepts...")

    distances, indices = nn.kneighbors(openalex_emb_norm, return_distance=True)

    # cosine similarity = 1 - cosine distance
    similarities = 1.0 - distances

    # --------------------------------------------------------
    # Project OpenAlex to surgical pair UMAP coordinates
    # --------------------------------------------------------
    print("Projecting OpenAlex concepts onto surgical pair map coordinates...")

    projected_rows = []

    for i, oa_row in openalex_df.iterrows():
        candidate_coords = []
        candidate_weights = []
        matched_solution_texts = []
        matched_pair_labels = []

        for rank in range(TOP_K_SOLUTION_NEIGHBORS):
            sol_idx = int(indices[i, rank])
            sim = float(similarities[i, rank])

            sol_id = str(solution_index.iloc[sol_idx]["_solution_id"])
            sol_text = solution_index.iloc[sol_idx]["_solution_text"]

            if sol_id not in solution_pair_lookup:
                continue

            pair_entries = solution_pair_lookup[sol_id]

            for pe in pair_entries[:10]:
                candidate_coords.append([pe["x"], pe["y"]])

                # weight = semantic similarity * surgical pair weight
                w = max(sim, 0) * np.log1p(max(pe["weight"], 0))
                candidate_weights.append(w)

                if len(matched_pair_labels) < 5:
                    matched_pair_labels.append(pe["pair_label"])

            if len(matched_solution_texts) < 8:
                matched_solution_texts.append(f"{sol_text} ({sim:.3f})")

        if len(candidate_coords) == 0:
            # fallback: use nearest solution if it has no selected-pair representation
            # mark as not plottable
            projected_rows.append({
                "concept_id": int(oa_row["concept_id"]),
                "concept_text": oa_row["concept_text"],
                "n_works": oa_row.get("n_works", np.nan),
                "n_mentions": oa_row.get("n_mentions", np.nan),
                "best_solution_cosine": float(similarities[i, 0]),
                "projected_x": np.nan,
                "projected_y": np.nan,
                "n_coordinate_support_points": 0,
                "matched_solutions": " | ".join(matched_solution_texts),
                "matched_surgical_pairs": "",
                "plotted": False,
            })
            continue

        coords_arr = np.array(candidate_coords, dtype=float)
        weights_arr = np.array(candidate_weights, dtype=float)

        px, py = weighted_xy(coords_arr, weights_arr)

        projected_rows.append({
            "concept_id": int(oa_row["concept_id"]),
            "concept_text": oa_row["concept_text"],
            "n_works": oa_row.get("n_works", np.nan),
            "n_mentions": oa_row.get("n_mentions", np.nan),
            "best_solution_cosine": float(similarities[i, 0]),
            "mean_topk_solution_cosine": float(np.mean(similarities[i, :])),
            "projected_x": px,
            "projected_y": py,
            "n_coordinate_support_points": int(len(candidate_coords)),
            "matched_solutions": " | ".join(matched_solution_texts),
            "matched_surgical_pairs": " | ".join(matched_pair_labels),
            "plotted": True,
        })

    projected_df = pd.DataFrame(projected_rows)

    projected_plot = projected_df[
        (projected_df["plotted"] == True)
        & (projected_df["best_solution_cosine"] >= MIN_BEST_COSINE_FOR_PLOT)
    ].copy()

    print(f"Projected OpenAlex concepts with valid coordinates: {projected_df['plotted'].sum():,}")
    print(f"Projected OpenAlex concepts plotted after similarity filter: {len(projected_plot):,}")

    # --------------------------------------------------------
    # Select labels
    # --------------------------------------------------------
    print("Selecting labels...")

    surgical_label_df = (
        surgical_pairs
        .sort_values("_weight", ascending=False)
        .head(MAX_SURGICAL_LABELS)
        .copy()
    )

    surgical_label_df["domain"] = "Surgical"
    surgical_label_df["label_text"] = surgical_label_df["_pair_label"].apply(lambda x: short_label(x, 55))
    surgical_label_df["x"] = surgical_label_df["umap_x"]
    surgical_label_df["y"] = surgical_label_df["umap_y"]

    openalex_label_df = (
        projected_plot
        .sort_values(["n_works", "best_solution_cosine"], ascending=[False, False])
        .head(MAX_OPENALEX_LABELS)
        .copy()
    )

    openalex_label_df["domain"] = "OpenAlex projected"
    openalex_label_df["label_text"] = openalex_label_df["concept_text"].apply(lambda x: short_label(x, 45))
    openalex_label_df["x"] = openalex_label_df["projected_x"]
    openalex_label_df["y"] = openalex_label_df["projected_y"]

    labels = pd.concat(
        [
            surgical_label_df[["domain", "label_text", "x", "y"]],
            openalex_label_df[["domain", "label_text", "x", "y"]],
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Density background
    # --------------------------------------------------------
    print("Creating density background...")

    density, extent = make_density_background(
        surgical_pairs["umap_x"].values,
        surgical_pairs["umap_y"].values,
        bins=DENSITY_BINS,
        sigma=DENSITY_SIGMA,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    print("Plotting...")

    plt.figure(figsize=(17, 12.5))

    plt.imshow(
        density,
        extent=extent,
        origin="lower",
        cmap="Greys",
        alpha=0.32,
        aspect="auto",
    )

    # surgical background
    plt.scatter(
        surgical_pairs["umap_x"],
        surgical_pairs["umap_y"],
        s=5,
        c="#d62728",
        alpha=0.35,
        linewidths=0,
        label="Surgical problem–solution pairs",
        zorder=2,
    )

    # projected OpenAlex
    if len(projected_plot) > 0:
        plt.scatter(
            projected_plot["projected_x"],
            projected_plot["projected_y"],
            s=11,
            c="#1f77b4",
            alpha=0.55,
            linewidths=0,
            label="OpenAlex concepts projected via nearest surgical solutions",
            zorder=3,
        )

        # stronger points by best cosine
        high = projected_plot.sort_values("best_solution_cosine", ascending=False).head(200)
        plt.scatter(
            high["projected_x"],
            high["projected_y"],
            s=18,
            c="#00bcd4",
            alpha=0.75,
            edgecolors="white",
            linewidths=0.3,
            label="High-similarity projected OpenAlex concepts",
            zorder=4,
        )

    # labels
    texts = []

    if len(labels) > 0:
        surg_labels = labels[labels["domain"] == "Surgical"]
        oa_labels = labels[labels["domain"] == "OpenAlex projected"]

        if len(surg_labels) > 0:
            plt.scatter(
                surg_labels["x"],
                surg_labels["y"],
                s=38,
                c="#ff3b30",
                edgecolors="white",
                linewidths=0.5,
                alpha=0.95,
                zorder=6,
            )

        if len(oa_labels) > 0:
            plt.scatter(
                oa_labels["x"],
                oa_labels["y"],
                s=38,
                c="#00bcd4",
                edgecolors="white",
                linewidths=0.5,
                alpha=0.95,
                zorder=6,
            )

        for _, row in labels.iterrows():
            if row["domain"] == "Surgical":
                face_color = "#fff1f0"
                edge_color = "#d62728"
            else:
                face_color = "#eef6ff"
                edge_color = "#1f77b4"

            txt = plt.text(
                row["x"],
                row["y"],
                row["label_text"],
                fontsize=7.0,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor=face_color,
                    edgecolor=edge_color,
                    alpha=0.90,
                    linewidth=0.5,
                ),
                zorder=7,
            )
            texts.append(txt)

        print("Adjusting labels...")

        adjust_text(
            texts,
            x=labels["x"].values,
            y=labels["y"].values,
            expand_text=(1.25, 1.45),
            expand_points=(1.35, 1.55),
            force_text=(0.55, 0.95),
            force_points=(0.25, 0.45),
            lim=600,
            arrowprops=dict(
                arrowstyle="-",
                color="gray",
                lw=0.45,
                alpha=0.65,
            ),
        )

    plt.title(
        "Surgical problem–solution map with OpenAlex concepts projected through surgical solution space",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )

    plt.xlabel("Surgical pair UMAP component 1", fontsize=11)
    plt.ylabel("Surgical pair UMAP component 2", fontsize=11)

    plt.legend(
        loc="lower right",
        frameon=True,
        fontsize=8.5,
        markerscale=1.7,
    )

    caption = (
        "Surgical problem–solution pairs define the UMAP base map. "
        "OpenAlex engineering mechanisms are projected onto this map through nearest-neighbor matching "
        "in surgical technical-solution embedding space; projected positions are weighted by nearest surgical-solution similarity "
        "and supporting surgical pair frequency."
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
    # Save outputs
    # --------------------------------------------------------
    print("Saving CSV outputs...")

    surgical_pairs.to_csv(OUT_SURGICAL_COORDS, index=False, encoding="utf-8-sig")
    projected_df.to_csv(OUT_OPENALEX_PROJECTION, index=False, encoding="utf-8-sig")
    labels.to_csv(OUT_LABELS, index=False, encoding="utf-8-sig")

    summary = {
        "method": "Surgical pair UMAP base map; OpenAlex projected via nearest surgical solution embeddings.",
        "surgical_edges_file": str(SURGICAL_EDGES_FILE),
        "surgical_pair_embedding_file": str(SURGICAL_PAIR_EMB_FILE),
        "surgical_solution_index_file": str(SURGICAL_SOLUTION_INDEX_FILE),
        "surgical_solution_embedding_file": str(SURGICAL_SOLUTION_EMB_FILE),
        "openalex_index_file": str(OPENALEX_INDEX_FILE),
        "openalex_embedding_file": str(OPENALEX_EMB_FILE),
        "top_n_surgical_pairs": int(TOP_N_SURGICAL_PAIRS),
        "top_n_openalex": int(TOP_N_OPENALEX),
        "top_k_solution_neighbors": int(TOP_K_SOLUTION_NEIGHBORS),
        "min_best_cosine_for_plot": float(MIN_BEST_COSINE_FOR_PLOT),
        "n_projected_valid": int(projected_df["plotted"].sum()),
        "n_projected_plotted": int(len(projected_plot)),
        "outputs": {
            "png": str(OUT_PNG),
            "pdf": str(OUT_PDF),
            "surgical_coords_csv": str(OUT_SURGICAL_COORDS),
            "openalex_projection_csv": str(OUT_OPENALEX_PROJECTION),
            "labels_csv": str(OUT_LABELS),
        },
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved surgical coordinates:", OUT_SURGICAL_COORDS)
    print("Saved OpenAlex projection:", OUT_OPENALEX_PROJECTION)
    print("Saved labels:", OUT_LABELS)
    print("Saved summary:", OUT_SUMMARY)

    print("============================================================")
    print("DONE")
    print("PNG:")
    print(OUT_PNG)
    print("============================================================")


if __name__ == "__main__":
    main()