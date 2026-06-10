from pathlib import Path
import json
import heapq
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

# Candidate mapping from previous candidate-space script
OPENALEX_TO_SOLUTION_TOPK = (
    BASE_DIR
    / "hypothesis_candidate_space_outputs"
    / "openalex_to_surgical_solution_cluster_topk.csv"
)

# Historical surgical cluster edges
CLUSTER_EDGES_FILE = (
    BASE_DIR
    / "surgical_time_normalized_cluster_edges"
    / "cluster_edges_time_normalized.csv"
)

# Prefer existing ML training dataset with precomputed features
MODEL_OUTPUT_DIR = BASE_DIR / "cluster_edge_strength_model_outputs"
ABLATION_OUTPUT_DIR = BASE_DIR / "cluster_edge_strength_ablation_outputs"

# Outputs
OUT_DIR = BASE_DIR / "openalex_ml_prediction_distribution_outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_EDGE_SCORES = OUT_DIR / "historical_cluster_edge_ml_predictions.csv"
OUT_DISTRIBUTION_CSV = OUT_DIR / "openalex_candidate_ml_prediction_distribution.csv"
OUT_THRESHOLD_CSV = OUT_DIR / "openalex_candidate_ml_prediction_threshold_counts.csv"
OUT_TOP_PREVIEW = OUT_DIR / "top_ml_predicted_openalex_problem_candidates_preview.csv"
OUT_SUMMARY_JSON = OUT_DIR / "openalex_candidate_ml_prediction_summary.json"

# Chunking
MAPPING_CHUNK_SIZE = 250_000

# Keep top candidates only for preview, not full 100M output
KEEP_TOP_N_PREVIEW = 100_000

# Probability thresholds to count
PROB_THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
    0.975,
    0.99,
    0.995,
]

# Histogram bins for approximate distribution / quantiles
N_HIST_BINS = 1000

# Expected feature names from your training summary
DEFAULT_FEATURE_COLS = [
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
            f"Could not find required column among: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def find_first_file(folder: Path, patterns: list[str]) -> Path | None:
    if not folder.exists():
        return None

    for pattern in patterns:
        hits = sorted(folder.glob(pattern))
        if hits:
            return hits[0]

    return None


def load_model_and_features():
    """
    Loads the trained surgical edge-strength classifier.
    Tries both main output and ablation output directories.
    """
    print("Loading trained ML model...")

    model_file = find_first_file(
        MODEL_OUTPUT_DIR,
        [
            "*model*.joblib",
            "*.joblib",
        ],
    )

    if model_file is None:
        model_file = find_first_file(
            ABLATION_OUTPUT_DIR,
            [
                "*combined*graph*embedding*.joblib",
                "*combined*.joblib",
                "*.joblib",
            ],
        )

    if model_file is None:
        raise FileNotFoundError(
            "Could not find a .joblib model in:\n"
            f"{MODEL_OUTPUT_DIR}\n"
            f"{ABLATION_OUTPUT_DIR}"
        )

    print("Model file:", model_file)

    obj = joblib.load(model_file)

    model = obj
    feature_cols = None

    if isinstance(obj, dict):
        for key in ["model", "classifier", "clf", "estimator"]:
            if key in obj:
                model = obj[key]
                break

        for key in ["feature_cols", "feature_names", "features", "feature_names_in_"]:
            if key in obj:
                feature_cols = list(obj[key])
                break

    if feature_cols is None and hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    print("Feature columns:")
    for c in feature_cols:
        print(" ", c)

    return model, feature_cols, model_file


def load_feature_table(feature_cols: list[str]) -> pd.DataFrame:
    """
    Loads precomputed training/scoring feature table if available.
    This is the safest route because it uses the exact features from your model pipeline.
    """
    print("\nSearching for precomputed edge feature/training dataset...")

    candidate_files = []

    if MODEL_OUTPUT_DIR.exists():
        candidate_files.extend(sorted(MODEL_OUTPUT_DIR.glob("*.csv")))

    if ABLATION_OUTPUT_DIR.exists():
        candidate_files.extend(sorted(ABLATION_OUTPUT_DIR.glob("*.csv")))

    usable = []

    for f in candidate_files:
        try:
            head = pd.read_csv(f, nrows=5, encoding="utf-8-sig")
        except Exception:
            continue

        pc_col = find_col(
            head,
            ["problem_cluster_id", "source_problem_cluster_id", "problem_cluster"],
            required=False,
        )

        sc_col = find_col(
            head,
            ["solution_cluster_id", "target_solution_cluster_id", "solution_cluster"],
            required=False,
        )

        has_features = all(c in head.columns for c in feature_cols)

        if pc_col and sc_col and has_features:
            usable.append(f)

    if not usable:
        raise FileNotFoundError(
            "Could not find a precomputed feature/training CSV containing:\n"
            "- problem_cluster_id\n"
            "- solution_cluster_id\n"
            "- all model feature columns\n\n"
            "Check cluster_edge_strength_model_outputs/ or cluster_edge_strength_ablation_outputs/."
        )

    feature_file = usable[0]
    print("Using feature table:", feature_file)

    df = pd.read_csv(feature_file, encoding="utf-8-sig", low_memory=False)

    pc_col = find_col(
        df,
        ["problem_cluster_id", "source_problem_cluster_id", "problem_cluster"],
        required=True,
    )

    sc_col = find_col(
        df,
        ["solution_cluster_id", "target_solution_cluster_id", "solution_cluster"],
        required=True,
    )

    df["_problem_cluster_id"] = df[pc_col].astype(int)
    df["_solution_cluster_id"] = df[sc_col].astype(int)

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Feature table missing model features: {missing_features}")

    print(f"Feature table rows: {len(df):,}")

    return df, feature_file


def predict_edge_probabilities(model, feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Predicts probability of historical upper-quintile-like edge.
    """
    print("\nPredicting ML probabilities for historical cluster edges...")

    X = feature_df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        proba = 1 / (1 + np.exp(-raw))
    else:
        raise TypeError("Loaded model has neither predict_proba nor decision_function.")

    out = feature_df[["_problem_cluster_id", "_solution_cluster_id"]].copy()
    out["ml_probability_top20_like"] = proba.astype(np.float32)

    # Remove duplicates if present: same problem-solution pair gets max probability
    out = (
        out
        .groupby(["_problem_cluster_id", "_solution_cluster_id"], as_index=False)
        .agg(ml_probability_top20_like=("ml_probability_top20_like", "max"))
    )

    print(f"Unique scored problem→solution cluster edges: {len(out):,}")

    out.to_csv(OUT_EDGE_SCORES, index=False, encoding="utf-8-sig")
    print("Saved edge ML predictions:", OUT_EDGE_SCORES)

    return out


def build_solution_to_problem_prob(edge_scores: pd.DataFrame):
    """
    solution_cluster_id -> list of (problem_cluster_id, ml_probability)
    """
    print("\nBuilding solution_cluster → problem_cluster probability lookup...")

    lookup = {}

    for sc, sub in edge_scores.groupby("_solution_cluster_id"):
        lookup[int(sc)] = list(
            zip(
                sub["_problem_cluster_id"].astype(int).values,
                sub["ml_probability_top20_like"].astype(float).values,
            )
        )

    print(f"Solution clusters with scored problem links: {len(lookup):,}")

    return lookup


def update_histogram(hist_counts: np.ndarray, scores: np.ndarray):
    """
    Scores are in [0,1]. Add to fixed-width histogram.
    """
    idx = np.floor(scores * N_HIST_BINS).astype(int)
    idx[idx < 0] = 0
    idx[idx >= N_HIST_BINS] = N_HIST_BINS - 1

    binc = np.bincount(idx, minlength=N_HIST_BINS)
    hist_counts += binc


def approximate_quantile_from_hist(hist_counts: np.ndarray, q: float) -> float:
    total = hist_counts.sum()
    if total == 0:
        return np.nan

    target = q * total
    cumsum = np.cumsum(hist_counts)
    bin_idx = np.searchsorted(cumsum, target)

    return float((bin_idx + 0.5) / N_HIST_BINS)


def process_concept_group(
    concept_id,
    concept_text,
    n_works,
    n_mentions,
    solution_clusters,
    solution_to_problem_prob,
    hist_counts,
    threshold_counts,
    top_heap,
):
    """
    For one OpenAlex concept:
    candidate problem score = max ML probability across all mapped solution clusters.
    Similarity is NOT used in the score.
    """
    problem_to_best_prob = {}

    for sc in solution_clusters:
        sc = int(sc)

        if sc not in solution_to_problem_prob:
            continue

        for pc, prob in solution_to_problem_prob[sc]:
            old = problem_to_best_prob.get(pc)
            if old is None or prob > old:
                problem_to_best_prob[pc] = prob

    if not problem_to_best_prob:
        return 0

    pcs = np.fromiter(problem_to_best_prob.keys(), dtype=np.int64)
    probs = np.fromiter(problem_to_best_prob.values(), dtype=np.float32)

    update_histogram(hist_counts, probs)

    for thr in PROB_THRESHOLDS:
        threshold_counts[thr] += int(np.sum(probs >= thr))

    # Keep top preview globally by ML probability only
    if KEEP_TOP_N_PREVIEW > 0:
        for pc, prob in zip(pcs, probs):
            item = (
                float(prob),
                int(concept_id),
                int(pc),
                str(concept_text),
                float(n_works) if pd.notna(n_works) else np.nan,
                float(n_mentions) if pd.notna(n_mentions) else np.nan,
            )

            if len(top_heap) < KEEP_TOP_N_PREVIEW:
                heapq.heappush(top_heap, item)
            else:
                if item[0] > top_heap[0][0]:
                    heapq.heapreplace(top_heap, item)

    return len(problem_to_best_prob)


# ============================================================
# MAIN
# ============================================================

def main():
    print("============================================================")
    print("OpenAlex candidate ML prediction distribution")
    print("STRICT SCORE = ML prediction only")
    print("No similarity term is used in the score.")
    print("============================================================")

    if not OPENALEX_TO_SOLUTION_TOPK.exists():
        raise FileNotFoundError(OPENALEX_TO_SOLUTION_TOPK)

    model, feature_cols, model_file = load_model_and_features()
    feature_df, feature_file = load_feature_table(feature_cols)
    edge_scores = predict_edge_probabilities(model, feature_df, feature_cols)
    solution_to_problem_prob = build_solution_to_problem_prob(edge_scores)

    print("\nProcessing OpenAlex → surgical solution-cluster mapping...")
    print("Mapping file:", OPENALEX_TO_SOLUTION_TOPK)
    print("Chunk size:", MAPPING_CHUNK_SIZE)

    hist_counts = np.zeros(N_HIST_BINS, dtype=np.int64)
    threshold_counts = {thr: 0 for thr in PROB_THRESHOLDS}
    top_heap = []

    total_candidate_pairs = 0
    total_openalex_concepts = 0
    openalex_with_candidates = 0

    carry = pd.DataFrame()

    use_cols = [
        "concept_id",
        "concept_text",
        "n_works",
        "n_mentions",
        "solution_cluster_id",
    ]

    chunk_iter = pd.read_csv(
        OPENALEX_TO_SOLUTION_TOPK,
        chunksize=MAPPING_CHUNK_SIZE,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=lambda c: c in use_cols,
    )

    for chunk_i, chunk in enumerate(chunk_iter, start=1):
        print(f"Chunk {chunk_i}: rows {len(chunk):,}")

        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = pd.DataFrame()

        # Ensure ordering
        chunk = chunk.sort_values(["concept_id"]).reset_index(drop=True)

        # Keep last concept as carry in case it continues in next chunk
        last_concept_id = chunk["concept_id"].iloc[-1]
        carry = chunk[chunk["concept_id"] == last_concept_id].copy()
        work = chunk[chunk["concept_id"] != last_concept_id].copy()

        if work.empty:
            continue

        grouped = work.groupby("concept_id", sort=False)

        for concept_id, g in grouped:
            first = g.iloc[0]

            solution_clusters = (
                g["solution_cluster_id"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            n_candidates = process_concept_group(
                concept_id=concept_id,
                concept_text=first.get("concept_text", ""),
                n_works=first.get("n_works", np.nan),
                n_mentions=first.get("n_mentions", np.nan),
                solution_clusters=solution_clusters,
                solution_to_problem_prob=solution_to_problem_prob,
                hist_counts=hist_counts,
                threshold_counts=threshold_counts,
                top_heap=top_heap,
            )

            total_openalex_concepts += 1
            total_candidate_pairs += n_candidates

            if n_candidates > 0:
                openalex_with_candidates += 1

        print(
            f"  processed concepts: {total_openalex_concepts:,} | "
            f"candidate pairs: {total_candidate_pairs:,}"
        )

    # Process final carry
    if not carry.empty:
        for concept_id, g in carry.groupby("concept_id", sort=False):
            first = g.iloc[0]

            solution_clusters = (
                g["solution_cluster_id"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            n_candidates = process_concept_group(
                concept_id=concept_id,
                concept_text=first.get("concept_text", ""),
                n_works=first.get("n_works", np.nan),
                n_mentions=first.get("n_mentions", np.nan),
                solution_clusters=solution_clusters,
                solution_to_problem_prob=solution_to_problem_prob,
                hist_counts=hist_counts,
                threshold_counts=threshold_counts,
                top_heap=top_heap,
            )

            total_openalex_concepts += 1
            total_candidate_pairs += n_candidates

            if n_candidates > 0:
                openalex_with_candidates += 1

    # ========================================================
    # Distribution outputs
    # ========================================================

    print("\nCreating distribution outputs...")

    bin_left = np.arange(N_HIST_BINS) / N_HIST_BINS
    bin_right = (np.arange(N_HIST_BINS) + 1) / N_HIST_BINS
    bin_mid = (bin_left + bin_right) / 2

    distribution_df = pd.DataFrame({
        "probability_bin_left": bin_left,
        "probability_bin_right": bin_right,
        "probability_bin_mid": bin_mid,
        "n_candidates": hist_counts,
    })

    distribution_df["fraction_candidates"] = (
        distribution_df["n_candidates"] / max(total_candidate_pairs, 1)
    )

    distribution_df.to_csv(OUT_DISTRIBUTION_CSV, index=False, encoding="utf-8-sig")

    threshold_rows = []

    for thr in PROB_THRESHOLDS:
        n = int(threshold_counts[thr])
        threshold_rows.append({
            "threshold": thr,
            "n_candidates_ge_threshold": n,
            "fraction_of_all_candidates": n / max(total_candidate_pairs, 1),
        })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(OUT_THRESHOLD_CSV, index=False, encoding="utf-8-sig")

    # Top preview
    if KEEP_TOP_N_PREVIEW > 0 and top_heap:
        top_sorted = sorted(top_heap, reverse=True)

        preview_df = pd.DataFrame(
            top_sorted,
            columns=[
                "ml_probability_top20_like",
                "concept_id",
                "problem_cluster_id",
                "concept_text",
                "n_works",
                "n_mentions",
            ],
        )

        preview_df.to_csv(OUT_TOP_PREVIEW, index=False, encoding="utf-8-sig")
    else:
        preview_df = pd.DataFrame()

    quantiles = {
        "p50": approximate_quantile_from_hist(hist_counts, 0.50),
        "p75": approximate_quantile_from_hist(hist_counts, 0.75),
        "p80": approximate_quantile_from_hist(hist_counts, 0.80),
        "p90": approximate_quantile_from_hist(hist_counts, 0.90),
        "p95": approximate_quantile_from_hist(hist_counts, 0.95),
        "p99": approximate_quantile_from_hist(hist_counts, 0.99),
        "p995": approximate_quantile_from_hist(hist_counts, 0.995),
    }

    summary = {
        "score_definition": (
            "Strictly ML prediction only. For each OpenAlex concept + surgical problem_cluster candidate, "
            "score = max predicted probability that the corresponding problem_cluster→mapped_solution_cluster "
            "resembles a historical upper-quintile surgical cluster edge. "
            "OpenAlex-to-solution similarity is not used in the score."
        ),
        "model_file": str(model_file),
        "feature_file": str(feature_file),
        "mapping_file": str(OPENALEX_TO_SOLUTION_TOPK),
        "total_openalex_concepts_processed": int(total_openalex_concepts),
        "openalex_concepts_with_candidates": int(openalex_with_candidates),
        "total_candidate_problem_concept_pairs": int(total_candidate_pairs),
        "threshold_counts": {
            str(k): {
                "n_candidates_ge_threshold": int(v),
                "fraction_of_all_candidates": float(v / max(total_candidate_pairs, 1)),
            }
            for k, v in threshold_counts.items()
        },
        "approx_quantiles": quantiles,
        "hist_bins": int(N_HIST_BINS),
        "keep_top_n_preview": int(KEEP_TOP_N_PREVIEW),
        "outputs": {
            "edge_scores": str(OUT_EDGE_SCORES),
            "distribution_csv": str(OUT_DISTRIBUTION_CSV),
            "threshold_csv": str(OUT_THRESHOLD_CSV),
            "top_preview_csv": str(OUT_TOP_PREVIEW),
            "summary_json": str(OUT_SUMMARY_JSON),
        },
    }

    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("============================================================")
    print("DONE")
    print("Total candidate problem_cluster + OpenAlex concept pairs:")
    print(f"{total_candidate_pairs:,}")

    print("\nApproximate ML probability quantiles:")
    for k, v in quantiles.items():
        print(f"{k}: {v:.4f}")

    print("\nThreshold counts:")
    for thr in PROB_THRESHOLDS:
        n = threshold_counts[thr]
        frac = n / max(total_candidate_pairs, 1)
        print(f"p >= {thr}: {n:,} ({frac:.4%})")

    print("\nOutputs:")
    print("Distribution:", OUT_DISTRIBUTION_CSV)
    print("Thresholds:", OUT_THRESHOLD_CSV)
    print("Top preview:", OUT_TOP_PREVIEW)
    print("Summary:", OUT_SUMMARY_JSON)
    print("============================================================")


if __name__ == "__main__":
    main()