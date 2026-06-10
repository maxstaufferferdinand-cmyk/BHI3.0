from pathlib import Path
import json
import heapq
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

# From previous candidate-space script
OPENALEX_TO_SOLUTION_TOPK = (
    BASE_DIR
    / "hypothesis_candidate_space_outputs"
    / "openalex_to_surgical_solution_cluster_topk.csv"
)

# From previous ML distribution script
EDGE_ML_PREDICTIONS = (
    BASE_DIR
    / "openalex_ml_prediction_distribution_outputs"
    / "historical_cluster_edge_ml_predictions.csv"
)

OUT_DIR = BASE_DIR / "openalex_high_ml_diversified_candidates_outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_STAGE1_TOP_PER_OPENALEX = OUT_DIR / "stage1_p99_top5_per_openalex_concept.csv"
OUT_STAGE2_TOP_PER_PROBLEM = OUT_DIR / "stage2_p99_top50_per_problem_cluster.csv"
OUT_STAGE3_TOP_PER_SOLUTION = OUT_DIR / "stage3_p99_top500_per_solution_cluster.csv"
OUT_FINAL = OUT_DIR / "final_diversified_high_ml_openalex_candidates.csv"
OUT_SUMMARY = OUT_DIR / "diversified_high_ml_candidate_selection_summary.json"

# Strict ML probability threshold
ML_PROB_THRESHOLD = 0.99

# Diversification limits
TOP_N_PER_OPENALEX_CONCEPT = 5
TOP_N_PER_PROBLEM_CLUSTER = 50
TOP_N_PER_SOLUTION_CLUSTER = 500

# Optional final cap. Set None to keep all after diversification.
FINAL_GLOBAL_TOP_N = 100_000

# Chunking
MAPPING_CHUNK_SIZE = 250_000

# Score column
SCORE_COL = "ml_probability_top20_like"


# ============================================================
# HELPERS
# ============================================================

def ensure_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def append_csv(df: pd.DataFrame, path: Path):
    header = not path.exists()
    df.to_csv(
        path,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8-sig",
    )


def load_solution_to_problem_ml_lookup(edge_scores_file: Path):
    print("Loading historical cluster-edge ML predictions...")
    edge_scores = pd.read_csv(edge_scores_file, encoding="utf-8-sig", low_memory=False)

    print(f"Edge-score rows: {len(edge_scores):,}")
    print("Columns:", list(edge_scores.columns))

    required = {"_problem_cluster_id", "_solution_cluster_id", SCORE_COL}
    missing = required - set(edge_scores.columns)
    if missing:
        raise ValueError(f"Edge-score file missing columns: {missing}")

    edge_scores["_problem_cluster_id"] = edge_scores["_problem_cluster_id"].astype(int)
    edge_scores["_solution_cluster_id"] = edge_scores["_solution_cluster_id"].astype(int)
    edge_scores[SCORE_COL] = pd.to_numeric(edge_scores[SCORE_COL], errors="coerce").fillna(0)

    # Keep only edges that can ever pass threshold
    edge_scores = edge_scores[edge_scores[SCORE_COL] >= ML_PROB_THRESHOLD].copy()

    print(f"Edges with ML p >= {ML_PROB_THRESHOLD}: {len(edge_scores):,}")

    lookup = {}

    for sc, sub in edge_scores.groupby("_solution_cluster_id"):
        lookup[int(sc)] = list(
            zip(
                sub["_problem_cluster_id"].astype(int).values,
                sub[SCORE_COL].astype(float).values,
            )
        )

    print(f"Solution clusters with high-ML problem links: {len(lookup):,}")

    return lookup, edge_scores


def process_one_openalex_concept_group(g: pd.DataFrame, solution_to_problem_ml: dict):
    """
    For one OpenAlex concept:
    - collect all problem clusters connected through mapped solution clusters
    - score = max ML probability over all mapped solution clusters
    - keep only p >= threshold
    - return top N per OpenAlex concept by ML probability
    Similarity is NOT used in the score.
    """
    first = g.iloc[0]

    concept_id = int(first["concept_id"])
    concept_text = first.get("concept_text", "")
    n_works = first.get("n_works", np.nan)
    n_mentions = first.get("n_mentions", np.nan)

    problem_best = {}

    solution_clusters = (
        g["solution_cluster_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    for sc in solution_clusters:
        if sc not in solution_to_problem_ml:
            continue

        for pc, prob in solution_to_problem_ml[sc]:
            if prob < ML_PROB_THRESHOLD:
                continue

            old = problem_best.get(pc)

            # Score is ML-only. Similarity is not used.
            if old is None or prob > old["ml_probability_top20_like"]:
                problem_best[pc] = {
                    "concept_id": concept_id,
                    "concept_text": concept_text,
                    "n_works": n_works,
                    "n_mentions": n_mentions,
                    "problem_cluster_id": int(pc),
                    "mapped_solution_cluster_id": int(sc),
                    "ml_probability_top20_like": float(prob),
                }

    if not problem_best:
        return pd.DataFrame()

    cand = pd.DataFrame(list(problem_best.values()))

    cand = (
        cand
        .sort_values(
            ["ml_probability_top20_like", "problem_cluster_id", "mapped_solution_cluster_id"],
            ascending=[False, True, True],
        )
        .head(TOP_N_PER_OPENALEX_CONCEPT)
        .copy()
    )

    cand["selection_stage"] = "stage1_top_per_openalex_concept"

    return cand


def top_n_per_group_file(
    input_file: Path,
    output_file: Path,
    group_col: str,
    top_n: int,
    stage_name: str,
):
    """
    Loads a manageable intermediate file and keeps top N per group.
    """
    print(f"\nApplying {stage_name}: top {top_n} per {group_col}")
    df = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)

    print(f"Input rows: {len(df):,}")

    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce").fillna(0)

    df = (
        df
        .sort_values(
            [group_col, SCORE_COL, "concept_id", "problem_cluster_id", "mapped_solution_cluster_id"],
            ascending=[True, False, True, True, True],
        )
        .groupby(group_col, as_index=False, group_keys=False)
        .head(top_n)
        .copy()
    )

    df["selection_stage"] = stage_name

    df = df.sort_values(SCORE_COL, ascending=False).reset_index(drop=True)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"Output rows: {len(df):,}")
    print("Saved:", output_file)

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("============================================================")
    print("Select diversified high-ML OpenAlex candidates")
    print("Score = ML probability only. No similarity term used.")
    print("============================================================")

    ensure_file(OPENALEX_TO_SOLUTION_TOPK)
    ensure_file(EDGE_ML_PREDICTIONS)

    # Remove old outputs to avoid accidental appending
    for f in [
        OUT_STAGE1_TOP_PER_OPENALEX,
        OUT_STAGE2_TOP_PER_PROBLEM,
        OUT_STAGE3_TOP_PER_SOLUTION,
        OUT_FINAL,
        OUT_SUMMARY,
    ]:
        if f.exists():
            f.unlink()

    solution_to_problem_ml, high_edge_scores = load_solution_to_problem_ml_lookup(
        EDGE_ML_PREDICTIONS
    )

    print("\nStage 1: p >= 0.99 and top 5 per OpenAlex concept")
    print("Mapping file:", OPENALEX_TO_SOLUTION_TOPK)

    use_cols = [
        "concept_id",
        "concept_text",
        "n_works",
        "n_mentions",
        "solution_cluster_id",
    ]

    carry = pd.DataFrame()
    total_concepts_processed = 0
    total_stage1_rows = 0
    concepts_with_stage1_candidates = 0

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

        chunk = chunk.sort_values("concept_id").reset_index(drop=True)

        # Carry last concept because its rows may continue in next chunk
        last_concept_id = chunk["concept_id"].iloc[-1]
        carry = chunk[chunk["concept_id"] == last_concept_id].copy()
        work = chunk[chunk["concept_id"] != last_concept_id].copy()

        if work.empty:
            continue

        out_parts = []

        for concept_id, g in work.groupby("concept_id", sort=False):
            cand = process_one_openalex_concept_group(g, solution_to_problem_ml)
            total_concepts_processed += 1

            if not cand.empty:
                concepts_with_stage1_candidates += 1
                total_stage1_rows += len(cand)
                out_parts.append(cand)

        if out_parts:
            out_df = pd.concat(out_parts, ignore_index=True)
            append_csv(out_df, OUT_STAGE1_TOP_PER_OPENALEX)

        print(
            f"  processed concepts: {total_concepts_processed:,} | "
            f"stage1 rows: {total_stage1_rows:,}"
        )

    # Process carry
    if not carry.empty:
        out_parts = []

        for concept_id, g in carry.groupby("concept_id", sort=False):
            cand = process_one_openalex_concept_group(g, solution_to_problem_ml)
            total_concepts_processed += 1

            if not cand.empty:
                concepts_with_stage1_candidates += 1
                total_stage1_rows += len(cand)
                out_parts.append(cand)

        if out_parts:
            out_df = pd.concat(out_parts, ignore_index=True)
            append_csv(out_df, OUT_STAGE1_TOP_PER_OPENALEX)

    print("\nStage 1 complete.")
    print(f"Concepts processed: {total_concepts_processed:,}")
    print(f"Concepts with candidates: {concepts_with_stage1_candidates:,}")
    print(f"Stage 1 rows: {total_stage1_rows:,}")
    print("Saved:", OUT_STAGE1_TOP_PER_OPENALEX)

    if total_stage1_rows == 0:
        raise RuntimeError("No candidates passed p >= 0.99 after top-per-OpenAlex selection.")

    # Stage 2: top per problem cluster
    stage2_df = top_n_per_group_file(
        input_file=OUT_STAGE1_TOP_PER_OPENALEX,
        output_file=OUT_STAGE2_TOP_PER_PROBLEM,
        group_col="problem_cluster_id",
        top_n=TOP_N_PER_PROBLEM_CLUSTER,
        stage_name="stage2_top_per_problem_cluster",
    )

    # Stage 3: top per mapped solution cluster
    stage3_df = top_n_per_group_file(
        input_file=OUT_STAGE2_TOP_PER_PROBLEM,
        output_file=OUT_STAGE3_TOP_PER_SOLUTION,
        group_col="mapped_solution_cluster_id",
        top_n=TOP_N_PER_SOLUTION_CLUSTER,
        stage_name="stage3_top_per_solution_cluster",
    )

    # Final global cap
    print("\nFinal global selection...")
    final_df = stage3_df.copy()

    final_df = final_df.sort_values(
        [SCORE_COL, "concept_id", "problem_cluster_id", "mapped_solution_cluster_id"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    if FINAL_GLOBAL_TOP_N is not None:
        final_df = final_df.head(FINAL_GLOBAL_TOP_N).copy()

    final_df["final_rank"] = np.arange(1, len(final_df) + 1)

    final_df.to_csv(OUT_FINAL, index=False, encoding="utf-8-sig")

    print(f"Final rows: {len(final_df):,}")
    print("Saved:", OUT_FINAL)

    # Summary
    summary = {
        "score_definition": (
            "Strict ML-only score. Candidate score is ml_probability_top20_like, "
            "the predicted probability that the mapped surgical problem_cluster→solution_cluster "
            "resembles a historical upper-quintile surgical cluster edge. "
            "OpenAlex-to-solution similarity is not used in the score."
        ),
        "ml_probability_threshold": float(ML_PROB_THRESHOLD),
        "top_n_per_openalex_concept": int(TOP_N_PER_OPENALEX_CONCEPT),
        "top_n_per_problem_cluster": int(TOP_N_PER_PROBLEM_CLUSTER),
        "top_n_per_solution_cluster": int(TOP_N_PER_SOLUTION_CLUSTER),
        "final_global_top_n": FINAL_GLOBAL_TOP_N,
        "inputs": {
            "openalex_to_solution_topk": str(OPENALEX_TO_SOLUTION_TOPK),
            "edge_ml_predictions": str(EDGE_ML_PREDICTIONS),
        },
        "counts": {
            "high_ml_edges_p99": int(len(high_edge_scores)),
            "concepts_processed": int(total_concepts_processed),
            "concepts_with_stage1_candidates": int(concepts_with_stage1_candidates),
            "stage1_rows_top_per_openalex": int(total_stage1_rows),
            "stage2_rows_top_per_problem": int(len(stage2_df)),
            "stage3_rows_top_per_solution": int(len(stage3_df)),
            "final_rows": int(len(final_df)),
            "unique_openalex_concepts_final": int(final_df["concept_id"].nunique()),
            "unique_problem_clusters_final": int(final_df["problem_cluster_id"].nunique()),
            "unique_solution_clusters_final": int(final_df["mapped_solution_cluster_id"].nunique()),
        },
        "score_distribution_final": {
            "min": float(final_df[SCORE_COL].min()),
            "median": float(final_df[SCORE_COL].median()),
            "mean": float(final_df[SCORE_COL].mean()),
            "max": float(final_df[SCORE_COL].max()),
        },
        "outputs": {
            "stage1": str(OUT_STAGE1_TOP_PER_OPENALEX),
            "stage2": str(OUT_STAGE2_TOP_PER_PROBLEM),
            "stage3": str(OUT_STAGE3_TOP_PER_SOLUTION),
            "final": str(OUT_FINAL),
            "summary": str(OUT_SUMMARY),
        },
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("============================================================")
    print("DONE")
    print("Summary:")
    print(json.dumps(summary["counts"], indent=2))
    print("\nFinal candidate file:")
    print(OUT_FINAL)
    print("============================================================")


if __name__ == "__main__":
    main()