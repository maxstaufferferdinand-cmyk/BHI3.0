from pathlib import Path
import json
import re
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

FINAL_CANDIDATES_FILE = (
    BASE_DIR
    / "openalex_high_ml_diversified_candidates_outputs"
    / "final_diversified_high_ml_openalex_candidates.csv"
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

CLUSTER_EDGES_FILE = (
    BASE_DIR
    / "surgical_time_normalized_cluster_edges"
    / "cluster_edges_time_normalized.csv"
)

OUT_DIR = BASE_DIR / "openalex_hypothesis_annotation_outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_ANNOTATED = OUT_DIR / "final_diversified_high_ml_openalex_candidates_annotated.csv"
OUT_TOP1000 = OUT_DIR / "top1000_for_manual_review.csv"
OUT_TOP10000 = OUT_DIR / "top10000_for_llm.csv"
OUT_SUMMARY = OUT_DIR / "candidate_annotation_summary.json"

N_TERMS_PER_CLUSTER = 8

TOP1000_PER_PROBLEM = 2
TOP1000_GLOBAL_FILL = 1000

TOP10000_PER_PROBLEM = 10
TOP10000_GLOBAL_FILL = 10000

SCORE_COL = "ml_probability_top20_like"


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
            f"Could not find required column among {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def make_cluster_label(terms: list[str], max_terms: int = 5) -> str:
    terms = [clean_text(t) for t in terms if clean_text(t)]
    terms = list(dict.fromkeys(terms))
    return " | ".join(terms[:max_terms])


def build_cluster_term_table(
    node_cluster_file: Path,
    side: str,
    n_terms: int = 8,
) -> pd.DataFrame:
    print(f"\nLoading {side} node clusters...")

    df = pd.read_csv(node_cluster_file, encoding="utf-8-sig", low_memory=False)

    print(f"{side} node cluster rows: {len(df):,}")
    print(f"{side} columns:", list(df.columns))

    cluster_col = find_col(
        df,
        [
            f"{side}_cluster_id",
            "cluster_id",
            f"{side}_cluster",
        ],
        required=True,
    )

    text_col = find_col(
        df,
        [
            f"{side}_text",
            side,
            "node_text",
            "concept_text",
            "text",
            f"{side}_concept",
            f"{side}_name",
        ],
        required=False,
    )

    size_col = find_col(
        df,
        [
            "cluster_size",
            "n_nodes",
            "size",
        ],
        required=False,
    )

    print(f"Detected {side} cluster_col:", cluster_col)
    print(f"Detected {side} text_col:", text_col)
    print(f"Detected {side} size_col:", size_col)

    df["_cluster_id"] = df[cluster_col].astype(int)

    if text_col is None:
        # If no term column exists, use cluster id as fallback
        out = (
            df[["_cluster_id"]]
            .drop_duplicates()
            .rename(columns={"_cluster_id": f"{side}_cluster_id"})
        )
        out[f"{side}_cluster_label"] = out[f"{side}_cluster_id"].apply(lambda x: f"{side}_cluster_{x}")
        out[f"{side}_cluster_top_terms"] = out[f"{side}_cluster_label"]
        out[f"{side}_cluster_n_nodes"] = np.nan
        return out

    df["_term"] = df[text_col].apply(clean_text)
    df = df[df["_term"].str.len() > 0].copy()

    # If there is a frequency-like column, sort by it; otherwise preserve order.
    freq_col = find_col(
        df,
        [
            "n_mentions",
            "count",
            "frequency",
            "freq",
            "weight",
        ],
        required=False,
    )

    if freq_col:
        df["_freq"] = pd.to_numeric(df[freq_col], errors="coerce").fillna(1)
        df = df.sort_values(["_cluster_id", "_freq"], ascending=[True, False])
    else:
        df["_freq"] = 1

    rows = []

    for cid, sub in df.groupby("_cluster_id"):
        terms = sub["_term"].tolist()
        unique_terms = list(dict.fromkeys(terms))
        top_terms = unique_terms[:n_terms]

        rows.append(
            {
                f"{side}_cluster_id": int(cid),
                f"{side}_cluster_label": make_cluster_label(top_terms, max_terms=5),
                f"{side}_cluster_top_terms": " | ".join(top_terms),
                f"{side}_cluster_n_nodes": int(len(sub)),
            }
        )

    out = pd.DataFrame(rows)

    return out


def build_cluster_edge_annotation() -> pd.DataFrame:
    print("\nLoading historical cluster edges for analogue annotations...")

    df = pd.read_csv(CLUSTER_EDGES_FILE, encoding="utf-8-sig", low_memory=False)

    print(f"Cluster edge rows: {len(df):,}")
    print("Cluster edge columns:", list(df.columns))

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

    weight_col = find_col(
        df,
        ["cluster_weight", "weight", "edge_weight"],
        required=False,
    )

    n_exact_col = find_col(
        df,
        ["n_exact_edges", "total_exact_edges", "exact_edges"],
        required=False,
    )

    first_year_col = find_col(
        df,
        ["first_year", "year"],
        required=False,
    )

    final_strength_col = find_col(
        df,
        ["final_edge_strength"],
        required=False,
    )

    within_year_col = find_col(
        df,
        ["cluster_within_year_percentile", "within_year_percentile"],
        required=False,
    )

    out = pd.DataFrame()
    out["problem_cluster_id"] = df[pc_col].astype(int)
    out["mapped_solution_cluster_id"] = df[sc_col].astype(int)

    if weight_col:
        out["historical_cluster_weight"] = pd.to_numeric(df[weight_col], errors="coerce")
    else:
        out["historical_cluster_weight"] = np.nan

    if n_exact_col:
        out["historical_n_exact_edges"] = pd.to_numeric(df[n_exact_col], errors="coerce")
    else:
        out["historical_n_exact_edges"] = np.nan

    if first_year_col:
        out["historical_first_year"] = pd.to_numeric(df[first_year_col], errors="coerce")
    else:
        out["historical_first_year"] = np.nan

    if final_strength_col:
        out["historical_final_edge_strength"] = pd.to_numeric(df[final_strength_col], errors="coerce")
    else:
        out["historical_final_edge_strength"] = np.nan

    if within_year_col:
        out["historical_within_year_percentile"] = pd.to_numeric(df[within_year_col], errors="coerce")
    else:
        out["historical_within_year_percentile"] = np.nan

    # Deduplicate pair if needed
    agg = (
        out
        .groupby(["problem_cluster_id", "mapped_solution_cluster_id"], as_index=False)
        .agg(
            historical_cluster_weight=("historical_cluster_weight", "max"),
            historical_n_exact_edges=("historical_n_exact_edges", "max"),
            historical_first_year=("historical_first_year", "min"),
            historical_final_edge_strength=("historical_final_edge_strength", "max"),
            historical_within_year_percentile=("historical_within_year_percentile", "max"),
        )
    )

    return agg


def make_balanced_top(df: pd.DataFrame, per_problem: int, global_fill: int) -> pd.DataFrame:
    """
    First keep top N per problem cluster, then globally fill/cap to target size.
    This keeps broad surgical coverage.
    """
    df = df.sort_values(
        [SCORE_COL, "concept_id", "problem_cluster_id", "mapped_solution_cluster_id"],
        ascending=[False, True, True, True],
    ).copy()

    balanced = (
        df
        .sort_values(
            ["problem_cluster_id", SCORE_COL, "concept_id"],
            ascending=[True, False, True],
        )
        .groupby("problem_cluster_id", as_index=False, group_keys=False)
        .head(per_problem)
        .copy()
    )

    if len(balanced) >= global_fill:
        out = balanced.sort_values(SCORE_COL, ascending=False).head(global_fill).copy()
    else:
        already = set(
            zip(
                balanced["concept_id"],
                balanced["problem_cluster_id"],
                balanced["mapped_solution_cluster_id"],
            )
        )

        remaining = df[
            ~df.apply(
                lambda r: (
                    r["concept_id"],
                    r["problem_cluster_id"],
                    r["mapped_solution_cluster_id"],
                )
                in already,
                axis=1,
            )
        ].copy()

        fill_needed = global_fill - len(balanced)
        fill = remaining.sort_values(SCORE_COL, ascending=False).head(fill_needed).copy()

        out = pd.concat([balanced, fill], ignore_index=True)
        out = out.sort_values(SCORE_COL, ascending=False).head(global_fill).copy()

    out = out.reset_index(drop=True)
    out["review_rank"] = np.arange(1, len(out) + 1)

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    print("============================================================")
    print("Annotate diversified OpenAlex hypothesis candidates")
    print("============================================================")

    for f in [
        FINAL_CANDIDATES_FILE,
        PROBLEM_NODE_CLUSTERS_FILE,
        SOLUTION_NODE_CLUSTERS_FILE,
        CLUSTER_EDGES_FILE,
    ]:
        if not f.exists():
            raise FileNotFoundError(f)

    print("Loading final candidate pool...")
    candidates = pd.read_csv(FINAL_CANDIDATES_FILE, encoding="utf-8-sig", low_memory=False)

    print(f"Candidate rows: {len(candidates):,}")
    print("Candidate columns:", list(candidates.columns))

    required = {
        "concept_id",
        "concept_text",
        "problem_cluster_id",
        "mapped_solution_cluster_id",
        SCORE_COL,
    }

    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate file missing required columns: {missing}")

    candidates["problem_cluster_id"] = candidates["problem_cluster_id"].astype(int)
    candidates["mapped_solution_cluster_id"] = candidates["mapped_solution_cluster_id"].astype(int)
    candidates[SCORE_COL] = pd.to_numeric(candidates[SCORE_COL], errors="coerce").fillna(0)

    problem_terms = build_cluster_term_table(
        PROBLEM_NODE_CLUSTERS_FILE,
        side="problem",
        n_terms=N_TERMS_PER_CLUSTER,
    )

    solution_terms = build_cluster_term_table(
        SOLUTION_NODE_CLUSTERS_FILE,
        side="solution",
        n_terms=N_TERMS_PER_CLUSTER,
    ).rename(
        columns={
            "solution_cluster_id": "mapped_solution_cluster_id",
        }
    )

    edge_ann = build_cluster_edge_annotation()

    print("\nMerging annotations...")

    annotated = candidates.merge(
        problem_terms,
        on="problem_cluster_id",
        how="left",
    )

    annotated = annotated.merge(
        solution_terms,
        on="mapped_solution_cluster_id",
        how="left",
    )

    annotated = annotated.merge(
        edge_ann,
        on=["problem_cluster_id", "mapped_solution_cluster_id"],
        how="left",
    )

    # Clean final display columns
    annotated["hypothesis_input_problem"] = annotated["problem_cluster_label"].fillna(
        annotated["problem_cluster_id"].apply(lambda x: f"problem_cluster_{x}")
    )

    annotated["hypothesis_input_openalex_mechanism"] = annotated["concept_text"].apply(clean_text)

    annotated["hypothesis_input_historical_solution_analogue"] = annotated["solution_cluster_label"].fillna(
        annotated["mapped_solution_cluster_id"].apply(lambda x: f"solution_cluster_{x}")
    )

    annotated["hypothesis_input_context"] = (
        "Surgical problem cluster: "
        + annotated["hypothesis_input_problem"].astype(str)
        + " || External OpenAlex mechanism: "
        + annotated["hypothesis_input_openalex_mechanism"].astype(str)
        + " || Historical surgical solution analogue: "
        + annotated["hypothesis_input_historical_solution_analogue"].astype(str)
    )

    annotated = annotated.sort_values(
        [SCORE_COL, "concept_id", "problem_cluster_id", "mapped_solution_cluster_id"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    annotated["annotated_rank"] = np.arange(1, len(annotated) + 1)

    print("Saving annotated full candidate pool...")
    annotated.to_csv(OUT_ANNOTATED, index=False, encoding="utf-8-sig")

    print("Creating balanced Top 1,000 and Top 10,000 files...")

    top1000 = make_balanced_top(
        annotated,
        per_problem=TOP1000_PER_PROBLEM,
        global_fill=TOP1000_GLOBAL_FILL,
    )

    top10000 = make_balanced_top(
        annotated,
        per_problem=TOP10000_PER_PROBLEM,
        global_fill=TOP10000_GLOBAL_FILL,
    )

    top1000.to_csv(OUT_TOP1000, index=False, encoding="utf-8-sig")
    top10000.to_csv(OUT_TOP10000, index=False, encoding="utf-8-sig")

    summary = {
        "input_candidates": str(FINAL_CANDIDATES_FILE),
        "n_input_candidates": int(len(candidates)),
        "n_annotated_candidates": int(len(annotated)),
        "unique_openalex_concepts": int(annotated["concept_id"].nunique()),
        "unique_problem_clusters": int(annotated["problem_cluster_id"].nunique()),
        "unique_solution_clusters": int(annotated["mapped_solution_cluster_id"].nunique()),
        "score_definition": "ML probability only, inherited from selection funnel.",
        "top1000": {
            "rows": int(len(top1000)),
            "unique_openalex_concepts": int(top1000["concept_id"].nunique()),
            "unique_problem_clusters": int(top1000["problem_cluster_id"].nunique()),
            "unique_solution_clusters": int(top1000["mapped_solution_cluster_id"].nunique()),
        },
        "top10000": {
            "rows": int(len(top10000)),
            "unique_openalex_concepts": int(top10000["concept_id"].nunique()),
            "unique_problem_clusters": int(top10000["problem_cluster_id"].nunique()),
            "unique_solution_clusters": int(top10000["mapped_solution_cluster_id"].nunique()),
        },
        "outputs": {
            "annotated": str(OUT_ANNOTATED),
            "top1000_for_manual_review": str(OUT_TOP1000),
            "top10000_for_llm": str(OUT_TOP10000),
            "summary": str(OUT_SUMMARY),
        },
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("============================================================")
    print("DONE")
    print("Annotated full pool:")
    print(OUT_ANNOTATED)
    print("Top 1000:")
    print(OUT_TOP1000)
    print("Top 10000:")
    print(OUT_TOP10000)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("============================================================")


if __name__ == "__main__":
    main()