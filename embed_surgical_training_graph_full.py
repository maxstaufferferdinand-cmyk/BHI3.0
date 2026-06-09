from pathlib import Path
import os
import re
import json
import time
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

INPUT_CSV = BASE_DIR / "llm_extraction_full_results.csv"

OUT_DIR = BASE_DIR / "surgical_training_graph_outputs"
OUT_DIR.mkdir(exist_ok=True)

# Graph-ready tables
OUT_EDGE_INSTANCES = OUT_DIR / "surgical_graph_edge_instances.csv"
OUT_EDGES_AGG = OUT_DIR / "surgical_graph_edges_aggregated.csv"
OUT_PROBLEM_NODES = OUT_DIR / "surgical_problem_nodes.csv"
OUT_SOLUTION_NODES = OUT_DIR / "surgical_solution_nodes.csv"
OUT_PAIR_NODES = OUT_DIR / "surgical_pair_nodes.csv"

# Embedding index tables
OUT_PROBLEM_INDEX = OUT_DIR / "surgical_problem_embedding_index.csv"
OUT_SOLUTION_INDEX = OUT_DIR / "surgical_solution_embedding_index.csv"
OUT_PAIR_INDEX = OUT_DIR / "surgical_pair_embedding_index.csv"

# Embedding arrays
OUT_PROBLEM_EMB = OUT_DIR / "surgical_problem_embeddings.npy"
OUT_SOLUTION_EMB = OUT_DIR / "surgical_solution_embeddings.npy"
OUT_PAIR_EMB = OUT_DIR / "surgical_pair_embeddings.npy"

# Cache files for resumable embedding calls
CACHE_PROBLEM = OUT_DIR / "embedding_cache_problem.jsonl"
CACHE_SOLUTION = OUT_DIR / "embedding_cache_solution.jsonl"
CACHE_PAIR = OUT_DIR / "embedding_cache_pair.jsonl"

OUT_SUMMARY = OUT_DIR / "surgical_training_graph_embedding_summary.txt"

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Embeddings are cheap and concept texts are short.
# If API errors occur, reduce to 128.
EMBED_BATCH_SIZE = 256

MAX_RETRIES = 6
SLEEP_BETWEEN_BATCHES = 0.1

# Set to an integer for debugging, e.g. 10000.
# Set to None for full run.
MAX_INCLUDED_ROWS = None

# Column candidates
POSSIBLE_DECISION_COLS = ["decision", "llm_decision", "include_decision"]
POSSIBLE_PROBLEM_COLS = ["surgical_problem", "manual_surgical_problem", "problem"]
POSSIBLE_SOLUTION_COLS = [
    "technical_solution",
    "manual_technical_solution",
    "manual_techniqual_solution",
    "solution",
]
POSSIBLE_PMID_COLS = ["pmid", "PMID"]
POSSIBLE_YEAR_COLS = ["year", "publication_year"]
POSSIBLE_TITLE_COLS = ["title"]
POSSIBLE_REASON_COLS = ["reason", "llm_reason"]

# ============================================================
# CLIENT
# ============================================================

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        'In PowerShell run: $env:OPENAI_API_KEY="sk-..."'
    )

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ============================================================
# HELPERS
# ============================================================

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_concept(x: str) -> str:
    x = safe_str(x)
    x = x.replace("\n", " ").replace("\r", " ")
    x = re.sub(r"\s+", " ", x)
    x = x.strip(" ;,.-")
    return x


def norm_key(x: str) -> str:
    """
    Stable normalization key for deduplication.
    Keeps the concept meaning, removes superficial punctuation/case differences.
    """
    x = clean_concept(x).lower()
    x = re.sub(r"[\u2010-\u2015]", "-", x)
    x = re.sub(r"[^a-z0-9α-ωΑ-Ω/+\- ]+", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def stable_id(prefix: str, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def normalize_decision(x: str) -> str:
    x = safe_str(x).lower().strip()
    if x in ["include", "included", "yes", "1", "true"]:
        return "include"
    if "include" in x and "reject" not in x:
        return "include"
    return "reject"


def most_common_nonempty(values: list[str]) -> str:
    vals = [clean_concept(v) for v in values if clean_concept(v)]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def make_problem_embedding_text(problem: str) -> str:
    return f"Surgical problem: {problem}."


def make_solution_embedding_text(solution: str) -> str:
    return f"Surgical technology or procedural solution: {solution}."


def make_pair_embedding_text(problem: str, solution: str) -> str:
    return f"Surgical problem: {problem}. Technical solution: {solution}."


def append_jsonl(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_embedding_cache(path: Path) -> dict[str, list[float]]:
    cache = {}
    if not path.exists():
        return cache

    print(f"Loading embedding cache: {path}")
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                item_id = safe_str(obj.get("id"))
                emb = obj.get("embedding")
                if item_id and isinstance(emb, list):
                    cache[item_id] = emb
                    n += 1
            except Exception:
                continue

    print(f"  loaded {n:,} cached embeddings")
    return cache


def embed_items_resumable(
    index_df: pd.DataFrame,
    id_col: str,
    text_col: str,
    cache_path: Path,
    label: str,
) -> np.ndarray:
    """
    Embeds all rows in index_df, using a JSONL cache so the script can resume.
    Returns embeddings ordered exactly like index_df.
    """
    cache = load_embedding_cache(cache_path)

    ids = index_df[id_col].astype(str).tolist()
    texts = index_df[text_col].astype(str).tolist()

    missing = [(item_id, text) for item_id, text in zip(ids, texts) if item_id not in cache]

    print(f"{label}: total={len(ids):,}, cached={len(cache):,}, missing={len(missing):,}")

    for start in range(0, len(missing), EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, len(missing))
        batch = missing[start:end]
        batch_ids = [x[0] for x in batch]
        batch_texts = [x[1] for x in batch]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch_texts,
                )

                rows_to_cache = []
                for item_id, item in zip(batch_ids, response.data):
                    emb = item.embedding
                    cache[item_id] = emb
                    rows_to_cache.append({
                        "id": item_id,
                        "embedding": emb,
                    })

                append_jsonl(cache_path, rows_to_cache)

                print(
                    f"  {label}: embedded {end:,}/{len(missing):,} missing "
                    f"({len(cache):,}/{len(ids):,} total cached)"
                )
                time.sleep(SLEEP_BETWEEN_BATCHES)
                break

            except Exception as e:
                print(f"  {label}: attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(2 * attempt)

    # Assemble ordered matrix
    ordered = []
    missing_after = []
    for item_id in ids:
        if item_id not in cache:
            missing_after.append(item_id)
        else:
            ordered.append(cache[item_id])

    if missing_after:
        raise RuntimeError(f"{label}: missing embeddings after run: {len(missing_after)}")

    return np.array(ordered, dtype=np.float32)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom[denom == 0] = 1
    return mat / denom


# ============================================================
# MAIN
# ============================================================

def main():
    print("============================================================")
    print("Surgical training graph embedding FULL RUN")
    print("============================================================")
    print("Input:", INPUT_CSV)
    print("Output dir:", OUT_DIR)
    print("Embedding model:", EMBEDDING_MODEL)
    print("Embedding batch size:", EMBED_BATCH_SIZE)
    print("Max included rows:", MAX_INCLUDED_ROWS)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    print("Input rows:", len(df))
    print("Input columns:", list(df.columns))

    decision_col = find_column(df, POSSIBLE_DECISION_COLS, required=True)
    problem_col = find_column(df, POSSIBLE_PROBLEM_COLS, required=True)
    solution_col = find_column(df, POSSIBLE_SOLUTION_COLS, required=True)

    pmid_col = find_column(df, POSSIBLE_PMID_COLS, required=False)
    year_col = find_column(df, POSSIBLE_YEAR_COLS, required=False)
    title_col = find_column(df, POSSIBLE_TITLE_COLS, required=False)
    reason_col = find_column(df, POSSIBLE_REASON_COLS, required=False)

    print("Detected columns:")
    print("  decision:", decision_col)
    print("  problem:", problem_col)
    print("  solution:", solution_col)
    print("  pmid:", pmid_col)
    print("  year:", year_col)
    print("  title:", title_col)
    print("  reason:", reason_col)

    df["_decision_norm"] = df[decision_col].apply(normalize_decision)
    df["_problem_raw"] = df[problem_col].apply(clean_concept)
    df["_solution_raw"] = df[solution_col].apply(clean_concept)
    df["_problem_key"] = df["_problem_raw"].apply(norm_key)
    df["_solution_key"] = df["_solution_raw"].apply(norm_key)

    inc = df[
        (df["_decision_norm"] == "include") &
        (df["_problem_key"].str.len() > 2) &
        (df["_solution_key"].str.len() > 2)
    ].copy()

    if MAX_INCLUDED_ROWS is not None:
        inc = inc.head(MAX_INCLUDED_ROWS).copy()

    inc = inc.reset_index(drop=True)
    print("Usable included rows:", len(inc))

    if len(inc) == 0:
        raise RuntimeError("No usable included rows found.")

    # Metadata columns with safe fallback
    inc["_pmid"] = inc[pmid_col].apply(safe_str) if pmid_col else ""
    inc["_year"] = inc[year_col].apply(safe_str) if year_col else ""
    inc["_title"] = inc[title_col].apply(safe_str) if title_col else ""
    inc["_reason"] = inc[reason_col].apply(safe_str) if reason_col else ""

    inc["_problem_id"] = inc["_problem_key"].apply(lambda x: stable_id("P", x))
    inc["_solution_id"] = inc["_solution_key"].apply(lambda x: stable_id("S", x))
    inc["_pair_key"] = inc["_problem_key"] + " || " + inc["_solution_key"]
    inc["_pair_id"] = inc["_pair_key"].apply(lambda x: stable_id("E", x))

    # --------------------------------------------------------
    # Edge instances: one row per included abstract
    # --------------------------------------------------------

    edge_instances = pd.DataFrame({
        "pmid": inc["_pmid"],
        "year": inc["_year"],
        "title": inc["_title"],
        "problem_id": inc["_problem_id"],
        "solution_id": inc["_solution_id"],
        "pair_id": inc["_pair_id"],
        "surgical_problem": inc["_problem_raw"],
        "technical_solution": inc["_solution_raw"],
        "problem_key": inc["_problem_key"],
        "solution_key": inc["_solution_key"],
        "pair_key": inc["_pair_key"],
    })

    edge_instances.to_csv(OUT_EDGE_INSTANCES, index=False, encoding="utf-8-sig")
    print("Saved edge instances:", OUT_EDGE_INSTANCES, len(edge_instances))

    # --------------------------------------------------------
    # Problem nodes
    # --------------------------------------------------------

    problem_nodes = (
        edge_instances
        .groupby(["problem_id", "problem_key"], as_index=False)
        .agg(
            label=("surgical_problem", lambda x: most_common_nonempty(list(x))),
            n_occurrences=("pmid", "count"),
            n_unique_solutions=("solution_id", "nunique"),
            first_year=("year", lambda x: min([v for v in x if safe_str(v)] or [""])),
            last_year=("year", lambda x: max([v for v in x if safe_str(v)] or [""])),
        )
    )

    problem_nodes["node_type"] = "surgical_problem"
    problem_nodes["embedding_text"] = problem_nodes["label"].apply(make_problem_embedding_text)

    problem_nodes = problem_nodes[
        [
            "problem_id", "node_type", "label", "problem_key",
            "n_occurrences", "n_unique_solutions", "first_year", "last_year",
            "embedding_text",
        ]
    ]

    problem_nodes.to_csv(OUT_PROBLEM_NODES, index=False, encoding="utf-8-sig")
    print("Saved problem nodes:", OUT_PROBLEM_NODES, len(problem_nodes))

    # --------------------------------------------------------
    # Solution nodes
    # --------------------------------------------------------

    solution_nodes = (
        edge_instances
        .groupby(["solution_id", "solution_key"], as_index=False)
        .agg(
            label=("technical_solution", lambda x: most_common_nonempty(list(x))),
            n_occurrences=("pmid", "count"),
            n_unique_problems=("problem_id", "nunique"),
            first_year=("year", lambda x: min([v for v in x if safe_str(v)] or [""])),
            last_year=("year", lambda x: max([v for v in x if safe_str(v)] or [""])),
        )
    )

    solution_nodes["node_type"] = "technical_solution"
    solution_nodes["embedding_text"] = solution_nodes["label"].apply(make_solution_embedding_text)

    solution_nodes = solution_nodes[
        [
            "solution_id", "node_type", "label", "solution_key",
            "n_occurrences", "n_unique_problems", "first_year", "last_year",
            "embedding_text",
        ]
    ]

    solution_nodes.to_csv(OUT_SOLUTION_NODES, index=False, encoding="utf-8-sig")
    print("Saved solution nodes:", OUT_SOLUTION_NODES, len(solution_nodes))

    # --------------------------------------------------------
    # Aggregated edges and pair nodes
    # --------------------------------------------------------

    edges_agg = (
        edge_instances
        .groupby(["pair_id", "problem_id", "solution_id", "problem_key", "solution_key", "pair_key"], as_index=False)
        .agg(
            surgical_problem=("surgical_problem", lambda x: most_common_nonempty(list(x))),
            technical_solution=("technical_solution", lambda x: most_common_nonempty(list(x))),
            weight=("pmid", "count"),
            first_year=("year", lambda x: min([v for v in x if safe_str(v)] or [""])),
            last_year=("year", lambda x: max([v for v in x if safe_str(v)] or [""])),
            pmids=("pmid", lambda x: ";".join([safe_str(v) for v in x if safe_str(v)][:50])),
        )
    )

    edges_agg["edge_type"] = "problem_addressed_by_solution"
    edges_agg["pair_embedding_text"] = [
        make_pair_embedding_text(p, s)
        for p, s in zip(edges_agg["surgical_problem"], edges_agg["technical_solution"])
    ]

    edges_agg = edges_agg[
        [
            "pair_id", "edge_type", "problem_id", "solution_id",
            "surgical_problem", "technical_solution",
            "problem_key", "solution_key", "pair_key",
            "weight", "first_year", "last_year", "pmids",
            "pair_embedding_text",
        ]
    ]

    edges_agg.to_csv(OUT_EDGES_AGG, index=False, encoding="utf-8-sig")
    print("Saved aggregated edges:", OUT_EDGES_AGG, len(edges_agg))

    pair_nodes = edges_agg.rename(columns={
        "pair_id": "pair_id",
        "pair_key": "pair_key",
        "pair_embedding_text": "embedding_text",
    }).copy()

    pair_nodes["node_type"] = "problem_solution_pair"
    pair_nodes["label"] = pair_nodes["surgical_problem"] + " -> " + pair_nodes["technical_solution"]

    pair_nodes = pair_nodes[
        [
            "pair_id", "node_type", "label", "pair_key",
            "problem_id", "solution_id",
            "surgical_problem", "technical_solution",
            "weight", "first_year", "last_year",
            "embedding_text",
        ]
    ]

    pair_nodes.to_csv(OUT_PAIR_NODES, index=False, encoding="utf-8-sig")
    print("Saved pair nodes:", OUT_PAIR_NODES, len(pair_nodes))

    # --------------------------------------------------------
    # Embedding index tables
    # --------------------------------------------------------

    problem_index = problem_nodes[["problem_id", "label", "problem_key", "embedding_text"]].copy()
    solution_index = solution_nodes[["solution_id", "label", "solution_key", "embedding_text"]].copy()
    pair_index = pair_nodes[["pair_id", "label", "pair_key", "problem_id", "solution_id", "embedding_text"]].copy()

    problem_index.to_csv(OUT_PROBLEM_INDEX, index=False, encoding="utf-8-sig")
    solution_index.to_csv(OUT_SOLUTION_INDEX, index=False, encoding="utf-8-sig")
    pair_index.to_csv(OUT_PAIR_INDEX, index=False, encoding="utf-8-sig")

    print("Saved embedding index tables:")
    print(" ", OUT_PROBLEM_INDEX, len(problem_index))
    print(" ", OUT_SOLUTION_INDEX, len(solution_index))
    print(" ", OUT_PAIR_INDEX, len(pair_index))

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    problem_emb = embed_items_resumable(
        index_df=problem_index,
        id_col="problem_id",
        text_col="embedding_text",
        cache_path=CACHE_PROBLEM,
        label="problem",
    )

    solution_emb = embed_items_resumable(
        index_df=solution_index,
        id_col="solution_id",
        text_col="embedding_text",
        cache_path=CACHE_SOLUTION,
        label="solution",
    )

    pair_emb = embed_items_resumable(
        index_df=pair_index,
        id_col="pair_id",
        text_col="embedding_text",
        cache_path=CACHE_PAIR,
        label="pair",
    )

    np.save(OUT_PROBLEM_EMB, problem_emb)
    np.save(OUT_SOLUTION_EMB, solution_emb)
    np.save(OUT_PAIR_EMB, pair_emb)

    print("Saved embedding arrays:")
    print(" ", OUT_PROBLEM_EMB, problem_emb.shape)
    print(" ", OUT_SOLUTION_EMB, solution_emb.shape)
    print(" ", OUT_PAIR_EMB, pair_emb.shape)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = []
    summary.append("Surgical training graph embedding full run")
    summary.append("=" * 70)
    summary.append(f"Input: {INPUT_CSV}")
    summary.append(f"Input rows: {len(df):,}")
    summary.append(f"Usable included rows / edge instances: {len(edge_instances):,}")
    summary.append("")
    summary.append(f"Unique problem nodes: {len(problem_nodes):,}")
    summary.append(f"Unique solution nodes: {len(solution_nodes):,}")
    summary.append(f"Unique problem-solution pair edges: {len(edges_agg):,}")
    summary.append("")
    summary.append(f"Embedding model: {EMBEDDING_MODEL}")
    summary.append(f"Problem embedding shape: {problem_emb.shape}")
    summary.append(f"Solution embedding shape: {solution_emb.shape}")
    summary.append(f"Pair embedding shape: {pair_emb.shape}")
    summary.append("")
    summary.append("Output files:")
    for p in [
        OUT_EDGE_INSTANCES,
        OUT_EDGES_AGG,
        OUT_PROBLEM_NODES,
        OUT_SOLUTION_NODES,
        OUT_PAIR_NODES,
        OUT_PROBLEM_INDEX,
        OUT_SOLUTION_INDEX,
        OUT_PAIR_INDEX,
        OUT_PROBLEM_EMB,
        OUT_SOLUTION_EMB,
        OUT_PAIR_EMB,
        CACHE_PROBLEM,
        CACHE_SOLUTION,
        CACHE_PAIR,
    ]:
        summary.append(str(p))

    OUT_SUMMARY.write_text("\n".join(summary), encoding="utf-8")
    print("Saved summary:", OUT_SUMMARY)

    print("============================================================")
    print("DONE")
    print("Next: inspect summary and node/edge counts.")
    print("============================================================")


if __name__ == "__main__":
    main()