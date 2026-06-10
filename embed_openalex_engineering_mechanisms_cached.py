from pathlib import Path
from datetime import datetime
import os
import re
import json
import time

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

INPUT_FILE = BASE_DIR / "openalex_engineering_mechanisms_full_fast.csv"

OUT_DIR = BASE_DIR / "openalex_engineering_mechanism_embeddings_cached"
OUT_DIR.mkdir(exist_ok=True)

OUT_INDEX = OUT_DIR / "openalex_engineering_mechanism_index.csv"
OUT_EMBEDDINGS = OUT_DIR / "openalex_engineering_mechanism_embeddings.npy"
OUT_WORK_LINKS = OUT_DIR / "openalex_engineering_mechanism_work_links.csv"
OUT_REMOVED = OUT_DIR / "openalex_engineering_mechanism_removed_terms.csv"
OUT_PROGRESS = OUT_DIR / "openalex_engineering_mechanism_progress.csv"
OUT_SUMMARY = OUT_DIR / "openalex_engineering_mechanism_embedding_summary.json"

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536

# Größer = weniger API-Abfragen. 100 ist meist stabiler als 25/10,
# aber nicht so groß, dass einzelne Calls riesig werden.
BATCH_SIZE = 100

# Lange Latenz tolerieren.
API_TIMEOUT_SECONDS = 300

# Wenn API/Netz kurz zickt.
MAX_ATTEMPTS = 8

# Pause zwischen erfolgreichen Requests.
SLEEP_SECONDS = 1.0

# Für vollen Run None lassen.
MAX_CONCEPTS = None

BAD_TERMS = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "unknown",
    "unclear",
    "unclear mechanism",
    "not specified",
    "not applicable",
    "no clear mechanism",
    "no mechanism",
    "missing",
    "missing output",
    "missing mechanism",
    "no output",
    "failed output",
    "error output",
    "invalid output",
    "empty output",
    "model output missing",
    "output missing",
}


# ============================================================
# HELPERS
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_concept(x) -> str:
    x = safe_str(x)
    x = x.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    x = re.sub(r"\s+", " ", x).strip()
    x = x.strip(" ;,.-:")
    x = x.lower()
    return x


def is_valid_concept(x: str) -> bool:
    if not x:
        return False

    if x in BAD_TERMS:
        return False

    if len(x) < 3:
        return False

    bad_prefixes = [
        "unclear mechanism",
        "missing output",
        "missing mechanism",
        "no output",
        "failed output",
        "error output",
        "invalid output",
        "empty output",
        "model output missing",
        "output missing",
    ]

    for prefix in bad_prefixes:
        if x.startswith(prefix):
            return False

    # Fälle wie "missing output missing output"
    tmp = x
    for bad in ["missing output", "unclear mechanism", "missing mechanism"]:
        tmp = tmp.replace(bad, "").strip(" ;,.-:")

    if tmp == "":
        return False

    return True


def read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    if suffix == ".json":
        return pd.read_json(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def append_progress(row: dict):
    row_df = pd.DataFrame([row])
    header = not OUT_PROGRESS.exists()
    row_df.to_csv(
        OUT_PROGRESS,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8-sig",
    )


def load_completed_batches() -> set[tuple[int, int]]:
    if not OUT_PROGRESS.exists():
        return set()

    try:
        prog = pd.read_csv(OUT_PROGRESS)
    except Exception:
        return set()

    if prog.empty:
        return set()

    ok = prog[prog["status"] == "success"].copy()

    completed = set()
    for _, row in ok.iterrows():
        completed.add((int(row["start"]), int(row["end"])))

    return completed


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("============================================================")
    print("OpenAlex engineering mechanism cached embedding run")
    print("============================================================")
    print("Input:", INPUT_FILE)
    print("Output dir:", OUT_DIR)
    print("Embedding model:", EMBEDDING_MODEL)
    print("Batch size:", BATCH_SIZE)
    print("Timeout seconds:", API_TIMEOUT_SECONDS)
    print("============================================================")

    df = read_input(INPUT_FILE)

    print(f"Rows in input: {len(df):,}")
    print("Columns:", list(df.columns))

    required = {"openalex_id", "mechanism_1", "mechanism_2"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    valid_rows = []
    removed_rows = []

    print("Cleaning mechanisms...")

    for _, row in df.iterrows():
        openalex_id = row["openalex_id"]

        for source_col in ["mechanism_1", "mechanism_2"]:
            raw = row[source_col]
            concept = clean_concept(raw)

            if is_valid_concept(concept):
                valid_rows.append({
                    "openalex_id": openalex_id,
                    "source_col": source_col,
                    "concept_text": concept,
                })
            else:
                removed_rows.append({
                    "openalex_id": openalex_id,
                    "source_col": source_col,
                    "raw_value": raw,
                    "normalized_value": concept,
                })

    long_df = pd.DataFrame(valid_rows)
    removed_df = pd.DataFrame(removed_rows)

    if long_df.empty:
        raise ValueError("No valid mechanisms found after cleaning.")

    print(f"Valid mechanism mentions: {len(long_df):,}")
    print(f"Removed mechanism mentions: {len(removed_df):,}")

    print("Deduplicating concepts...")

    concept_index = (
        long_df
        .groupby("concept_text", as_index=False)
        .agg(
            n_mentions=("concept_text", "size"),
            n_works=("openalex_id", "nunique"),
        )
        .sort_values(
            ["n_works", "n_mentions", "concept_text"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    if MAX_CONCEPTS is not None:
        concept_index = concept_index.head(MAX_CONCEPTS).copy()
        concept_index = concept_index.reset_index(drop=True)

    concept_index.insert(0, "concept_id", range(len(concept_index)))

    work_links = long_df.merge(
        concept_index[["concept_id", "concept_text"]],
        on="concept_text",
        how="inner",
    )[["concept_id", "concept_text", "openalex_id", "source_col"]]

    print(f"Unique concepts to embed: {len(concept_index):,}")

    concept_index.to_csv(OUT_INDEX, index=False, encoding="utf-8-sig")
    work_links.to_csv(OUT_WORK_LINKS, index=False, encoding="utf-8-sig")

    if not removed_df.empty:
        removed_summary = (
            removed_df
            .groupby("normalized_value", as_index=False)
            .agg(n_removed=("normalized_value", "size"))
            .sort_values("n_removed", ascending=False)
        )
        removed_summary.to_csv(OUT_REMOVED, index=False, encoding="utf-8-sig")

    print("Saved index:", OUT_INDEX)
    print("Saved work links:", OUT_WORK_LINKS)
    print("Saved removed terms:", OUT_REMOVED)

    return df, long_df, removed_df, concept_index


def prepare_embedding_memmap(n_rows: int):
    if OUT_EMBEDDINGS.exists():
        try:
            arr = np.load(OUT_EMBEDDINGS, mmap_mode="r+")
            if arr.shape == (n_rows, EMBEDDING_DIM):
                print("Existing embedding matrix found:")
                print(" ", OUT_EMBEDDINGS)
                print(" ", arr.shape)
                return arr
            else:
                print("Existing embedding matrix has wrong shape:")
                print(" ", arr.shape)
                print("Expected:", (n_rows, EMBEDDING_DIM))
                print("Deleting and recreating.")
                OUT_EMBEDDINGS.unlink()
        except Exception as e:
            print("Could not read existing embedding matrix:", e)
            print("Deleting and recreating.")
            OUT_EMBEDDINGS.unlink()

    print("Creating embedding matrix immediately:")
    print(" ", OUT_EMBEDDINGS)
    print("Shape:", (n_rows, EMBEDDING_DIM))

    arr = open_memmap(
        OUT_EMBEDDINGS,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, EMBEDDING_DIM),
    )

    arr.flush()
    return arr


def embed_batch(client: OpenAI, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    emb = [item.embedding for item in response.data]
    arr = np.array(emb, dtype=np.float32)

    if arr.shape != (len(texts), EMBEDDING_DIM):
        raise ValueError(
            f"Unexpected shape {arr.shape}; expected {(len(texts), EMBEDDING_DIM)}"
        )

    return arr


def run_embeddings(concept_index: pd.DataFrame):
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=API_TIMEOUT_SECONDS,
        max_retries=0,
    )

    texts = concept_index["concept_text"].tolist()
    n = len(texts)

    emb_memmap = prepare_embedding_memmap(n)

    completed = load_completed_batches()

    print("Completed cached batches found:", len(completed))
    print("Starting/resuming embedding...")

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)

        if (start, end) in completed:
            print(f"cached {end:,}/{n:,}  rows {start:,}-{end:,}")
            continue

        batch = texts[start:end]

        print(f"embedding {end:,}/{n:,}  rows {start:,}-{end:,}")

        success = False
        last_error = ""

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                batch_emb = embed_batch(client, batch)

                emb_memmap[start:end, :] = batch_emb
                emb_memmap.flush()

                append_progress({
                    "timestamp": now_str(),
                    "start": start,
                    "end": end,
                    "n_rows": end - start,
                    "status": "success",
                    "attempt": attempt,
                    "error": "",
                })

                print(f"saved/cached {end:,}/{n:,}")
                success = True
                break

            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                print("All previous successful batches are already saved.")
                raise

            except Exception as e:
                last_error = str(e)
                wait = min(600, 10 * (2 ** (attempt - 1)))

                print(f"attempt {attempt}/{MAX_ATTEMPTS} failed for rows {start:,}-{end:,}: {e}")
                print(f"waiting {wait}s...")

                append_progress({
                    "timestamp": now_str(),
                    "start": start,
                    "end": end,
                    "n_rows": end - start,
                    "status": "failed_attempt",
                    "attempt": attempt,
                    "error": last_error,
                })

                time.sleep(wait)

        if not success:
            append_progress({
                "timestamp": now_str(),
                "start": start,
                "end": end,
                "n_rows": end - start,
                "status": "failed_batch",
                "attempt": MAX_ATTEMPTS,
                "error": last_error,
            })

            raise RuntimeError(
                f"Batch rows {start}-{end} failed after {MAX_ATTEMPTS} attempts. "
                "Previous successful batches are saved. Restart the same script to resume."
            )

        time.sleep(SLEEP_SECONDS)

    emb_memmap.flush()
    print("All embeddings complete.")


def write_summary(df, long_df, removed_df, concept_index):
    summary = {
        "timestamp": now_str(),
        "input_file": str(INPUT_FILE),
        "output_dir": str(OUT_DIR),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
        "api_timeout_seconds": API_TIMEOUT_SECONDS,
        "max_attempts": MAX_ATTEMPTS,
        "sleep_seconds": SLEEP_SECONDS,
        "max_concepts": MAX_CONCEPTS,
        "input_rows": int(len(df)),
        "valid_mechanism_mentions": int(len(long_df)),
        "removed_mechanism_mentions": int(len(removed_df)),
        "unique_concepts_embedded": int(len(concept_index)),
        "outputs": {
            "index": str(OUT_INDEX),
            "embeddings": str(OUT_EMBEDDINGS),
            "work_links": str(OUT_WORK_LINKS),
            "removed_terms": str(OUT_REMOVED),
            "progress": str(OUT_PROGRESS),
        },
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved summary:", OUT_SUMMARY)


def main():
    df, long_df, removed_df, concept_index = build_tables()

    run_embeddings(concept_index)

    write_summary(df, long_df, removed_df, concept_index)

    print("============================================================")
    print("DONE")
    print("Index:")
    print(OUT_INDEX)
    print("Embeddings:")
    print(OUT_EMBEDDINGS)
    print("Progress:")
    print(OUT_PROGRESS)
    print("============================================================")


if __name__ == "__main__":
    main()