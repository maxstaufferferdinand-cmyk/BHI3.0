from pathlib import Path
import os
import json
import time
import traceback
import pandas as pd
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

INPUT_CSV = BASE_DIR / "openalex_engineering_transfer_2020_2026_full.csv"

OUTPUT_CSV = BASE_DIR / "openalex_engineering_mechanisms_full_fast.csv"
OUTPUT_JSONL = BASE_DIR / "openalex_engineering_mechanisms_full_fast.jsonl"
FAILED_JSONL = BASE_DIR / "openalex_engineering_mechanisms_full_fast_failed_batches.jsonl"
LOG_FILE = BASE_DIR / "openalex_engineering_mechanisms_full_fast_log.txt"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Aggressiv, aber schneller. Falls JSON-Fehler: 50 oder 40.
BATCH_SIZE = 30

# CSV chunkweise lesen
CHUNKSIZE = 5000

# Abstract kürzen, sonst wird es zu langsam/teuer
MAX_TITLE_CHARS = 500
MAX_ABSTRACT_CHARS = 1500

MAX_RETRIES = 5
SLEEP_BETWEEN_BATCHES = 0.2

# Für echten Full Run: None.
# Für kurzen Check z. B. 1000 setzen.
MAX_RECORDS = None

RESUME = True

# ============================================================
# OPENAI CLIENT
# ============================================================

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        'Set it in PowerShell with: $env:OPENAI_API_KEY="sk-..."'
    )

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ============================================================
# SHORT PROMPT
# ============================================================

SYSTEM_PROMPT = """
You extract central engineering mechanisms from scientific abstracts.

Task:
For each OpenAlex engineering abstract, extract exactly TWO central technical, physical, material, mechanical, fluidic, robotic, sensor, or computational mechanisms.

Output JSON only.
Return a JSON array with one object per input abstract.

Output schema:
[
  {
    "openalex_id": "...",
    "mechanism_1": "...",
    "mechanism_2": "...",
    "confidence": 0.0
  }
]

Rules:
- Each mechanism must be 1-5 words.
- Mechanisms must be concise concepts, not full sentences.
- Extract mechanisms, not generic topics.
- Prefer transferable engineering principles.
- Try to extract the most sophisticated or specific or experimental concepts or materials of the abstract.
- Try to avoid generic or broadly used terms like magnetic field, artificial intelligence, mechanical engineering etc.
- Avoid broad generic phrases
- If only one strong mechanism is present, infer the second best-supported mechanism from the abstract.
- If the abstract is too vague or not engineering-relevant, return:
  mechanism_1 = "unclear mechanism"
  mechanism_2 = "unclear mechanism"
  confidence <= 0.3
- Use only information supported by the title and abstract.
- Do not add explanation outside JSON.
""".strip()

# ============================================================
# HELPERS
# ============================================================

def log(msg: str):
    print(msg, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def strip_json_fences(raw: str) -> str:
    raw = raw.strip()

    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()
    elif raw.startswith("```JSON"):
        raw = raw[len("```JSON"):].strip()
    elif raw.startswith("```"):
        raw = raw[len("```"):].strip()

    if raw.endswith("```"):
        raw = raw[:-3].strip()

    return raw


def normalize_mechanism(text: str) -> str:
    text = safe_str(text)
    text = text.replace("\n", " ").strip()

    # Mechanismen sollen kurz sein. Nicht hart bei 5 abschneiden,
    # aber extrem lange Outputs abfangen.
    words = text.split()
    if len(words) > 8:
        text = " ".join(words[:8])

    return text


def normalize_result(obj):
    return {
        "openalex_id": safe_str(obj.get("openalex_id")),
        "mechanism_1": normalize_mechanism(obj.get("mechanism_1")),
        "mechanism_2": normalize_mechanism(obj.get("mechanism_2")),
    }


def append_jsonl(path: Path, rows: list):
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path: Path, rows: list):
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8-sig")


def load_done_ids() -> set:
    done = set()

    if not RESUME or not OUTPUT_JSONL.exists():
        return done

    log(f"Loading already processed IDs from {OUTPUT_JSONL} ...")

    n = 0
    with OUTPUT_JSONL.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                oid = safe_str(obj.get("openalex_id"))
                if oid:
                    done.add(oid)
            except Exception:
                continue

            if n % 100000 == 0:
                log(f"  loaded {n:,} lines, done IDs={len(done):,}")

    return done


def make_payload(batch_df: pd.DataFrame) -> str:
    records = []

    for _, row in batch_df.iterrows():
        records.append({
            "openalex_id": safe_str(row.get("openalex_id")),
            "title": safe_str(row.get("title"))[:MAX_TITLE_CHARS],
            "abstract": safe_str(row.get("abstract"))[:MAX_ABSTRACT_CHARS],
        })

    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def parse_model_json(raw: str):
    clean = strip_json_fences(raw)
    data = json.loads(clean)

    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        data = data["results"]

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError("Model output is not JSON array.")

    return data


def call_llm(batch_df: pd.DataFrame, batch_number: int):
    user_prompt = (
        "Extract exactly two central transferable engineering mechanisms from each abstract. "
        "Return JSON array only.\n\n"
        f"{make_payload(batch_df)}"
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw = response.choices[0].message.content
            data = parse_model_json(raw)
            normalized = [normalize_result(x) for x in data]

            input_ids = [safe_str(x) for x in batch_df["openalex_id"].tolist()]
            output_ids = {safe_str(x.get("openalex_id")) for x in normalized}

            # Fehlende Outputs ergänzen, damit Batch nicht kaputtgeht
            for oid in input_ids:
                if oid not in output_ids:
                    normalized.append({
                        "openalex_id": oid,
                        "mechanism_1": "missing output",
                        "mechanism_2": "missing output",
                    })

            # Doppelte Outputs entfernen, Reihenfolge nach Input wiederherstellen
            by_id = {}
            for obj in normalized:
                oid = safe_str(obj.get("openalex_id"))
                if oid and oid not in by_id:
                    by_id[oid] = obj

            ordered = []
            for oid in input_ids:
                if oid in by_id:
                    ordered.append(by_id[oid])
                else:
                    ordered.append({
                        "openalex_id": oid,
                        "mechanism_1": "missing output",
                        "mechanism_2": "missing output",
                    })

            return ordered, raw

        except Exception as e:
            last_error = str(e)
            log(f"Batch {batch_number}, attempt {attempt}/{MAX_RETRIES} failed: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"Batch {batch_number} failed after retries: {last_error}")


def fail_batch(batch_df: pd.DataFrame, batch_number: int, error_msg: str):
    rows = []

    for _, row in batch_df.iterrows():
        rows.append({
            "openalex_id": safe_str(row.get("openalex_id")),
            "mechanism_1": "batch failed",
            "mechanism_2": "batch failed",
        })

    append_jsonl(OUTPUT_JSONL, rows)
    append_csv(OUTPUT_CSV, rows)

    failed = {
        "batch_number": batch_number,
        "error": error_msg,
        "openalex_ids": [safe_str(x) for x in batch_df["openalex_id"].tolist()],
        "traceback": traceback.format_exc(),
    }
    append_jsonl(FAILED_JSONL, [failed])


# ============================================================
# MAIN
# ============================================================

def main():
    log("============================================================")
    log("OpenAlex engineering mechanism extraction FULL FAST")
    log("============================================================")
    log(f"Model: {MODEL}")
    log(f"Input: {INPUT_CSV}")
    log(f"Output CSV: {OUTPUT_CSV}")
    log(f"Output JSONL: {OUTPUT_JSONL}")
    log(f"Batch size: {BATCH_SIZE}")
    log(f"Chunk size: {CHUNKSIZE}")
    log(f"Max records: {MAX_RECORDS}")
    log(f"Resume: {RESUME}")
    log(f"Max abstract chars: {MAX_ABSTRACT_CHARS}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    done_ids = load_done_ids()
    log(f"Already processed IDs: {len(done_ids):,}")

    usecols = ["openalex_id", "title", "abstract"]

    reader = pd.read_csv(
        INPUT_CSV,
        usecols=lambda c: c in usecols,
        encoding="utf-8-sig",
        low_memory=False,
        chunksize=CHUNKSIZE
    )

    total_seen = 0
    total_skipped = 0
    total_sent = 0
    total_written = 0
    batch_number = 0
    buffer = []

    for chunk_idx, chunk in enumerate(reader, start=1):
        chunk = chunk[
            chunk["openalex_id"].notna() &
            chunk["abstract"].notna() &
            (chunk["abstract"].astype(str).str.len() > 50)
        ].copy()

        for _, row in chunk.iterrows():
            total_seen += 1
            oid = safe_str(row.get("openalex_id"))

            if RESUME and oid in done_ids:
                total_skipped += 1
                continue

            buffer.append(row)

            if len(buffer) >= BATCH_SIZE:
                if MAX_RECORDS is not None and total_sent >= MAX_RECORDS:
                    break

                batch_number += 1
                batch_df = pd.DataFrame(buffer)
                buffer = []

                if MAX_RECORDS is not None:
                    remaining = MAX_RECORDS - total_sent
                    batch_df = batch_df.head(remaining)

                if len(batch_df) == 0:
                    break

                total_sent += len(batch_df)

                log(
                    f"Batch {batch_number}: sending={len(batch_df)} | "
                    f"total_sent={total_sent:,} | total_seen={total_seen:,} | skipped={total_skipped:,}"
                )

                try:
                    results, raw = call_llm(batch_df, batch_number)

                    append_jsonl(OUTPUT_JSONL, results)
                    append_csv(OUTPUT_CSV, results)

                    for r in results:
                        oid2 = safe_str(r.get("openalex_id"))
                        if oid2:
                            done_ids.add(oid2)

                    total_written += len(results)

                    missing_n = sum(
                        1 for r in results
                        if r["mechanism_1"] in ["missing output", "batch failed"]
                    )

                    log(
                        f"  wrote={len(results)} | missing_or_failed={missing_n} | "
                        f"total_written={total_written:,}"
                    )

                except Exception as e:
                    err = str(e)
                    log(f"  BATCH FAILED: {err}")
                    fail_batch(batch_df, batch_number, err)

                time.sleep(SLEEP_BETWEEN_BATCHES)

        log(
            f"Chunk {chunk_idx} done | total_seen={total_seen:,} | "
            f"sent={total_sent:,} | skipped={total_skipped:,} | written={total_written:,}"
        )

        if MAX_RECORDS is not None and total_sent >= MAX_RECORDS:
            break

    # Restbuffer
    if buffer and (MAX_RECORDS is None or total_sent < MAX_RECORDS):
        batch_number += 1
        batch_df = pd.DataFrame(buffer)

        if MAX_RECORDS is not None:
            remaining = MAX_RECORDS - total_sent
            batch_df = batch_df.head(remaining)

        total_sent += len(batch_df)

        log(f"Final batch {batch_number}: sending={len(batch_df)} | total_sent={total_sent:,}")

        try:
            results, raw = call_llm(batch_df, batch_number)
            append_jsonl(OUTPUT_JSONL, results)
            append_csv(OUTPUT_CSV, results)
            total_written += len(results)

            log(f"  wrote={len(results)} | total_written={total_written:,}")

        except Exception as e:
            err = str(e)
            log(f"  FINAL BATCH FAILED: {err}")
            fail_batch(batch_df, batch_number, err)

    log("============================================================")
    log("DONE")
    log(f"Total seen: {total_seen:,}")
    log(f"Total skipped: {total_skipped:,}")
    log(f"Total sent: {total_sent:,}")
    log(f"Total written: {total_written:,}")
    log(f"Output CSV: {OUTPUT_CSV}")
    log(f"Output JSONL: {OUTPUT_JSONL}")
    log(f"Failed batches: {FAILED_JSONL}")
    log("============================================================")


if __name__ == "__main__":
    main()