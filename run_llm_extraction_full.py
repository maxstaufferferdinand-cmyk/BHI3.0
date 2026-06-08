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

INPUT_CSV = BASE_DIR / "pubmed_visceral_surgery_technology_negative_clean_1980_2025.csv"
PROMPT_FILE = BASE_DIR / "llm_extraction_prompt_with_12_examples.txt"

OUTPUT_JSONL = BASE_DIR / "llm_extraction_full_results.jsonl"
OUTPUT_CSV = BASE_DIR / "llm_extraction_full_results.csv"
FAILED_JSONL = BASE_DIR / "llm_extraction_full_failed_batches.jsonl"
LOG_FILE = BASE_DIR / "llm_extraction_full_log.txt"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Für vollen Run konservativ halten.
# 10 ist meist okay. Wenn JSON-Fehler auftreten: auf 5 reduzieren.
BATCH_SIZE = 10

# CSV wird chunkweise gelesen
CHUNKSIZE = 1000

# Bei OpenAI Rate Limits/Serverfehlern
MAX_RETRIES = 5
SLEEP_BETWEEN_BATCHES = 0.5

# Für Test: z. B. 1000 setzen. Für vollen Run: None.
MAX_RECORDS = None

# Wenn True, werden bereits verarbeitete PMIDs übersprungen.
RESUME = True

# ============================================================
# CLIENT
# ============================================================

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError(
        'OPENAI_API_KEY is not set. In PowerShell use:\n'
        '$env:OPENAI_API_KEY="sk-..."'
    )

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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


def load_prompt() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


def load_done_pmids() -> set:
    """
    Reads existing JSONL output and returns PMIDs already processed.
    This enables resuming after interruption.
    """
    done = set()

    if not OUTPUT_JSONL.exists() or not RESUME:
        return done

    with OUTPUT_JSONL.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pmid = safe_str(obj.get("pmid"))
                if pmid:
                    done.add(pmid)
            except Exception:
                continue

    return done


def normalize_result(obj):
    """
    Expected model output:
    {
      pmid,
      decision,
      reason,
      surgical_problem,
      technical_solution
    }
    """
    return {
        "pmid": safe_str(obj.get("pmid")),
        "decision": safe_str(obj.get("decision")).lower(),
        "reason": safe_str(obj.get("reason")),
        "surgical_problem": safe_str(obj.get("surgical_problem")),
        "technical_solution": safe_str(obj.get("technical_solution")),
    }


def append_jsonl(path: Path, objects: list):
    with path.open("a", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_csv(path: Path, rows: list):
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8-sig")


def make_batch_payload(batch_df: pd.DataFrame) -> str:
    records = []

    for _, row in batch_df.iterrows():
        records.append({
            "pmid": safe_str(row.get("pmid")),
            "year": safe_str(row.get("year")),
            "journal": safe_str(row.get("journal")),
            "title": safe_str(row.get("title")),
            "abstract": safe_str(row.get("abstract")),
        })

    return json.dumps(records, ensure_ascii=False, indent=2)


def strip_markdown_fences(raw: str) -> str:
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


def parse_model_json(raw: str):
    raw = strip_markdown_fences(raw)
    data = json.loads(raw)

    if isinstance(data, dict):
        # Falls Modell {"results": [...]} zurückgibt
        if "results" in data and isinstance(data["results"], list):
            return data["results"]
        # Falls nur ein Objekt kommt
        return [data]

    if isinstance(data, list):
        return data

    raise ValueError("Model output is neither a JSON array nor a JSON object.")


def call_llm(system_prompt: str, batch_df: pd.DataFrame, batch_number: int):
    user_content = (
        "Extract concepts for the following PubMed abstracts.\n"
        "Return JSON only. Return a JSON array with one object per input abstract.\n"
        "Each object must contain exactly these fields:\n"
        "pmid, decision, reason, surgical_problem, technical_solution.\n\n"
        f"Batch number: {batch_number}\n\n"
        f"{make_batch_payload(batch_df)}"
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )

            raw = response.choices[0].message.content.strip()
            data = parse_model_json(raw)

            normalized = [normalize_result(x) for x in data]

            # Check PMIDs. If model forgot some rows, add fail rows.
            input_pmids = [safe_str(x) for x in batch_df["pmid"].tolist()]
            output_pmids = {safe_str(x.get("pmid")) for x in normalized}

            missing_pmids = [p for p in input_pmids if p not in output_pmids]
            if missing_pmids:
                for p in missing_pmids:
                    normalized.append({
                        "pmid": p,
                        "decision": "error",
                        "reason": "missing from model output",
                        "surgical_problem": "",
                        "technical_solution": "",
                    })

            # If model produced extra PMIDs, keep them but mark not fatal.
            return normalized, raw

        except Exception as e:
            last_error = str(e)
            log(f"Batch {batch_number}, attempt {attempt}/{MAX_RETRIES} failed: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"Batch {batch_number} failed after retries: {last_error}")


def fail_batch(batch_df: pd.DataFrame, batch_number: int, error_msg: str):
    fail_rows = []

    for _, row in batch_df.iterrows():
        fail_rows.append({
            "pmid": safe_str(row.get("pmid")),
            "decision": "error",
            "reason": f"batch_failed: {error_msg}",
            "surgical_problem": "",
            "technical_solution": "",
        })

    append_jsonl(OUTPUT_JSONL, fail_rows)
    append_csv(OUTPUT_CSV, fail_rows)

    failed_batch_info = {
        "batch_number": batch_number,
        "pmids": [safe_str(x) for x in batch_df["pmid"].tolist()],
        "error": error_msg,
        "traceback": traceback.format_exc(),
    }
    append_jsonl(FAILED_JSONL, [failed_batch_info])


# ============================================================
# MAIN
# ============================================================

def main():
    log("============================================================")
    log("FULL LLM EXTRACTION RUN")
    log("============================================================")
    log(f"Model: {MODEL}")
    log(f"Input: {INPUT_CSV}")
    log(f"Prompt: {PROMPT_FILE}")
    log(f"Output JSONL: {OUTPUT_JSONL}")
    log(f"Output CSV: {OUTPUT_CSV}")
    log(f"Batch size: {BATCH_SIZE}")
    log(f"Chunk size: {CHUNKSIZE}")
    log(f"Resume: {RESUME}")
    log(f"Max records: {MAX_RECORDS}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    system_prompt = load_prompt()
    done_pmids = load_done_pmids()

    log(f"Already processed PMIDs loaded: {len(done_pmids):,}")

    usecols = ["pmid", "year", "journal", "title", "abstract"]

    total_seen = 0
    total_sent = 0
    total_skipped = 0
    total_written = 0
    batch_number = 0

    buffer = []

    reader = pd.read_csv(
        INPUT_CSV,
        usecols=lambda c: c in usecols,
        encoding="utf-8-sig",
        low_memory=False,
        chunksize=CHUNKSIZE
    )

    for chunk_idx, chunk in enumerate(reader, start=1):
        # Basic safety filtering
        chunk["pmid"] = chunk["pmid"].astype(str).str.strip()
        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce")

        chunk = chunk[
            (chunk["year"] >= 1980) &
            (chunk["year"] <= 2025) &
            (chunk["abstract"].notna()) &
            (chunk["abstract"].astype(str).str.len() > 50)
        ].copy()

        for _, row in chunk.iterrows():
            total_seen += 1

            pmid = safe_str(row.get("pmid"))

            if RESUME and pmid in done_pmids:
                total_skipped += 1
                continue

            buffer.append(row)

            if len(buffer) >= BATCH_SIZE:
                batch_number += 1

                batch_df = pd.DataFrame(buffer)
                buffer = []

                if MAX_RECORDS is not None and total_sent >= MAX_RECORDS:
                    break

                if MAX_RECORDS is not None:
                    remaining = MAX_RECORDS - total_sent
                    batch_df = batch_df.head(remaining)

                total_sent += len(batch_df)

                log(f"Batch {batch_number}: sending {len(batch_df)} abstracts | total_sent={total_sent:,} | total_seen={total_seen:,} | skipped={total_skipped:,}")

                try:
                    results, raw = call_llm(system_prompt, batch_df, batch_number)

                    append_jsonl(OUTPUT_JSONL, results)
                    append_csv(OUTPUT_CSV, results)

                    for r in results:
                        if r["pmid"]:
                            done_pmids.add(r["pmid"])

                    include_n = sum(1 for r in results if r["decision"] == "include")
                    reject_n = sum(1 for r in results if r["decision"] == "reject")
                    error_n = sum(1 for r in results if r["decision"] == "error")

                    total_written += len(results)

                    log(f"  wrote={len(results)} | include={include_n} | reject={reject_n} | error={error_n} | total_written={total_written:,}")

                except Exception as e:
                    error_msg = str(e)
                    log(f"  BATCH FAILED: {error_msg}")
                    fail_batch(batch_df, batch_number, error_msg)

                time.sleep(SLEEP_BETWEEN_BATCHES)

        if MAX_RECORDS is not None and total_sent >= MAX_RECORDS:
            break

        log(f"Chunk {chunk_idx} done | total_seen={total_seen:,} | total_sent={total_sent:,} | skipped={total_skipped:,}")

    # Process remaining buffer
    if buffer and (MAX_RECORDS is None or total_sent < MAX_RECORDS):
        batch_number += 1
        batch_df = pd.DataFrame(buffer)

        if MAX_RECORDS is not None:
            remaining = MAX_RECORDS - total_sent
            batch_df = batch_df.head(remaining)

        total_sent += len(batch_df)

        log(f"Final batch {batch_number}: sending {len(batch_df)} abstracts | total_sent={total_sent:,}")

        try:
            results, raw = call_llm(system_prompt, batch_df, batch_number)

            append_jsonl(OUTPUT_JSONL, results)
            append_csv(OUTPUT_CSV, results)

            include_n = sum(1 for r in results if r["decision"] == "include")
            reject_n = sum(1 for r in results if r["decision"] == "reject")
            error_n = sum(1 for r in results if r["decision"] == "error")

            total_written += len(results)

            log(f"  wrote={len(results)} | include={include_n} | reject={reject_n} | error={error_n} | total_written={total_written:,}")

        except Exception as e:
            error_msg = str(e)
            log(f"  FINAL BATCH FAILED: {error_msg}")
            fail_batch(batch_df, batch_number, error_msg)

    log("============================================================")
    log("DONE")
    log(f"Total seen: {total_seen:,}")
    log(f"Total skipped due to resume: {total_skipped:,}")
    log(f"Total sent: {total_sent:,}")
    log(f"Total written: {total_written:,}")
    log(f"Output JSONL: {OUTPUT_JSONL}")
    log(f"Output CSV: {OUTPUT_CSV}")
    log(f"Failed batches: {FAILED_JSONL}")
    log("============================================================")


if __name__ == "__main__":
    main()