from pathlib import Path
import os
import json
import time
import pandas as pd
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

INPUT_CSV = BASE_DIR / "openalex_engineering_transfer_2020_2026_full.csv"

OUTPUT_CSV = BASE_DIR / "openalex_engineering_mechanisms_test_20.csv"
OUTPUT_JSONL = BASE_DIR / "openalex_engineering_mechanisms_test_20.jsonl"
OUTPUT_RAW_JSONL = BASE_DIR / "openalex_engineering_mechanisms_test_20_raw.jsonl"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

N_TEST = 20
BATCH_SIZE = 10
TEMPERATURE = 0

# If True: use first 20 rows.
# If False: sample 20 rows randomly.
USE_FIRST_ROWS = True
RANDOM_STATE = 42

# ============================================================
# CLIENT
# ============================================================

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        'Set it in PowerShell with: $env:OPENAI_API_KEY="sk-..."'
    )

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ============================================================
# PROMPT
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
- Try to avoid generic or broadly used terms like magnetic field, artificial intelligence etc.
- Good mechanisms include:
  - wet hydrogel adhesion
  - porous membrane filtration
  - self-healing material
  - shape memory actuation
  - continuum manipulation
  - microfluidic valving
  - thermal ablation
  - ultrasonic energy transfer
- Avoid broad generic phrases:
  - engineering
  - material
  - model
  - system
  - device
  - technology
  - experiment
  - performance improvement
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
    # Keep it short but do not hard-truncate words aggressively.
    words = text.split()
    if len(words) > 8:
        text = " ".join(words[:8])
    return text


def normalize_result(obj):
    return {
        "openalex_id": safe_str(obj.get("openalex_id")),
        "mechanism_1": normalize_mechanism(obj.get("mechanism_1")),
        "mechanism_2": normalize_mechanism(obj.get("mechanism_2")),
        "confidence": obj.get("confidence", ""),
    }


def make_payload(batch_df: pd.DataFrame) -> str:
    records = []
    for _, row in batch_df.iterrows():
        records.append({
            "openalex_id": safe_str(row.get("openalex_id")),
            "year": safe_str(row.get("year")),
            "source": safe_str(row.get("source")),
            "title": safe_str(row.get("title")),
            "abstract": safe_str(row.get("abstract")),
            "module": safe_str(row.get("module")),
            "query": safe_str(row.get("query")),
            "concepts": safe_str(row.get("concepts")),
            "topics": safe_str(row.get("topics")),
        })
    return json.dumps(records, ensure_ascii=False, indent=2)


def call_llm(batch_df: pd.DataFrame, batch_number: int):
    user_prompt = (
        "Extract exactly two central transferable engineering mechanisms from each abstract.\n"
        "Return JSON only, as an array with one object per input abstract.\n\n"
        f"Batch number: {batch_number}\n\n"
        f"{make_payload(batch_df)}"
    )

    for attempt in range(1, 6):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw = response.choices[0].message.content
            clean = strip_json_fences(raw)
            data = json.loads(clean)

            if isinstance(data, dict) and "results" in data:
                data = data["results"]

            if not isinstance(data, list):
                raise ValueError("Model output is not a JSON array.")

            normalized = [normalize_result(x) for x in data]

            # Ensure all input IDs appear. Add error rows for missing IDs.
            input_ids = [safe_str(x) for x in batch_df["openalex_id"].tolist()]
            output_ids = {safe_str(x["openalex_id"]) for x in normalized}

            for oid in input_ids:
                if oid not in output_ids:
                    normalized.append({
                        "openalex_id": oid,
                        "mechanism_1": "missing output",
                        "mechanism_2": "missing output",
                        "confidence": 0.0,
                    })

            return normalized, raw

        except Exception as e:
            print(f"Batch {batch_number}, attempt {attempt}/5 failed: {e}")
            if attempt == 5:
                raise
            time.sleep(2 * attempt)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Engineering mechanism extraction test")
    print("Model:", MODEL)
    print("Input:", INPUT_CSV)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    usecols = [
        "openalex_id", "doi", "year", "source", "title", "abstract",
        "module", "query", "concepts", "topics"
    ]

    df = pd.read_csv(
        INPUT_CSV,
        usecols=lambda c: c in usecols,
        encoding="utf-8-sig",
        low_memory=False
    )

    df = df[df["abstract"].notna() & (df["abstract"].astype(str).str.len() > 50)].copy()

    if USE_FIRST_ROWS:
        sample = df.head(N_TEST).copy()
    else:
        sample = df.sample(n=N_TEST, random_state=RANDOM_STATE).copy()

    print(f"Testing rows: {len(sample)}")

    # Remove old output
    for p in [OUTPUT_CSV, OUTPUT_JSONL, OUTPUT_RAW_JSONL]:
        if p.exists():
            p.unlink()

    all_results = []

    n_batches = (len(sample) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(sample), BATCH_SIZE), start=1):
        end = min(start + BATCH_SIZE, len(sample))
        batch_df = sample.iloc[start:end].copy()

        print(f"Batch {batch_idx}/{n_batches}: rows {start + 1}-{end}")

        results, raw = call_llm(batch_df, batch_idx)

        # Add source metadata back to results
        meta = batch_df[
            ["openalex_id", "year", "source", "title", "module", "query"]
        ].copy()

        res_df = pd.DataFrame(results)
        merged = meta.merge(res_df, on="openalex_id", how="left")

        batch_rows = merged.to_dict(orient="records")
        all_results.extend(batch_rows)

        with OUTPUT_JSONL.open("a", encoding="utf-8") as f:
            for row in batch_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with OUTPUT_RAW_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "batch": batch_idx,
                "raw_model_output": raw
            }, ensure_ascii=False) + "\n")

        pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        print(pd.DataFrame(batch_rows)[
            ["year", "source", "title", "mechanism_1", "mechanism_2", "confidence"]
        ].to_string(max_colwidth=80))

        time.sleep(0.5)

    print("\nDONE")
    print("CSV:", OUTPUT_CSV)
    print("JSONL:", OUTPUT_JSONL)
    print("RAW:", OUTPUT_RAW_JSONL)


if __name__ == "__main__":
    main()