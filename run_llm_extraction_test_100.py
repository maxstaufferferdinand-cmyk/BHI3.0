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

INPUT_CSV = BASE_DIR / "pubmed_visceral_surgery_technology_negative_clean_1980_2025.csv"

PROMPT_FILE = BASE_DIR / "llm_extraction_prompt_with_12_examples.txt"

SAMPLE_100_CSV = BASE_DIR / "sample_100_for_llm_test.csv"

OUTPUT_JSONL = BASE_DIR / "llm_extraction_test_100_results.jsonl"
OUTPUT_CSV = BASE_DIR / "llm_extraction_test_100_results.csv"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

BATCH_SIZE = 10
RANDOM_STATE = 42

# Set this to True if you want to recreate sample_100 even if it already exists.
FORCE_RESAMPLE = False

# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_prompt() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {PROMPT_FILE}\n"
            "Put llm_extraction_prompt_with_12_examples.txt into the BreakingRules folder."
        )
    return PROMPT_FILE.read_text(encoding="utf-8")


def create_sample_100():
    """
    Creates a mixed sample:
    - random/background records
    - tech-enriched records from relevant modules and title/abstract keywords
    """
    if SAMPLE_100_CSV.exists() and not FORCE_RESAMPLE:
        print(f"Sample already exists, using: {SAMPLE_100_CSV}")
        return

    print("Reading cleaned PubMed CSV...")
    usecols = [
        "pmid", "year", "journal", "title", "abstract",
        "publication_types", "mesh_terms", "keywords",
        "matched_query_modules", "n_matched_query_modules"
    ]

    df = pd.read_csv(
        INPUT_CSV,
        usecols=lambda c: c in usecols,
        encoding="utf-8-sig",
        low_memory=False
    )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year"] >= 1980) & (df["year"] <= 2025)]
    df = df[df["abstract"].notna() & (df["abstract"].astype(str).str.len() > 100)].copy()

    text = (
        df["title"].fillna("").astype(str) + " " +
        df["abstract"].fillna("").astype(str) + " " +
        df["matched_query_modules"].fillna("").astype(str)
    ).str.lower()

    tech_keywords = [
        "robot", "robotic", "da vinci", "hugo", "versius", "senhance",
        "stapler", "stapling", "eea", "endo gia", "echelon", "signia",
        "anastomosis", "anastomotic",
        "mesh", "sealant", "hydrogel", "tachosil", "floseal", "bioglue",
        "ligasure", "harmonic", "thunderbeat", "enseal", "electrosurgical",
        "fluorescence", "icg", "near infrared", "navigation", "augmented reality",
        "computer vision", "artificial intelligence", "machine learning", "deep learning",
        "endoscopic submucosal dissection", "esd", "emr", "eftr", "ftrd",
        "stent", "otSC".lower(), "overstitch", "axios", "lams",
        "balloon dilation", "endoflip", "manometry",
        "magnetic compression", "sensor", "pressure", "tracking"
    ]

    tech_mask = pd.Series(False, index=df.index)
    for kw in tech_keywords:
        tech_mask |= text.str.contains(kw, regex=False, na=False)

    tech_df = df[tech_mask].copy()
    random_df = df[~tech_mask].copy()

    # Build sample: 70 tech-enriched + 30 random/background
    n_tech = min(70, len(tech_df))
    n_random = min(30, len(random_df))

    sample_parts = []

    if n_tech > 0:
        sample_parts.append(tech_df.sample(n=n_tech, random_state=RANDOM_STATE))

    if n_random > 0:
        sample_parts.append(random_df.sample(n=n_random, random_state=RANDOM_STATE + 1))

    sample = pd.concat(sample_parts, ignore_index=True)

    # If fewer than 100 due to some weird case, fill from remaining df
    if len(sample) < 100:
        remaining = df[~df["pmid"].isin(sample["pmid"])]
        fill_n = min(100 - len(sample), len(remaining))
        if fill_n > 0:
            sample = pd.concat(
                [sample, remaining.sample(n=fill_n, random_state=RANDOM_STATE + 2)],
                ignore_index=True
            )

    sample = sample.head(100).copy()
    sample.to_csv(SAMPLE_100_CSV, index=False, encoding="utf-8-sig")

    print(f"Wrote sample: {SAMPLE_100_CSV}")
    print(f"Rows: {len(sample)}")


def make_batch_payload(batch_df: pd.DataFrame) -> str:
    records = []

    for _, row in batch_df.iterrows():
        records.append({
            "pmid": safe_str(row.get("pmid")),
            "year": safe_str(row.get("year")),
            "journal": safe_str(row.get("journal")),
            "title": safe_str(row.get("title")),
            "abstract": safe_str(row.get("abstract")),
            "matched_query_modules": safe_str(row.get("matched_query_modules")),
        })

    return json.dumps(records, ensure_ascii=False, indent=2)


def call_llm(system_prompt: str, batch_df: pd.DataFrame, batch_number: int):
    user_content = (
        "Extract concepts for the following PubMed abstracts.\n"
        "Return JSON only. Return a JSON array with one object per input abstract.\n\n"
        f"Batch number: {batch_number}\n\n"
        f"{make_batch_payload(batch_df)}"
    )

    for attempt in range(1, 6):
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

            # Strip accidental markdown fences if present
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw.replace("json\n", "", 1).replace("JSON\n", "", 1).strip()

            data = json.loads(raw)

            if not isinstance(data, list):
                raise ValueError("Model output is not a JSON array.")

            return data, raw

        except Exception as e:
            print(f"Batch {batch_number}, attempt {attempt}/5 failed: {e}")
            if attempt == 5:
                raise
            time.sleep(2 * attempt)


def normalize_result(obj):
    """
    Expected output columns.
    Missing fields become empty strings.
    """
    return {
        "pmid": safe_str(obj.get("pmid")),
        "decision": safe_str(obj.get("decision")),
        "reason": safe_str(obj.get("reason")),
        "surgical_problem": safe_str(obj.get("surgical_problem")),
        "technical_solution": safe_str(obj.get("technical_solution")),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Model:", MODEL)

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            'Set it in PowerShell with: $env:OPENAI_API_KEY="sk-..."'
        )

    create_sample_100()

    system_prompt = load_prompt()

    print("Reading sample 100...")
    df = pd.read_csv(SAMPLE_100_CSV, encoding="utf-8-sig", low_memory=False)

    required = ["pmid", "year", "journal", "title", "abstract"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Sample file missing columns: {missing}")

    all_results = []

    # Clear previous outputs
    if OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(df), BATCH_SIZE), start=1):
        end = min(start + BATCH_SIZE, len(df))
        batch_df = df.iloc[start:end].copy()

        print(f"Running batch {batch_idx}/{n_batches}: rows {start + 1}-{end}")

        results, raw = call_llm(system_prompt, batch_df, batch_idx)

        normalized = [normalize_result(x) for x in results]

        # Append to JSONL immediately, so progress is saved
        with OUTPUT_JSONL.open("a", encoding="utf-8") as f:
            for obj in normalized:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        all_results.extend(normalized)

        # Save CSV after every batch
        pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        include_count = sum(1 for x in normalized if x["decision"].lower() == "include")
        reject_count = sum(1 for x in normalized if x["decision"].lower() == "reject")
        print(f"  batch results: include={include_count}, reject={reject_count}")

        time.sleep(0.5)

    print("\nDONE")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"CSV:   {OUTPUT_CSV}")

    res_df = pd.DataFrame(all_results)
    print("\nDecision counts:")
    print(res_df["decision"].value_counts(dropna=False))

    print("\nFirst 20 results:")
    print(res_df.head(20).to_string(max_colwidth=100))


if __name__ == "__main__":
    main()