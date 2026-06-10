from pathlib import Path
from datetime import datetime
import os
import json
import time
import pandas as pd
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

INPUT_FILE = (
    BASE_DIR
    / "openalex_hypothesis_annotation_outputs"
    / "top1000_for_manual_review.csv"
)

OUT_DIR = BASE_DIR / "openalex_llm_hypothesis_test_outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_SAMPLE_INPUT = OUT_DIR / "llm_test_50_random_input_candidates.csv"
OUT_RESULTS = OUT_DIR / "llm_test_50_random_hypotheses.csv"
OUT_RAW = OUT_DIR / "llm_test_50_random_raw_outputs.jsonl"
OUT_SUMMARY = OUT_DIR / "llm_test_50_random_summary.json"

N_TEST = 50
RANDOM_STATE = 42

# OpenAI model can be overridden in PowerShell:
# $env:OPENAI_LLM_MODEL="gpt-5.5"
MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-5.5")

MAX_ATTEMPTS = 5
SLEEP_SECONDS = 0.8


# ============================================================
# HELPERS
# ============================================================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def build_prompt(row):
    problem = safe_str(row.get("hypothesis_input_problem", ""))
    mechanism = safe_str(row.get("hypothesis_input_openalex_mechanism", ""))
    analogue = safe_str(row.get("hypothesis_input_historical_solution_analogue", ""))
    problem_terms = safe_str(row.get("problem_cluster_top_terms", ""))
    solution_terms = safe_str(row.get("solution_cluster_top_terms", ""))
    ml_prob = safe_str(row.get("ml_probability_top20_like", ""))

    return f"""
You are helping generate surgical research hypotheses for a cross-domain literature-based discovery project.

Task:
Formulate ONE concise hypothesis in 1â€“2 sentences.

Inputs:
- Surgical problem cluster: {problem}
- Surgical problem representative terms: {problem_terms}
- External OpenAlex engineering mechanism: {mechanism}
- Historical surgical solution analogue: {analogue}
- Historical surgical solution representative terms: {solution_terms}
- ML support: the mapped problemâ€“solution configuration was predicted as high-strength-like with probability {ml_prob}

Rules:
- Do NOT claim clinical efficacy.
- Do NOT say this is proven.
- Do NOT invent specific trial results, animal data, or patient outcomes.
- Make the hypothesis mechanistic and testable.
- Mention the surgical problem and the engineering mechanism.
- Prefer a creative but scientifically plausible formulation.
- Output valid JSON only.

JSON schema:
{{
  "hypothesis_title": "...",
  "hypothesis_text": "...",
  "mechanistic_rationale": "...",
  "testing_route": "..."
}}
""".strip()


def call_llm(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate concise, cautious, mechanistic surgical research hypotheses. "
                    "You must output valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

    )

    return response.choices[0].message.content


def parse_json_safely(text):
    try:
        return json.loads(text)
    except Exception:
        # Try to recover if model wrapped JSON in text accidentally
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass

    return {
        "hypothesis_title": "",
        "hypothesis_text": text,
        "mechanistic_rationale": "",
        "testing_route": "",
    }


def append_csv_row(row_dict, path):
    df = pd.DataFrame([row_dict])
    header = not path.exists()
    df.to_csv(
        path,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8-sig",
    )


def append_raw_jsonl(obj, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    print("============================================================")
    print("LLM test run: 50 random OpenAlex surgical hypotheses")
    print("============================================================")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in PowerShell first."
        )

    print("Input:", INPUT_FILE)
    print("Model:", MODEL)

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)

    print(f"Input rows: {len(df):,}")
    print("Columns:", list(df.columns))

    if len(df) < N_TEST:
        sample = df.copy()
    else:
        sample = df.sample(n=N_TEST, random_state=RANDOM_STATE).copy()

    sample = sample.reset_index(drop=True)
    sample["llm_test_index"] = range(1, len(sample) + 1)

    sample.to_csv(OUT_SAMPLE_INPUT, index=False, encoding="utf-8-sig")
    print("Saved sampled input:", OUT_SAMPLE_INPUT)

    # Resume: skip already completed llm_test_index
    done_indices = set()

    if OUT_RESULTS.exists():
        old = pd.read_csv(OUT_RESULTS, encoding="utf-8-sig", low_memory=False)
        if "llm_test_index" in old.columns:
            done_indices = set(old["llm_test_index"].astype(int).tolist())

    print(f"Already completed: {len(done_indices)}")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    n_done_now = 0

    for _, row in sample.iterrows():
        idx = int(row["llm_test_index"])

        if idx in done_indices:
            print(f"cached {idx}/{len(sample)}")
            continue

        prompt = build_prompt(row)

        print(f"LLM hypothesis {idx}/{len(sample)}")

        raw_text = None
        last_error = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                raw_text = call_llm(client, prompt)
                break

            except Exception as e:
                last_error = str(e)
                wait = min(120, 5 * attempt)
                print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed: {e}")
                print(f"  waiting {wait}s...")
                time.sleep(wait)

        if raw_text is None:
            result = {
                "hypothesis_title": "",
                "hypothesis_text": "",
                "mechanistic_rationale": "",
                "testing_route": "",
                "error": last_error,
            }
        else:
            parsed = parse_json_safely(raw_text)
            result = {
                "hypothesis_title": safe_str(parsed.get("hypothesis_title", "")),
                "hypothesis_text": safe_str(parsed.get("hypothesis_text", "")),
                "mechanistic_rationale": safe_str(parsed.get("mechanistic_rationale", "")),
                "testing_route": safe_str(parsed.get("testing_route", "")),
                "error": "",
            }

        out_row = {
            "timestamp": now_str(),
            "llm_test_index": idx,
            "review_rank": row.get("review_rank", ""),
            "annotated_rank": row.get("annotated_rank", ""),
            "final_rank": row.get("final_rank", ""),
            "concept_id": row.get("concept_id", ""),
            "concept_text": row.get("concept_text", ""),
            "problem_cluster_id": row.get("problem_cluster_id", ""),
            "mapped_solution_cluster_id": row.get("mapped_solution_cluster_id", ""),
            "ml_probability_top20_like": row.get("ml_probability_top20_like", ""),
            "problem_cluster_label": row.get("problem_cluster_label", ""),
            "solution_cluster_label": row.get("solution_cluster_label", ""),
            "hypothesis_input_context": row.get("hypothesis_input_context", ""),
            **result,
        }

        append_csv_row(out_row, OUT_RESULTS)

        append_raw_jsonl(
            {
                "timestamp": now_str(),
                "llm_test_index": idx,
                "prompt": prompt,
                "raw_output": raw_text,
                "parsed_result": result,
            },
            OUT_RAW,
        )

        n_done_now += 1
        print("  saved")

        time.sleep(SLEEP_SECONDS)

    summary = {
        "timestamp": now_str(),
        "input_file": str(INPUT_FILE),
        "sample_input": str(OUT_SAMPLE_INPUT),
        "results": str(OUT_RESULTS),
        "raw_outputs": str(OUT_RAW),
        "n_test": int(len(sample)),
        "n_previously_done": int(len(done_indices)),
        "n_done_this_run": int(n_done_now),
        "model": MODEL,
        "random_state": RANDOM_STATE,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("============================================================")
    print("DONE")
    print("Results:")
    print(OUT_RESULTS)
    print("Raw outputs:")
    print(OUT_RAW)
    print("Summary:")
    print(OUT_SUMMARY)
    print("============================================================")


if __name__ == "__main__":
    main()
