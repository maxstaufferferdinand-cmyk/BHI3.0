from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import time
import random
import pandas as pd
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

INPUT_FILE = (
    BASE_DIR
    / "openalex_hypothesis_annotation_outputs"
    / "final_diversified_high_ml_openalex_candidates_annotated.csv"
)

OUT_DIR = BASE_DIR / "openalex_llm_hypothesis_full_outputs"
OUT_DIR.mkdir(exist_ok=True)

MASTER_RESULTS = OUT_DIR / "full_llm_hypotheses_MASTER.csv"
MASTER_RAW = OUT_DIR / "full_llm_hypotheses_MASTER_raw.jsonl"
MASTER_SUMMARY = OUT_DIR / "full_llm_hypotheses_MASTER_summary.json"

MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-5.5")

# One-by-one calls: safest for long runs
CALL_SLEEP_SECONDS = 1.0

# Every N successful calls, take a longer pause
LONG_PAUSE_EVERY_N_CALLS = 100
LONG_PAUSE_SECONDS = 60

# Long latency tolerance
REQUEST_TIMEOUT_SECONDS = 900

# If API hangs/fails/rate-limits: long backoff
MAX_ATTEMPTS = 7
BACKOFF_SECONDS = [60, 120, 300, 600, 900, 1200, 1800]

# After final failure: save error row and continue.
# You can rerun later; completed ranks are skipped.
CONTINUE_AFTER_FINAL_FAILURE = True


# ============================================================
# HELPERS
# ============================================================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


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


def append_jsonl(obj, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_completed_ranks(paths):
    done = set()

    for path in paths:
        if not path.exists():
            continue

        try:
            old = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except Exception:
            continue

        if "final_rank" in old.columns:
            done.update(
                pd.to_numeric(old["final_rank"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )

    return done


def build_prompt(row):
    problem = safe_str(row.get("hypothesis_input_problem", ""))
    mechanism = safe_str(row.get("hypothesis_input_openalex_mechanism", ""))
    analogue = safe_str(row.get("hypothesis_input_historical_solution_analogue", ""))

    problem_terms = safe_str(row.get("problem_cluster_top_terms", ""))
    solution_terms = safe_str(row.get("solution_cluster_top_terms", ""))

    ml_prob = safe_str(row.get("ml_probability_top20_like", ""))
    historical_weight = safe_str(row.get("historical_cluster_weight", ""))
    historical_edges = safe_str(row.get("historical_n_exact_edges", ""))
    first_year = safe_str(row.get("historical_first_year", ""))

    return f"""
You are helping generate surgical research hypotheses for a cross-domain literature-based discovery project.

Task:
Formulate ONE concise research hypothesis in 1–2 sentences.

Inputs:
- Surgical problem cluster: {problem}
- Surgical problem representative terms: {problem_terms}
- External engineering/OpenAlex mechanism: {mechanism}
- Historical surgical solution analogue: {analogue}
- Historical surgical solution representative terms: {solution_terms}
- ML support: the mapped problem–solution configuration was predicted as high-strength-like with probability {ml_prob}
- Historical analogue metadata: cluster weight {historical_weight}, exact edges {historical_edges}, first year {first_year}

Rules:
- Do NOT claim clinical efficacy.
- Do NOT claim this is proven.
- Do NOT invent trial results, animal data, patient outcomes, device names, or citations.
- Make the hypothesis mechanistic and testable.
- Mention the surgical problem and the external mechanism.
- Prefer a creative but scientifically plausible formulation.
- Keep it suitable for later expert review.
- Output valid JSON only.

JSON schema:
{{
  "hypothesis_title": "...",
  "hypothesis_text": "...",
  "mechanistic_rationale": "...",
  "testing_route": "...",
  "main_risk_or_caveat": "..."
}}
""".strip()


def call_llm(client, prompt):
    # No temperature parameter: gpt-5.5 in your setup rejected non-default temperature.
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate concise, cautious, mechanistic surgical research hypotheses. "
                    "You output valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    return response.choices[0].message.content


def parse_json_safely(text):
    if text is None:
        return {
            "hypothesis_title": "",
            "hypothesis_text": "",
            "mechanistic_rationale": "",
            "testing_route": "",
            "main_risk_or_caveat": "",
            "parse_error": "no_text",
        }

    try:
        obj = json.loads(text)
        obj["parse_error"] = ""
        return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            obj["parse_error"] = ""
            return obj
        except Exception:
            pass

    return {
        "hypothesis_title": "",
        "hypothesis_text": text,
        "mechanistic_rationale": "",
        "testing_route": "",
        "main_risk_or_caveat": "",
        "parse_error": "json_parse_failed",
    }


def make_output_row(row, result, error=""):
    return {
        "timestamp": now_str(),

        "final_rank": row.get("final_rank", ""),
        "annotated_rank": row.get("annotated_rank", ""),
        "review_rank": row.get("review_rank", ""),

        "concept_id": row.get("concept_id", ""),
        "concept_text": row.get("concept_text", ""),
        "n_works": row.get("n_works", ""),
        "n_mentions": row.get("n_mentions", ""),

        "problem_cluster_id": row.get("problem_cluster_id", ""),
        "problem_cluster_label": row.get("problem_cluster_label", ""),
        "problem_cluster_top_terms": row.get("problem_cluster_top_terms", ""),

        "mapped_solution_cluster_id": row.get("mapped_solution_cluster_id", ""),
        "solution_cluster_label": row.get("solution_cluster_label", ""),
        "solution_cluster_top_terms": row.get("solution_cluster_top_terms", ""),

        "ml_probability_top20_like": row.get("ml_probability_top20_like", ""),
        "historical_cluster_weight": row.get("historical_cluster_weight", ""),
        "historical_n_exact_edges": row.get("historical_n_exact_edges", ""),
        "historical_first_year": row.get("historical_first_year", ""),
        "historical_final_edge_strength": row.get("historical_final_edge_strength", ""),
        "historical_within_year_percentile": row.get("historical_within_year_percentile", ""),

        "hypothesis_title": safe_str(result.get("hypothesis_title", "")),
        "hypothesis_text": safe_str(result.get("hypothesis_text", "")),
        "mechanistic_rationale": safe_str(result.get("mechanistic_rationale", "")),
        "testing_route": safe_str(result.get("testing_route", "")),
        "main_risk_or_caveat": safe_str(result.get("main_risk_or_caveat", "")),

        "parse_error": safe_str(result.get("parse_error", "")),
        "api_error": safe_str(error),

        "hypothesis_input_context": row.get("hypothesis_input_context", ""),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="First final_rank to process, inclusive.")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum number of rows to process.")
    parser.add_argument("--part-name", type=str, default="", help="Optional label, e.g. part01.")
    parser.add_argument("--only-errors", action="store_true", help="Reserved for later; not used in this first full-run script.")
    args = parser.parse_args()

    start_rank = int(args.start)
    limit = int(args.limit)
    end_rank = start_rank + limit - 1

    part_label = args.part_name.strip()
    if not part_label:
        part_label = f"rank_{start_rank}_to_{end_rank}"

    part_results = OUT_DIR / f"full_llm_hypotheses_{part_label}.csv"
    part_raw = OUT_DIR / f"full_llm_hypotheses_{part_label}_raw.jsonl"
    part_summary = OUT_DIR / f"full_llm_hypotheses_{part_label}_summary.json"

    print("============================================================")
    print("Full LLM hypothesis generation in parts")
    print("============================================================")
    print("Input:", INPUT_FILE)
    print("Output dir:", OUT_DIR)
    print("Model:", MODEL)
    print("Start rank:", start_rank)
    print("End rank:", end_rank)
    print("Limit:", limit)
    print("Part results:", part_results)
    print("Master results:", MASTER_RESULTS)
    print("Request timeout seconds:", REQUEST_TIMEOUT_SECONDS)
    print("============================================================")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY is not set in this PowerShell session.")

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)

    if "final_rank" not in df.columns:
        raise ValueError("Input file needs a final_rank column.")

    df["final_rank"] = pd.to_numeric(df["final_rank"], errors="coerce")
    df = df.dropna(subset=["final_rank"]).copy()
    df["final_rank"] = df["final_rank"].astype(int)

    df = df.sort_values("final_rank").reset_index(drop=True)

    part_df = df[
        (df["final_rank"] >= start_rank)
        & (df["final_rank"] <= end_rank)
    ].copy()

    if len(part_df) > limit:
        part_df = part_df.head(limit).copy()

    print(f"Total input rows: {len(df):,}")
    print(f"Rows selected for this part: {len(part_df):,}")

    done_ranks = load_completed_ranks([part_results, MASTER_RESULTS])
    print(f"Already completed ranks found: {len(done_ranks):,}")

    n_attempted = 0
    n_success = 0
    n_failed = 0
    n_skipped = 0

    for _, row in part_df.iterrows():
        final_rank = int(row["final_rank"])

        if final_rank in done_ranks:
            print(f"cached final_rank {final_rank}")
            n_skipped += 1
            continue

        n_attempted += 1

        print(
            f"[{now_str()}] final_rank {final_rank} "
            f"({n_attempted}/{len(part_df)})"
        )

        prompt = build_prompt(row)

        raw_text = None
        last_error = ""

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                raw_text = call_llm(client, prompt)
                last_error = ""
                break

            except KeyboardInterrupt:
                print("Interrupted by user. Exiting cleanly.")
                raise

            except Exception as e:
                last_error = str(e)

                wait = BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
                jitter = random.randint(0, 30)
                wait_total = wait + jitter

                print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed:")
                print(f"  {last_error}")
                print(f"  sleeping {wait_total} seconds...")

                time.sleep(wait_total)

        if raw_text is None:
            n_failed += 1

            result = {
                "hypothesis_title": "",
                "hypothesis_text": "",
                "mechanistic_rationale": "",
                "testing_route": "",
                "main_risk_or_caveat": "",
                "parse_error": "",
            }

            out_row = make_output_row(row, result, error=last_error)

            append_csv_row(out_row, part_results)
            append_csv_row(out_row, MASTER_RESULTS)

            append_jsonl(
                {
                    "timestamp": now_str(),
                    "final_rank": final_rank,
                    "status": "failed",
                    "error": last_error,
                    "prompt": prompt,
                    "raw_output": None,
                },
                part_raw,
            )

            append_jsonl(
                {
                    "timestamp": now_str(),
                    "final_rank": final_rank,
                    "status": "failed",
                    "error": last_error,
                    "prompt": prompt,
                    "raw_output": None,
                },
                MASTER_RAW,
            )

            print("  saved failure row and continued")

            if not CONTINUE_AFTER_FINAL_FAILURE:
                raise RuntimeError(last_error)

        else:
            parsed = parse_json_safely(raw_text)
            out_row = make_output_row(row, parsed, error="")

            append_csv_row(out_row, part_results)
            append_csv_row(out_row, MASTER_RESULTS)

            raw_obj = {
                "timestamp": now_str(),
                "final_rank": final_rank,
                "status": "success",
                "prompt": prompt,
                "raw_output": raw_text,
                "parsed": parsed,
            }

            append_jsonl(raw_obj, part_raw)
            append_jsonl(raw_obj, MASTER_RAW)

            n_success += 1
            done_ranks.add(final_rank)

            print("  saved")

        # Gentle pacing
        time.sleep(CALL_SLEEP_SECONDS)

        if n_success > 0 and n_success % LONG_PAUSE_EVERY_N_CALLS == 0:
            print(
                f"  long pause after {n_success} successful calls: "
                f"{LONG_PAUSE_SECONDS} seconds"
            )
            time.sleep(LONG_PAUSE_SECONDS)

    summary = {
        "timestamp": now_str(),
        "model": MODEL,
        "input_file": str(INPUT_FILE),
        "part_label": part_label,
        "start_rank": start_rank,
        "end_rank": end_rank,
        "limit": limit,
        "rows_selected_for_part": int(len(part_df)),
        "n_attempted_this_run": int(n_attempted),
        "n_success_this_run": int(n_success),
        "n_failed_this_run": int(n_failed),
        "n_skipped_cached": int(n_skipped),
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "max_attempts": MAX_ATTEMPTS,
        "continue_after_final_failure": CONTINUE_AFTER_FINAL_FAILURE,
        "part_results": str(part_results),
        "part_raw": str(part_raw),
        "master_results": str(MASTER_RESULTS),
        "master_raw": str(MASTER_RAW),
    }

    with open(part_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(MASTER_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("============================================================")
    print("DONE")
    print(json.dumps(summary, indent=2))
    print("============================================================")


if __name__ == "__main__":
    main()