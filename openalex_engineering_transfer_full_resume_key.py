from pathlib import Path
import os
import requests
import pandas as pd
import time
import json
import traceback
import re

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

OUTPUT_CSV = BASE_DIR / "openalex_engineering_transfer_2020_2026_full.csv"
OUTPUT_JSONL = BASE_DIR / "openalex_engineering_transfer_2020_2026_full.jsonl"
OUTPUT_QUERIES = BASE_DIR / "openalex_engineering_transfer_2020_2026_full_queries.txt"
OUTPUT_LOG = BASE_DIR / "openalex_engineering_transfer_2020_2026_full_log.txt"
FAILED_JSONL = BASE_DIR / "openalex_engineering_transfer_2020_2026_failed_queries.jsonl"
STATE_JSON = BASE_DIR / "openalex_engineering_transfer_2020_2026_state.json"

FROM_DATE = "2020-01-01"
TO_DATE = "2026-12-31"

MAILTO = "n12028114@students.meduniwien.ac.at"

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()
if not OPENALEX_API_KEY:
    raise RuntimeError(
        "OPENALEX_API_KEY is not set.\n"
        'Set it in PowerShell with: $env:OPENALEX_API_KEY="YOUR_KEY"'
    )

# OpenAlex docs recommend max 100 per page
PER_PAGE = 100

# slower but safer with paid/budgeted API
SLEEP_SECONDS = 0.25
MAX_RETRIES = 8

# Full run: None
MAX_TOTAL_RECORDS = None

RESUME = True
OVERWRITE_OUTPUTS = False

# If True, parse log and skip queries that were already fully finished.
SKIP_FINISHED_QUERIES_FROM_LOG = True

# ============================================================
# SEARCH MODULES
# ============================================================

SEARCH_MODULES = {
    "mechanical_engineering": [
        '"mechanical engineering"',
    ],

    "materials_engineering": [
        '"materials engineering"',
    ],

    "microfluidics": [
        "microfluidics",
        "microfluidic",
    ],

    "pressure_flow_control": [
        '"pressure regulation" fluid',
        '"pressure control" fluid',
        '"flow control" pressure',
        '"closed-loop pressure control"',
        '"fluidic pressure control"',
        '"capillary flow" material',
        '"porous material" fluid transport',
        '"leak detection" pressure',
    ],

    "sensors": [
        '"pressure sensor"',
        '"force sensor"',
        '"tactile sensor"',
        '"strain sensor"',
        '"fiber optic sensor" pressure',
        '"flexible sensor"',
        '"soft sensor"',
        '"shape sensing"',
        '"electromagnetic tracking"',
        '"sensor fusion" tracking',
    ],

    "sealing_leakage": [
        '"leak sealing"',
        '"leakage sealing"',
        '"self-sealing material"',
        '"fluid leakage" sealant',
        '"wet surface" sealant',
        '"anti-leakage" material',
        '"sealing hydrogel"',
        '"barrier membrane" leakage',
    ],

    "bioadhesive_hydrogel": [
        '"bioadhesive hydrogel"',
        '"adhesive hydrogel"',
        '"tissue adhesive hydrogel"',
        '"surgical adhesive hydrogel"',
        '"hydrogel sealant"',
        '"hemostatic hydrogel"',
        '"mussel-inspired hydrogel"',
        '"wet adhesion hydrogel"',
        '"self-healing hydrogel" adhesive',
        '"injectable adhesive hydrogel"',
    ],

    "soft_robotics": [
        '"soft robotics"',
        '"soft robot"',
        '"soft actuator"',
        '"soft gripper"',
        '"pneumatic soft actuator"',
        '"magnetic soft robot"',
        '"miniature soft robot"',
        '"soft robotic catheter"',
    ],

    "continuum_robot": [
        '"continuum robot"',
        '"continuum robotics"',
        '"tendon-driven continuum robot"',
        '"concentric tube robot"',
        '"flexible continuum manipulator"',
    ],
}

NEGATIVE_TERMS = [
    "fighter jet",
    "missile",
    "weapon",
    "warfare",
    "military aircraft",
    "rocket",
    "satellite orbit",
    "aircraft wing",
    "aerospace vehicle",
    "gas turbine",
    "combustion engine",
    "diesel engine",
    "ship hull",
    "naval",
    "ballistic",
    "railway traffic",
    "traffic flow",
    "bridge construction",
    "building construction",
    "earthquake engineering",
    "concrete beam",
    "asphalt pavement",
]

# ============================================================
# HELPERS
# ============================================================

def log(msg: str):
    print(msg, flush=True)
    with OUTPUT_LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def query_key(module_name: str, query: str) -> str:
    return f"{module_name}||{query}"


def add_auth_params(params: dict) -> dict:
    params["api_key"] = OPENALEX_API_KEY
    if MAILTO:
        params["mailto"] = MAILTO
    return params


def reconstruct_abstract(inv_idx):
    if not inv_idx or not isinstance(inv_idx, dict):
        return ""

    positions = []
    for word, pos_list in inv_idx.items():
        if isinstance(pos_list, list):
            for pos in pos_list:
                positions.append((pos, word))

    if not positions:
        return ""

    positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positions)


def get_source_name(work):
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name") or ""


def get_source_type(work):
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("type") or ""


def get_authors(work, max_n=20):
    authorships = work.get("authorships") or []
    names = []
    for a in authorships[:max_n]:
        au = a.get("author") or {}
        name = au.get("display_name")
        if name:
            names.append(name)
    return "; ".join(names)


def get_concepts(work, max_n=20):
    concepts = work.get("concepts") or []
    out = []
    for c in concepts[:max_n]:
        name = c.get("display_name")
        score = c.get("score")
        if name:
            if isinstance(score, (int, float)):
                out.append(f"{name} ({score:.2f})")
            else:
                out.append(name)
    return "; ".join(out)


def get_topics(work, max_n=12):
    topics = work.get("topics") or []
    out = []
    for t in topics[:max_n]:
        name = t.get("display_name")
        score = t.get("score")
        if name:
            if isinstance(score, (int, float)):
                out.append(f"{name} ({score:.2f})")
            else:
                out.append(name)
    return "; ".join(out)


def is_negative_topic(title, abstract):
    txt = f"{title} {abstract}".lower()
    return any(term.lower() in txt for term in NEGATIVE_TERMS)


def normalize_work(work, module_name, query):
    title = work.get("display_name") or ""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

    if not abstract:
        return None

    if is_negative_topic(title, abstract):
        return None

    return {
        "openalex_id": work.get("id", ""),
        "doi": work.get("doi", ""),
        "year": work.get("publication_year", ""),
        "publication_date": work.get("publication_date", ""),
        "source": get_source_name(work),
        "source_type": get_source_type(work),
        "title": title,
        "abstract": abstract,
        "cited_by_count": work.get("cited_by_count", ""),
        "type": work.get("type", ""),
        "authors": get_authors(work),
        "concepts": get_concepts(work),
        "topics": get_topics(work),
        "module": module_name,
        "query": query,
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


def load_seen_ids():
    seen = set()
    if not RESUME or not OUTPUT_JSONL.exists():
        return seen

    log(f"Loading existing IDs from {OUTPUT_JSONL} ...")
    n = 0
    with OUTPUT_JSONL.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                oid = obj.get("openalex_id")
                if oid:
                    seen.add(oid)
            except Exception:
                continue

            if n % 100000 == 0:
                log(f"  loaded {n:,} lines, unique IDs={len(seen):,}")

    return seen


def load_finished_queries_from_log():
    """
    Finds lines like:
    FINISHED query: "mechanical engineering" | count=...
    and maps them to module/query by using surrounding MODULE line.
    """
    finished = set()

    if not OUTPUT_LOG.exists() or not SKIP_FINISHED_QUERIES_FROM_LOG:
        return finished

    current_module = None

    with OUTPUT_LOG.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            m = re.match(r"=== MODULE: (.+) ===", line)
            if m:
                current_module = m.group(1).strip()
                continue

            if line.startswith("FINISHED query:") and current_module:
                # extract text between "FINISHED query:" and " | count="
                q = line.replace("FINISHED query:", "", 1).strip()
                if " | count=" in q:
                    q = q.split(" | count=", 1)[0].strip()
                finished.add(query_key(current_module, q))

    return finished


def load_state():
    if not STATE_JSON.exists():
        return {}

    try:
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state):
    STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def request_json(url, params, label):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=120)

            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

            r.raise_for_status()
            data = r.json()

            cost = data.get("meta", {}).get("cost_usd")
            remaining = r.headers.get("x-ratelimit-remaining")
            reset = r.headers.get("x-ratelimit-reset")

            if cost is not None:
                log(f"    cost_usd={cost} | x-ratelimit-remaining={remaining} | reset={reset}")

            return data

        except Exception as e:
            last_error = str(e)

            # If daily budget exhausted, retries won't help until reset/payment.
            if "Insufficient budget" in last_error or "Rate limit exceeded" in last_error:
                log(f"{label}: budget/rate problem: {last_error}")
                raise

            if attempt == MAX_RETRIES:
                raise

            wait = attempt * 3
            log(f"{label}: retry {attempt}/{MAX_RETRIES} after error: {e}. Waiting {wait}s")
            time.sleep(wait)

    raise RuntimeError(last_error)


def openalex_count(query):
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"from_publication_date:{FROM_DATE},to_publication_date:{TO_DATE},has_abstract:true",
        "search": query,
        "per-page": 1,
    }
    params = add_auth_params(params)
    data = request_json(url, params, label=f"COUNT {query}")
    return int(data.get("meta", {}).get("count", 0))


def fetch_query_full(module_name, query, seen_ids, state):
    url = "https://api.openalex.org/works"
    key = query_key(module_name, query)

    # If state has cursor, resume from it.
    query_state = state.get(key, {})
    cursor = query_state.get("next_cursor", "*")
    page = int(query_state.get("page", 0))

    query_total_new = int(query_state.get("query_total_new", 0))
    query_total_seen = int(query_state.get("query_total_seen", 0))

    log(f"Starting/resuming query key={key} at page={page + 1}, cursor={'*' if cursor == '*' else 'saved_cursor'}")

    while True:
        page += 1

        if MAX_TOTAL_RECORDS is not None and len(seen_ids) >= MAX_TOTAL_RECORDS:
            break

        params = {
            "filter": f"from_publication_date:{FROM_DATE},to_publication_date:{TO_DATE},has_abstract:true",
            "search": query,
            "per-page": PER_PAGE,
            "cursor": cursor,
        }
        params = add_auth_params(params)

        data = request_json(
            url,
            params,
            label=f"FETCH {module_name} | {query} | page {page}"
        )

        results = data.get("results") or []
        if not results:
            break

        batch_rows = []
        raw_this_page = len(results)

        for work in results:
            query_total_seen += 1

            oid = work.get("id") or ""
            if not oid:
                continue

            if oid in seen_ids:
                continue

            row = normalize_work(work, module_name, query)
            if row is None:
                continue

            seen_ids.add(oid)
            batch_rows.append(row)
            query_total_new += 1

            if MAX_TOTAL_RECORDS is not None and len(seen_ids) >= MAX_TOTAL_RECORDS:
                break

        append_jsonl(OUTPUT_JSONL, batch_rows)
        append_csv(OUTPUT_CSV, batch_rows)

        next_cursor = data.get("meta", {}).get("next_cursor")

        # Save state after every page
        state[key] = {
            "module": module_name,
            "query": query,
            "next_cursor": next_cursor,
            "page": page,
            "query_total_new": query_total_new,
            "query_total_seen": query_total_seen,
            "global_unique": len(seen_ids),
            "status": "running",
        }
        save_state(state)

        if page == 1 or page % 10 == 0:
            log(
                f"  {module_name} | page={page} | raw_page={raw_this_page} | "
                f"raw_seen={query_total_seen:,} | new_written={query_total_new:,} | "
                f"global_unique={len(seen_ids):,}"
            )

        if not next_cursor:
            break

        cursor = next_cursor
        time.sleep(SLEEP_SECONDS)

    # Mark complete
    state[key] = {
        "module": module_name,
        "query": query,
        "next_cursor": None,
        "page": page,
        "query_total_new": query_total_new,
        "query_total_seen": query_total_seen,
        "global_unique": len(seen_ids),
        "status": "finished",
    }
    save_state(state)

    return {
        "module": module_name,
        "query": query,
        "raw_seen": query_total_seen,
        "new_written": query_total_new,
        "pages": page,
    }


def main():
    if OVERWRITE_OUTPUTS:
        for p in [OUTPUT_CSV, OUTPUT_JSONL, OUTPUT_LOG, FAILED_JSONL, STATE_JSON]:
            if p.exists():
                p.unlink()

    log("============================================================")
    log("OpenAlex Engineering Transfer FULL RUN 2020-2026 with API key + resume")
    log("============================================================")
    log(f"Output CSV: {OUTPUT_CSV}")
    log(f"Output JSONL: {OUTPUT_JSONL}")
    log(f"Date range: {FROM_DATE} to {TO_DATE}")
    log(f"PER_PAGE: {PER_PAGE}")
    log(f"MAX_TOTAL_RECORDS: {MAX_TOTAL_RECORDS}")
    log(f"RESUME: {RESUME}")
    log(f"API key present: {bool(OPENALEX_API_KEY)}")

    with OUTPUT_QUERIES.open("w", encoding="utf-8") as f:
        for module, queries in SEARCH_MODULES.items():
            f.write(f"\n### {module}\n")
            for q in queries:
                f.write(q + "\n")

    seen_ids = load_seen_ids()
    log(f"Already seen IDs: {len(seen_ids):,}")

    finished_from_log = load_finished_queries_from_log()
    log(f"Finished queries detected from log: {len(finished_from_log):,}")

    state = load_state()

    summary_rows = []

    for module_name, queries in SEARCH_MODULES.items():
        log("")
        log(f"=== MODULE: {module_name} ===")

        for query in queries:
            key = query_key(module_name, query)

            if key in finished_from_log:
                log(f"SKIP already finished from log: {query}")
                continue

            if state.get(key, {}).get("status") == "finished":
                log(f"SKIP already finished from state: {query}")
                continue

            if MAX_TOTAL_RECORDS is not None and len(seen_ids) >= MAX_TOTAL_RECORDS:
                log("MAX_TOTAL_RECORDS reached; stopping.")
                break

            try:
                count = openalex_count(query)
                log(f"Query: {query} | count={count:,}")

                result_summary = fetch_query_full(module_name, query, seen_ids, state)
                result_summary["count"] = count
                summary_rows.append(result_summary)

                log(
                    f"FINISHED query: {query} | count={count:,} | "
                    f"raw_seen={result_summary['raw_seen']:,} | "
                    f"new_written={result_summary['new_written']:,} | "
                    f"global_unique={len(seen_ids):,}"
                )

            except Exception as e:
                error_obj = {
                    "module": module_name,
                    "query": query,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(FAILED_JSONL, [error_obj])
                log(f"FAILED query: {query} | {e}")

                # If budget issue, stop entire run cleanly.
                if "Insufficient budget" in str(e) or "Rate limit exceeded" in str(e):
                    log("Stopping because OpenAlex budget/rate limit was reached.")
                    break

        if MAX_TOTAL_RECORDS is not None and len(seen_ids) >= MAX_TOTAL_RECORDS:
            break

    summary_path = BASE_DIR / "openalex_engineering_transfer_2020_2026_full_query_summary_incremental.csv"
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    log("")
    log("============================================================")
    log("DONE or PAUSED")
    log(f"Global unique IDs currently: {len(seen_ids):,}")
    log(f"Output CSV: {OUTPUT_CSV}")
    log(f"Output JSONL: {OUTPUT_JSONL}")
    log(f"State: {STATE_JSON}")
    log(f"Failed queries: {FAILED_JSONL}")
    log("============================================================")


if __name__ == "__main__":
    main()