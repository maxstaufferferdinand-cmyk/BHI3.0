from pathlib import Path
import requests
import pandas as pd
import time
import json
import traceback

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

OUTPUT_CSV = BASE_DIR / "openalex_engineering_transfer_2020_2026_full.csv"
OUTPUT_JSONL = BASE_DIR / "openalex_engineering_transfer_2020_2026_full.jsonl"
OUTPUT_QUERIES = BASE_DIR / "openalex_engineering_transfer_2020_2026_full_queries.txt"
OUTPUT_LOG = BASE_DIR / "openalex_engineering_transfer_2020_2026_full_log.txt"
FAILED_JSONL = BASE_DIR / "openalex_engineering_transfer_2020_2026_failed_queries.jsonl"

FROM_DATE = "2020-01-01"
TO_DATE = "2026-12-31"

MAILTO = "n12028114@students.meduniwien.ac.at"

PER_PAGE = 200
SLEEP_SECONDS = 0.15
MAX_RETRIES = 6

# Für echten Full Run: None.
# Für Sicherheitstest: z. B. 5000 setzen.
MAX_TOTAL_RECORDS = None

# Wenn True: bestehende JSONL wird gelesen und bereits geholte OpenAlex IDs werden übersprungen.
RESUME = True

# Optional: Wenn du später mehr willst, auf True setzen und bestehende Outputs löschen.
OVERWRITE_OUTPUTS = False

# ============================================================
# MODULES
# bewusst als grobe, aber gezielte OpenAlex-search strings.
# Keine künstliche Limitierung pro Modul.
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

# Negative terms are only post-filtering title/abstract.
# Keep conservative: exclude clearly irrelevant military/aerospace/civil-only material.
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


def reconstruct_abstract(inv_idx):
    """
    OpenAlex abstract_inverted_index -> plain text.
    """
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
    with OUTPUT_JSONL.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
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

    return seen


def openalex_count(query):
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"from_publication_date:{FROM_DATE},to_publication_date:{TO_DATE},has_abstract:true",
        "search": query,
        "per-page": 1,
        "mailto": MAILTO,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=90)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return int(r.json().get("meta", {}).get("count", 0))
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = attempt * 2
            log(f"Count retry {attempt}/{MAX_RETRIES} after error: {e}. Waiting {wait}s")
            time.sleep(wait)


def fetch_query_full(module_name, query, seen_ids):
    """
    Full cursor paging for one query until no more results.
    Deduplicates against seen_ids and appends continuously.
    """
    url = "https://api.openalex.org/works"
    cursor = "*"
    page = 0
    query_total_new = 0
    query_total_seen = 0
    query_total_kept_after_filter = 0

    while True:
        page += 1

        if MAX_TOTAL_RECORDS is not None and len(seen_ids) >= MAX_TOTAL_RECORDS:
            break

        params = {
            "filter": f"from_publication_date:{FROM_DATE},to_publication_date:{TO_DATE},has_abstract:true",
            "search": query,
            "per-page": PER_PAGE,
            "cursor": cursor,
            "mailto": MAILTO,
        }

        data = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(url, params=params, timeout=120)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = attempt * 2
                log(f"Fetch retry {attempt}/{MAX_RETRIES} for {module_name} | {query} | page {page}: {e}. Waiting {wait}s")
                time.sleep(wait)

        results = data.get("results") or []
        if not results:
            break

        batch_rows = []

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
            query_total_kept_after_filter += 1

            if MAX_TOTAL_RECORDS is not None and len(seen_ids) >= MAX_TOTAL_RECORDS:
                break

        append_jsonl(OUTPUT_JSONL, batch_rows)
        append_csv(OUTPUT_CSV, batch_rows)

        if page == 1 or page % 10 == 0:
            log(
                f"  {module_name} | page={page} | raw_seen={query_total_seen:,} | "
                f"new_written={query_total_new:,} | global_unique={len(seen_ids):,}"
            )

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(SLEEP_SECONDS)

    return {
        "module": module_name,
        "query": query,
        "raw_seen": query_total_seen,
        "new_written": query_total_new,
        "kept_after_filter": query_total_kept_after_filter,
        "pages": page,
    }


def main():
    if OVERWRITE_OUTPUTS:
        for p in [OUTPUT_CSV, OUTPUT_JSONL, OUTPUT_LOG, FAILED_JSONL]:
            if p.exists():
                p.unlink()

    log("============================================================")
    log("OpenAlex Engineering Transfer FULL RUN 2020-2026")
    log("============================================================")
    log(f"Output CSV: {OUTPUT_CSV}")
    log(f"Output JSONL: {OUTPUT_JSONL}")
    log(f"Date range: {FROM_DATE} to {TO_DATE}")
    log(f"PER_PAGE: {PER_PAGE}")
    log(f"MAX_TOTAL_RECORDS: {MAX_TOTAL_RECORDS}")
    log(f"RESUME: {RESUME}")

    with OUTPUT_QUERIES.open("w", encoding="utf-8") as f:
        for module, queries in SEARCH_MODULES.items():
            f.write(f"\n### {module}\n")
            for q in queries:
                f.write(q + "\n")

    seen_ids = load_seen_ids()
    log(f"Already seen IDs: {len(seen_ids):,}")

    summary_rows = []

    for module_name, queries in SEARCH_MODULES.items():
        log("")
        log(f"=== MODULE: {module_name} ===")

        for query in queries:
            if MAX_TOTAL_RECORDS is not None and len(seen_ids) >= MAX_TOTAL_RECORDS:
                log("MAX_TOTAL_RECORDS reached; stopping.")
                break

            try:
                count = openalex_count(query)
                log(f"Query: {query} | count={count:,}")

                result_summary = fetch_query_full(module_name, query, seen_ids)
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

        if MAX_TOTAL_RECORDS is not None and len(seen_ids) >= MAX_TOTAL_RECORDS:
            break

    summary_path = BASE_DIR / "openalex_engineering_transfer_2020_2026_full_query_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    log("")
    log("============================================================")
    log("DONE")
    log(f"Global unique written/seen IDs: {len(seen_ids):,}")
    log(f"Output CSV: {OUTPUT_CSV}")
    log(f"Output JSONL: {OUTPUT_JSONL}")
    log(f"Query summary: {summary_path}")
    log(f"Failed queries: {FAILED_JSONL}")
    log("============================================================")


if __name__ == "__main__":
    main()