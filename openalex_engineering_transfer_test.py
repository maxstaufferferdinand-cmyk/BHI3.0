from pathlib import Path
import requests
import pandas as pd
import time
import json
from urllib.parse import quote

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

OUTPUT_CSV = BASE_DIR / "openalex_engineering_transfer_2020_2026_test.csv"
OUTPUT_JSONL = BASE_DIR / "openalex_engineering_transfer_2020_2026_test.jsonl"
OUTPUT_QUERIES = BASE_DIR / "openalex_engineering_transfer_2020_2026_queries.txt"

FROM_DATE = "2020-01-01"
TO_DATE = "2026-12-31"

# OpenAlex empfiehlt mailto für die polite pool.
# Setze hier deine E-Mail ein.
MAILTO = "n12028114@students.meduniwien.ac.at"

PER_PAGE = 200
MAX_PAGES_PER_QUERY = 3        # Testlauf: max 600 Works pro Query
MAX_TOTAL_RECORDS = 3000       # Testlauf-Hardcap
SLEEP_SECONDS = 0.15

# ============================================================
# SEARCH MODULES
# engmaschig: übertragbare Engineering-Prinzipien,
# keine allgemeine Aerospace/Kampfjet-Suche
# ============================================================

SEARCH_MODULES = {
    "fluid_pressure_flow_control": [
        '"flow control" pressure regulation microfluidic valve',
        '"pressure regulation" fluidic system sensor',
        '"leak detection" fluid system pressure',
        '"closed-loop" pressure control fluid',
        '"communicating vessels" flow pressure sensor',
        '"capillary flow" porous material control',
        '"microfluidic" valve actuator soft',
    ],

    "sealing_joining_leakage_barriers": [
        '"leak sealing" material hydrogel',
        '"self-sealing" material fluid leakage',
        '"adhesive hydrogel" wet tissue',
        '"bioadhesive" hydrogel sealing',
        '"sealant" wet surface mechanical',
        '"reversible adhesion" wet surface',
        '"interfacial toughness" hydrogel adhesive',
    ],

    "soft_robotics_actuation_confined_spaces": [
        '"soft robot" confined space manipulation',
        '"continuum robot" flexible manipulation',
        '"soft actuator" pneumatic manipulation',
        '"magnetic soft robot" navigation',
        '"miniature robot" soft actuator',
        '"tendon-driven" continuum robot',
        '"haptic feedback" teleoperation soft robot',
    ],

    "sensors_force_pressure_strain_tactile": [
        '"force sensor" flexible tactile',
        '"pressure sensor" soft material',
        '"strain sensor" flexible wearable',
        '"tactile sensor" robotic gripper',
        '"fiber optic sensor" force pressure',
        '"capacitive pressure sensor" flexible',
        '"soft sensor" deformation pressure',
    ],

    "smart_materials_shape_memory_responsive": [
        '"shape memory alloy" actuator',
        '"nitinol" actuator biomedical',
        '"stimuli-responsive hydrogel" actuator',
        '"self-healing material" hydrogel',
        '"magnetically responsive material" soft actuator',
        '"thermoresponsive hydrogel" mechanical',
        '"deployable structure" soft material',
    ],

    "surface_friction_gripping_handling": [
        '"soft gripper" delicate object',
        '"variable stiffness" gripper',
        '"friction control" soft robot',
        '"jamming gripper" soft robotic',
        '"gecko-inspired adhesive" wet',
        '"suction gripper" soft robotic',
        '"tissue handling" robotic gripper',
    ],

    "thermal_ultrasonic_energy_transfer": [
        '"ultrasonic cutting" energy device',
        '"ultrasonic welding" polymer tissue',
        '"radiofrequency ablation" thermal model',
        '"microwave ablation" thermal model',
        '"thermal spread" radiofrequency tissue',
        '"plasma treatment" surface sealing',
        '"laser tissue welding" material',
    ],

    "imaging_tracking_navigation_3d": [
        '"electromagnetic tracking" navigation',
        '"optical tracking" deformable object',
        '"3D reconstruction" endoscopic image',
        '"real-time tracking" deformable surface',
        '"sensor fusion" navigation tracking',
        '"augmented reality" image guided intervention',
        '"hyperspectral imaging" tissue perfusion',
    ],

    "materials_porosity_transport_filtration": [
        '"porous material" fluid transport',
        '"membrane filtration" fouling control',
        '"hydrogel" diffusion barrier',
        '"selective permeability" hydrogel membrane',
        '"wicking material" fluid management',
        '"porous scaffold" mechanical fluid',
        '"capillary pump" microfluidic',
    ],

    "mechanical_fastening_closure_compression": [
        '"mechanical closure" soft tissue',
        '"compression device" tissue approximation',
        '"deployable clip" mechanical closure',
        '"sutureless closure" adhesive mechanical',
        '"staple formation" mechanical model',
        '"constant force" compression device',
        '"clamping force" soft tissue',
    ],
}

# Ausschlussbegriffe im Titel/Abstract, damit wir nicht in Aerospace/Combat/irrelevante Felder abdriften.
NEGATIVE_TERMS = [
    "fighter jet", "missile", "weapon", "combustion engine", "gas turbine",
    "aerospace vehicle", "aircraft wing", "ship hull", "naval", "ballistic",
    "satellite orbit", "rocket", "warfare", "military aircraft",
]

# ============================================================
# HELPERS
# ============================================================

def reconstruct_abstract(inv_idx):
    """
    OpenAlex stores abstracts as inverted index:
    {"word": [positions...]}
    Reconstruct into plain text.
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


def get_concepts(work, max_n=12):
    concepts = work.get("concepts") or []
    out = []
    for c in concepts[:max_n]:
        name = c.get("display_name")
        score = c.get("score")
        if name:
            out.append(f"{name} ({score:.2f})" if isinstance(score, (int, float)) else name)
    return "; ".join(out)


def get_authors(work, max_n=12):
    authorships = work.get("authorships") or []
    names = []
    for a in authorships[:max_n]:
        au = a.get("author") or {}
        name = au.get("display_name")
        if name:
            names.append(name)
    return "; ".join(names)


def is_negative_topic(title, abstract):
    txt = f"{title} {abstract}".lower()
    return any(term.lower() in txt for term in NEGATIVE_TERMS)


def openalex_count(query: str) -> int:
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"from_publication_date:{FROM_DATE},to_publication_date:{TO_DATE},has_abstract:true",
        "search": query,
        "per-page": 1,
        "mailto": MAILTO,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return int(data.get("meta", {}).get("count", 0))


def fetch_openalex_query(module_name: str, query: str):
    """
    Cursor pagination for one search query.
    Test-limited by MAX_PAGES_PER_QUERY.
    """
    url = "https://api.openalex.org/works"
    cursor = "*"
    rows = []

    for page in range(1, MAX_PAGES_PER_QUERY + 1):
        params = {
            "filter": f"from_publication_date:{FROM_DATE},to_publication_date:{TO_DATE},has_abstract:true",
            "search": query,
            "per-page": PER_PAGE,
            "cursor": cursor,
            "mailto": MAILTO,
        }

        for attempt in range(1, 6):
            try:
                r = requests.get(url, params=params, timeout=90)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == 5:
                    raise
                wait = attempt * 2
                print(f"Retry {attempt}/5 after error: {e}. Waiting {wait}s...")
                time.sleep(wait)

        results = data.get("results") or []
        if not results:
            break

        for w in results:
            title = w.get("display_name") or ""
            abstract = reconstruct_abstract(w.get("abstract_inverted_index"))

            if not abstract:
                continue

            if is_negative_topic(title, abstract):
                continue

            rows.append({
                "openalex_id": w.get("id", ""),
                "doi": w.get("doi", ""),
                "year": w.get("publication_year", ""),
                "publication_date": w.get("publication_date", ""),
                "source": get_source_name(w),
                "source_type": get_source_type(w),
                "title": title,
                "abstract": abstract,
                "cited_by_count": w.get("cited_by_count", ""),
                "type": w.get("type", ""),
                "authors": get_authors(w),
                "concepts": get_concepts(w),
                "module": module_name,
                "query": query,
            })

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(SLEEP_SECONDS)

    return rows


def main():
    print("OpenAlex Engineering Transfer Test")
    print(f"Date range: {FROM_DATE} to {TO_DATE}")
    print(f"Max pages/query: {MAX_PAGES_PER_QUERY}")
    print(f"Max total records: {MAX_TOTAL_RECORDS}")

    all_rows = []
    seen_ids = set()

    # Save queries for transparency
    with OUTPUT_QUERIES.open("w", encoding="utf-8") as f:
        for module, queries in SEARCH_MODULES.items():
            f.write(f"\n### {module}\n")
            for q in queries:
                f.write(q + "\n")

    for module_name, queries in SEARCH_MODULES.items():
        print(f"\n=== MODULE: {module_name} ===")

        for query in queries:
            if MAX_TOTAL_RECORDS is not None and len(all_rows) >= MAX_TOTAL_RECORDS:
                break

            try:
                count = openalex_count(query)
            except Exception as e:
                print(f"Count failed for query: {query} | {e}")
                count = None

            print(f"Query: {query} | count={count}")

            # Guardrail: if query is too broad, skip in test run
            if count is not None and count > 50000:
                print("  SKIP: too broad for test run (>50,000)")
                continue

            try:
                rows = fetch_openalex_query(module_name, query)
            except Exception as e:
                print(f"  FETCH FAILED: {e}")
                continue

            new_n = 0
            for row in rows:
                oid = row["openalex_id"]
                if oid in seen_ids:
                    continue
                seen_ids.add(oid)
                all_rows.append(row)
                new_n += 1

                if MAX_TOTAL_RECORDS is not None and len(all_rows) >= MAX_TOTAL_RECORDS:
                    break

            print(f"  fetched={len(rows)}, new_unique={new_n}, total_unique={len(all_rows)}")

        if MAX_TOTAL_RECORDS is not None and len(all_rows) >= MAX_TOTAL_RECORDS:
            break

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("No records fetched.")
        return

    # Sort by year desc and citations desc for easier inspection
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce")
    df = df.sort_values(["year", "cited_by_count"], ascending=[False, False])

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print("\nDONE")
    print(f"Rows: {len(df):,}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Queries: {OUTPUT_QUERIES}")

    print("\nTop 20 records:")
    cols = ["year", "source", "title", "module", "cited_by_count"]
    print(df[cols].head(20).to_string(max_colwidth=120))


if __name__ == "__main__":
    main()