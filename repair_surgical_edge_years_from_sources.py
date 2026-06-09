from pathlib import Path
import re
import pandas as pd
import numpy as np


BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")
GRAPH_DIR = BASE_DIR / "surgical_training_graph_outputs"

EDGE_FILE = GRAPH_DIR / "surgical_graph_edges_aggregated.csv"
OUT_FILE = GRAPH_DIR / "surgical_graph_edges_aggregated_with_years.csv"
CANDIDATE_REPORT = GRAPH_DIR / "year_mapping_candidate_files.txt"

PMID_COL_CANDIDATES = [
    "pmid", "PMID", "pubmed_id", "PubMedID", "id", "article_id"
]

YEAR_COL_CANDIDATES = [
    "year", "publication_year", "pub_year", "pubdate_year",
    "publication_date", "date", "pub_date", "PublicationYear"
]


def parse_year(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    m = re.search(r"(19[0-9]{2}|20[0-9]{2})", s)
    if not m:
        return np.nan
    y = int(m.group(1))
    if 1900 <= y <= 2030:
        return y
    return np.nan


def parse_pmids(x):
    if pd.isna(x):
        return []
    s = str(x)
    return re.findall(r"\d{5,9}", s)


def find_column(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def scan_csv_for_pmid_year(path):
    try:
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    except Exception:
        try:
            header = pd.read_csv(path, nrows=0, encoding="latin1")
        except Exception:
            return None

    cols = list(header.columns)
    pmid_col = find_column(cols, PMID_COL_CANDIDATES)
    year_col = find_column(cols, YEAR_COL_CANDIDATES)

    if pmid_col is None or year_col is None:
        return None

    return pmid_col, year_col


def main():
    print("Loading edge file...")
    edges = pd.read_csv(EDGE_FILE, encoding="utf-8-sig", low_memory=False)
    print("Edges:", edges.shape)

    if "pmids" not in edges.columns:
        raise ValueError("Edge file has no 'pmids' column.")

    print("Scanning CSV files for PMID-year mappings...")

    csv_files = list(BASE_DIR.rglob("*.csv"))
    pmid_to_year = {}
    report_lines = []

    for path in csv_files:
        # Skip the edge files themselves and output directories likely not useful
        if path.name in [EDGE_FILE.name, OUT_FILE.name]:
            continue
        if "surgical_compatibility_model_outputs" in str(path):
            continue
        if "surgical_link_prediction_outputs" in str(path):
            continue

        found = scan_csv_for_pmid_year(path)
        if found is None:
            continue

        pmid_col, year_col = found
        report_lines.append(f"{path} | pmid_col={pmid_col} | year_col={year_col}")

        print(f"Reading candidate: {path.name} | pmid={pmid_col} | year={year_col}")

        try:
            df = pd.read_csv(
                path,
                usecols=[pmid_col, year_col],
                encoding="utf-8-sig",
                low_memory=False,
            )
        except Exception:
            try:
                df = pd.read_csv(
                    path,
                    usecols=[pmid_col, year_col],
                    encoding="latin1",
                    low_memory=False,
                )
            except Exception as e:
                print(f"  skipped due to read error: {e}")
                continue

        df = df.dropna(subset=[pmid_col, year_col]).copy()
        df[pmid_col] = df[pmid_col].astype(str).str.extract(r"(\d{5,9})", expand=False)
        df["year_num"] = df[year_col].apply(parse_year)
        df = df.dropna(subset=[pmid_col, "year_num"]).copy()

        before = len(pmid_to_year)
        for row in df.itertuples(index=False):
            pmid = str(getattr(row, pmid_col))
            year = int(getattr(row, "year_num"))
            if pmid not in pmid_to_year:
                pmid_to_year[pmid] = year
            else:
                # keep earliest plausible year if duplicates disagree
                pmid_to_year[pmid] = min(pmid_to_year[pmid], year)

        added = len(pmid_to_year) - before
        print(f"  mapped PMIDs added: {added:,} | total mapping: {len(pmid_to_year):,}")

    CANDIDATE_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nTotal PMID-year mappings:", f"{len(pmid_to_year):,}")

    if len(pmid_to_year) == 0:
        print("No PMID-year mapping found. Need to inspect source file names/columns manually.")
        return

    first_years = []
    last_years = []
    mapped_counts = []

    for pmid_str in edges["pmids"]:
        pmids = parse_pmids(pmid_str)
        years = [pmid_to_year[p] for p in pmids if p in pmid_to_year]

        if years:
            first_years.append(int(min(years)))
            last_years.append(int(max(years)))
            mapped_counts.append(len(years))
        else:
            first_years.append(np.nan)
            last_years.append(np.nan)
            mapped_counts.append(0)

    edges["first_year"] = first_years
    edges["last_year"] = last_years
    edges["n_pmids_with_year"] = mapped_counts

    n_with_year = edges["first_year"].notna().sum()
    print(f"Edges with recovered year: {n_with_year:,}/{len(edges):,}")
    print("Year distribution:")
    print(edges["first_year"].dropna().astype(int).describe())

    edges.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    print("\nSaved repaired edge file:")
    print(OUT_FILE)
    print("\nCandidate report:")
    print(CANDIDATE_REPORT)


if __name__ == "__main__":
    main()