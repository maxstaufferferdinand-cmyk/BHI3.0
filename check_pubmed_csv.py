from pathlib import Path
import pandas as pd
import csv

csv_path = Path(r"C:\Users\Max Stauffer\BreakingRules\pubmed_visceral_surgery_technology_precise_1980_2025_v7_no_broad_duplicates.csv")

print("File exists:", csv_path.exists())
print("File size GB:", round(csv_path.stat().st_size / (1024**3), 2))

# 1) Erste 10 Zeilen laden
df_head = pd.read_csv(csv_path, nrows=10, encoding="utf-8", low_memory=False)

print("\nColumns:")
print(list(df_head.columns))

print("\nFirst 10 rows, selected columns:")
cols_to_show = [c for c in ["pmid", "year", "journal", "title", "abstract"] if c in df_head.columns]
print(df_head[cols_to_show].to_string(max_colwidth=250))

# 2) Spaltenzahl und grobe CSV-Struktur prüfen
with open(csv_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    print("\nNumber of columns in header:", len(header))
    print("Header:", header)

    bad_rows = 0
    checked = 0
    for i, row in enumerate(reader, start=2):
        checked += 1
        if len(row) != len(header):
            bad_rows += 1
            if bad_rows <= 5:
                print(f"Bad row {i}: {len(row)} columns instead of {len(header)}")
        if checked >= 10000:
            break

print(f"\nChecked first {checked} data rows.")
print("Malformed rows among first 10,000:", bad_rows)

# 3) Gesamtzahl Zeilen schnell zählen
with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
    total_lines = sum(1 for _ in f)

print("\nTotal lines including header:", total_lines)
print("Total data rows:", total_lines - 1)

# 4) Letzte 5 Zeilen prüfen, ob Datei sauber endet
print("\nLast 5 rows:")
tail = pd.read_csv(csv_path, encoding="utf-8", low_memory=False).tail(5)
print(tail[cols_to_show].to_string(max_colwidth=250))