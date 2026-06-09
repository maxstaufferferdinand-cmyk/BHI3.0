from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")
GRAPH_DIR = BASE_DIR / "surgical_training_graph_outputs"
OUT_DIR = BASE_DIR / "surgical_edge_strength_analysis"
OUT_DIR.mkdir(exist_ok=True)

EDGE_FILE = GRAPH_DIR / "surgical_graph_edges_aggregated_with_years.csv"

OUT_SUMMARY = OUT_DIR / "edge_strength_distribution_summary.txt"
OUT_TABLE_TOP = OUT_DIR / "top_strongest_edges.csv"
OUT_TABLE_CUTOFFS = OUT_DIR / "edge_strength_cutoff_counts.csv"
OUT_HIST_PNG = OUT_DIR / "edge_weight_distribution_hist.png"
OUT_HIST_LOG_PNG = OUT_DIR / "edge_weight_distribution_log_hist.png"
OUT_CUMULATIVE_PNG = OUT_DIR / "edge_weight_cumulative_distribution.png"
OUT_YEAR_WEIGHT_PNG = OUT_DIR / "edge_weight_by_first_year.png"


def main():
    print("Loading edges...")
    df = pd.read_csv(EDGE_FILE, encoding="utf-8-sig", low_memory=False)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    if "weight" not in df.columns:
        raise ValueError("No weight column found.")
    if "first_year" not in df.columns:
        raise ValueError("No first_year column found.")

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1).astype(int)
    df["first_year"] = pd.to_numeric(df["first_year"], errors="coerce")
    df["last_year"] = pd.to_numeric(df["last_year"], errors="coerce")

    df["duration_years"] = df["last_year"] - df["first_year"]
    df.loc[df["duration_years"] < 0, "duration_years"] = np.nan

    n_edges = len(df)
    n_with_year = df["first_year"].notna().sum()

    weights = df["weight"].values

    percentiles = [0, 25, 50, 75, 80, 85, 90, 95, 97.5, 99, 99.5, 99.9, 100]
    pct_values = np.percentile(weights, percentiles)

    cutoff_list = [1, 2, 3, 4, 5, 10, 20, 50, 100]
    cutoff_rows = []

    for c in cutoff_list:
        n_ge = int((df["weight"] >= c).sum())
        n_eq = int((df["weight"] == c).sum())
        cutoff_rows.append({
            "cutoff": c,
            "n_equal_cutoff": n_eq,
            "n_weight_ge_cutoff": n_ge,
            "fraction_weight_ge_cutoff": n_ge / n_edges if n_edges else np.nan,
        })

    cutoff_df = pd.DataFrame(cutoff_rows)
    cutoff_df.to_csv(OUT_TABLE_CUTOFFS, index=False, encoding="utf-8-sig")

    # Frequency table for small weights
    freq_table = (
        df["weight"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    freq_table.columns = ["weight", "n_edges"]
    freq_table["fraction"] = freq_table["n_edges"] / n_edges
    freq_table["cumulative_fraction"] = freq_table["n_edges"].cumsum() / n_edges

    # Top strongest edges
    top_cols = [
        "pair_id", "problem_id", "solution_id",
        "surgical_problem", "technical_solution",
        "weight", "first_year", "last_year", "duration_years", "pmids"
    ]
    top_cols = [c for c in top_cols if c in df.columns]

    top_df = df.sort_values(["weight", "duration_years"], ascending=[False, False]).head(200)
    top_df[top_cols].to_csv(OUT_TABLE_TOP, index=False, encoding="utf-8-sig")

    # Histogram raw
    plt.figure(figsize=(7, 5), dpi=300)
    plt.hist(df["weight"], bins=50)
    plt.xlabel("Edge weight")
    plt.ylabel("Number of problem-solution edges")
    plt.title("Distribution of surgical problem-solution edge weights")
    plt.tight_layout()
    plt.savefig(OUT_HIST_PNG, dpi=300)
    plt.close()

    # Histogram log x
    plt.figure(figsize=(7, 5), dpi=300)
    plt.hist(np.log1p(df["weight"]), bins=50)
    plt.xlabel("log(1 + edge weight)")
    plt.ylabel("Number of problem-solution edges")
    plt.title("Log-transformed edge weight distribution")
    plt.tight_layout()
    plt.savefig(OUT_HIST_LOG_PNG, dpi=300)
    plt.close()

    # Cumulative distribution
    sorted_w = np.sort(df["weight"].values)
    y = np.arange(1, len(sorted_w) + 1) / len(sorted_w)

    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(sorted_w, y, linewidth=2)
    plt.xscale("log")
    plt.xlabel("Edge weight, log scale")
    plt.ylabel("Cumulative fraction of edges")
    plt.title("Cumulative distribution of edge weights")
    plt.tight_layout()
    plt.savefig(OUT_CUMULATIVE_PNG, dpi=300)
    plt.close()

    # Weight by first year boxplot-ish scatter summary
    year_df = df.dropna(subset=["first_year"]).copy()
    year_df["first_year"] = year_df["first_year"].astype(int)

    if len(year_df) > 0:
        year_summary = (
            year_df.groupby("first_year")["weight"]
            .agg(["count", "median", "mean", "max"])
            .reset_index()
        )

        plt.figure(figsize=(8, 5), dpi=300)
        plt.plot(year_summary["first_year"], year_summary["median"], label="Median", linewidth=2)
        plt.plot(year_summary["first_year"], year_summary["mean"], label="Mean", linewidth=2)
        plt.xlabel("First year")
        plt.ylabel("Edge weight")
        plt.title("Edge weight by first year")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(OUT_YEAR_WEIGHT_PNG, dpi=300)
        plt.close()
    else:
        year_summary = pd.DataFrame()

    summary = []
    summary.append("Surgical edge strength distribution")
    summary.append("=" * 80)
    summary.append("")
    summary.append(f"Input file: {EDGE_FILE}")
    summary.append(f"Total edges: {n_edges:,}")
    summary.append(f"Edges with first_year: {n_with_year:,}")
    summary.append("")
    summary.append("Weight summary:")
    summary.append(str(df["weight"].describe()))
    summary.append("")
    summary.append("Weight percentiles:")
    for p, v in zip(percentiles, pct_values):
        summary.append(f"{p:>5}%: {v:.2f}")
    summary.append("")
    summary.append("Counts by proposed cutoffs:")
    summary.append(cutoff_df.to_string(index=False))
    summary.append("")
    summary.append("Small-weight frequency table, first 30 rows:")
    summary.append(freq_table.head(30).to_string(index=False))
    summary.append("")
    summary.append("Top 30 strongest edges:")
    summary.append(top_df[top_cols].head(30).to_string(index=False))
    summary.append("")
    summary.append("Output files:")
    for p in [
        OUT_SUMMARY,
        OUT_TABLE_TOP,
        OUT_TABLE_CUTOFFS,
        OUT_HIST_PNG,
        OUT_HIST_LOG_PNG,
        OUT_CUMULATIVE_PNG,
        OUT_YEAR_WEIGHT_PNG,
    ]:
        summary.append(str(p))

    OUT_SUMMARY.write_text("\n".join(summary), encoding="utf-8")

    print("\nSaved:")
    print(OUT_SUMMARY)
    print(OUT_TABLE_TOP)
    print(OUT_TABLE_CUTOFFS)
    print(OUT_HIST_PNG)
    print(OUT_HIST_LOG_PNG)
    print(OUT_CUMULATIVE_PNG)
    print(OUT_YEAR_WEIGHT_PNG)

    print("\nPreview cutoff table:")
    print(cutoff_df.to_string(index=False))

    print("\nDONE")


if __name__ == "__main__":
    main()