from pathlib import Path
import os
import re
import json
import time
import numpy as np
import pandas as pd
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

INPUT_CSV = BASE_DIR / "llm_extraction_full_results.csv"

OUT_TABLE = BASE_DIR / "surgical_embedding_table_pilot_50.csv"
OUT_PROBLEM_EMB = BASE_DIR / "surgical_problem_embeddings_pilot_50.npy"
OUT_SOLUTION_EMB = BASE_DIR / "surgical_solution_embeddings_pilot_50.npy"
OUT_MECHANISM_VIEW_EMB = BASE_DIR / "surgical_mechanism_view_embeddings_pilot_50.npy"
OUT_PAIR_EMB = BASE_DIR / "surgical_pair_embeddings_pilot_50.npy"
OUT_NEIGHBORS = BASE_DIR / "surgical_embedding_neighbors_pilot_50.csv"
OUT_SUMMARY = BASE_DIR / "surgical_embedding_pilot_50_summary.txt"

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

N_PILOT = 50
RANDOM_SAMPLE = False
RANDOM_STATE = 42

# Falls deine Spalten anders heißen, unten in find_column() ergänzen.
POSSIBLE_DECISION_COLS = ["decision", "llm_decision", "include_decision"]
POSSIBLE_PROBLEM_COLS = ["surgical_problem", "manual_surgical_problem", "problem"]
POSSIBLE_SOLUTION_COLS = ["technical_solution", "manual_technical_solution", "manual_techniqual_solution", "solution"]
POSSIBLE_PMID_COLS = ["pmid", "PMID"]
POSSIBLE_YEAR_COLS = ["year", "publication_year"]
POSSIBLE_TITLE_COLS = ["title"]
POSSIBLE_ABSTRACT_COLS = ["abstract"]

# Wie viele Nachbarn pro Beispiel in der QC-Tabelle
N_NEIGHBORS = 5

# ============================================================
# CLIENT
# ============================================================

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. In PowerShell run:\n"
        '$env:OPENAI_API_KEY="sk-..."'
    )

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ============================================================
# HELPERS
# ============================================================

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_concept(x: str) -> str:
    x = safe_str(x)
    x = x.replace("\n", " ").replace("\r", " ")
    x = re.sub(r"\s+", " ", x).strip()
    x = x.strip(" ;,.-")
    return x


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def normalize_decision(x: str) -> str:
    x = safe_str(x).lower()
    x = x.strip()
    if x in ["include", "included", "yes", "1", "true"]:
        return "include"
    if "include" in x and "reject" not in x:
        return "include"
    return "reject"


def make_problem_text(problem: str) -> str:
    return f"Surgical problem: {problem}."


def make_solution_text(solution: str) -> str:
    return f"Technical or procedural solution in visceral surgery: {solution}."


def make_mechanism_view_text(solution: str) -> str:
    return (
        "Underlying technical mechanism or operative principle of the solution: "
        f"{solution}. Focus on device, material, imaging, access, energy, drainage, "
        "pressure, flow, robotic, endoscopic, or reconstruction mechanism."
    )


def make_pair_text(problem: str, solution: str) -> str:
    return f"Surgical problem: {problem}. Technical solution: {solution}."


def embed_texts(texts: list[str], label: str, batch_size: int = 100) -> np.ndarray:
    all_embeddings = []

    print(f"Embedding {label}: {len(texts)} texts using {EMBEDDING_MODEL}")

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]

        for attempt in range(1, 6):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                emb = [item.embedding for item in response.data]
                all_embeddings.extend(emb)
                print(f"  {label}: embedded {end}/{len(texts)}")
                break

            except Exception as e:
                print(f"  {label}: attempt {attempt}/5 failed: {e}")
                if attempt == 5:
                    raise
                time.sleep(2 * attempt)

    return np.array(all_embeddings, dtype=np.float32)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom[denom == 0] = 1
    return mat / denom


def cosine_neighbors(emb: np.ndarray, labels: list[str], view_name: str, n_neighbors: int = 5) -> pd.DataFrame:
    emb_norm = l2_normalize(emb)
    sim = emb_norm @ emb_norm.T

    rows = []
    n = sim.shape[0]

    for i in range(n):
        order = np.argsort(-sim[i])
        order = [j for j in order if j != i][:n_neighbors]

        for rank, j in enumerate(order, start=1):
            rows.append({
                "view": view_name,
                "query_index": i,
                "query_label": labels[i],
                "neighbor_rank": rank,
                "neighbor_index": j,
                "neighbor_label": labels[j],
                "cosine_similarity": float(sim[i, j]),
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    print("============================================================")
    print("Surgical concept embedding pilot 50")
    print("============================================================")
    print("Input:", INPUT_CSV)
    print("Embedding model:", EMBEDDING_MODEL)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    print("Rows in input:", len(df))
    print("Columns:", list(df.columns))

    decision_col = find_column(df, POSSIBLE_DECISION_COLS, required=True)
    problem_col = find_column(df, POSSIBLE_PROBLEM_COLS, required=True)
    solution_col = find_column(df, POSSIBLE_SOLUTION_COLS, required=True)

    pmid_col = find_column(df, POSSIBLE_PMID_COLS, required=False)
    year_col = find_column(df, POSSIBLE_YEAR_COLS, required=False)
    title_col = find_column(df, POSSIBLE_TITLE_COLS, required=False)
    abstract_col = find_column(df, POSSIBLE_ABSTRACT_COLS, required=False)

    print("Detected columns:")
    print("  decision:", decision_col)
    print("  problem:", problem_col)
    print("  solution:", solution_col)
    print("  pmid:", pmid_col)
    print("  year:", year_col)
    print("  title:", title_col)
    print("  abstract:", abstract_col)

    df["_decision_norm"] = df[decision_col].apply(normalize_decision)
    df["_problem"] = df[problem_col].apply(clean_concept)
    df["_solution"] = df[solution_col].apply(clean_concept)

    inc = df[
        (df["_decision_norm"] == "include") &
        (df["_problem"].str.len() > 2) &
        (df["_solution"].str.len() > 2)
    ].copy()

    print("Included usable rows:", len(inc))

    if len(inc) < N_PILOT:
        print(f"WARNING: only {len(inc)} usable included rows available; using all.")
        pilot = inc.copy()
    else:
        if RANDOM_SAMPLE:
            pilot = inc.sample(n=N_PILOT, random_state=RANDOM_STATE).copy()
        else:
            pilot = inc.head(N_PILOT).copy()

    pilot = pilot.reset_index(drop=True)
    pilot["pilot_index"] = range(len(pilot))

    # Output metadata table
    out = pd.DataFrame({
        "pilot_index": pilot["pilot_index"],
        "pmid": pilot[pmid_col] if pmid_col else "",
        "year": pilot[year_col] if year_col else "",
        "title": pilot[title_col] if title_col else "",
        "surgical_problem": pilot["_problem"],
        "technical_solution": pilot["_solution"],
    })

    out["problem_embedding_text"] = out["surgical_problem"].apply(make_problem_text)
    out["solution_embedding_text"] = out["technical_solution"].apply(make_solution_text)
    out["mechanism_view_embedding_text"] = out["technical_solution"].apply(make_mechanism_view_text)
    out["pair_embedding_text"] = [
        make_pair_text(p, s)
        for p, s in zip(out["surgical_problem"], out["technical_solution"])
    ]

    # Simple deterministic IDs for later graph use
    out["problem_id"] = ["P_" + str(i).zfill(5) for i in range(len(out))]
    out["solution_id"] = ["S_" + str(i).zfill(5) for i in range(len(out))]
    out["pair_id"] = ["E_" + str(i).zfill(5) for i in range(len(out))]

    # Save table before embedding too
    out.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
    print("Saved table:", OUT_TABLE)

    # Embeddings
    problem_emb = embed_texts(out["problem_embedding_text"].tolist(), "problem")
    solution_emb = embed_texts(out["solution_embedding_text"].tolist(), "solution")
    mechanism_view_emb = embed_texts(out["mechanism_view_embedding_text"].tolist(), "mechanism_view")
    pair_emb = embed_texts(out["pair_embedding_text"].tolist(), "pair")

    np.save(OUT_PROBLEM_EMB, problem_emb)
    np.save(OUT_SOLUTION_EMB, solution_emb)
    np.save(OUT_MECHANISM_VIEW_EMB, mechanism_view_emb)
    np.save(OUT_PAIR_EMB, pair_emb)

    print("Saved embeddings:")
    print(" ", OUT_PROBLEM_EMB, problem_emb.shape)
    print(" ", OUT_SOLUTION_EMB, solution_emb.shape)
    print(" ", OUT_MECHANISM_VIEW_EMB, mechanism_view_emb.shape)
    print(" ", OUT_PAIR_EMB, pair_emb.shape)

    # Neighbor QC
    problem_neighbors = cosine_neighbors(
        problem_emb,
        out["surgical_problem"].tolist(),
        "problem_embedding",
        n_neighbors=N_NEIGHBORS,
    )

    solution_neighbors = cosine_neighbors(
        solution_emb,
        out["technical_solution"].tolist(),
        "solution_embedding",
        n_neighbors=N_NEIGHBORS,
    )

    mechanism_neighbors = cosine_neighbors(
        mechanism_view_emb,
        out["technical_solution"].tolist(),
        "mechanism_view_embedding",
        n_neighbors=N_NEIGHBORS,
    )

    pair_labels = [
        f"{p} -> {s}"
        for p, s in zip(out["surgical_problem"], out["technical_solution"])
    ]
    pair_neighbors = cosine_neighbors(
        pair_emb,
        pair_labels,
        "pair_embedding",
        n_neighbors=N_NEIGHBORS,
    )

    neighbors = pd.concat(
        [problem_neighbors, solution_neighbors, mechanism_neighbors, pair_neighbors],
        ignore_index=True,
    )

    neighbors.to_csv(OUT_NEIGHBORS, index=False, encoding="utf-8-sig")
    print("Saved neighbors:", OUT_NEIGHBORS)

    # Summary
    summary = []
    summary.append("Surgical concept embedding pilot 50")
    summary.append("=" * 60)
    summary.append(f"Input: {INPUT_CSV}")
    summary.append(f"Rows in input: {len(df):,}")
    summary.append(f"Usable included rows: {len(inc):,}")
    summary.append(f"Pilot rows: {len(out):,}")
    summary.append(f"Embedding model: {EMBEDDING_MODEL}")
    summary.append("")
    summary.append("Embedding shapes:")
    summary.append(f"problem: {problem_emb.shape}")
    summary.append(f"solution: {solution_emb.shape}")
    summary.append(f"mechanism_view: {mechanism_view_emb.shape}")
    summary.append(f"pair: {pair_emb.shape}")
    summary.append("")
    summary.append("Files:")
    summary.append(str(OUT_TABLE))
    summary.append(str(OUT_PROBLEM_EMB))
    summary.append(str(OUT_SOLUTION_EMB))
    summary.append(str(OUT_MECHANISM_VIEW_EMB))
    summary.append(str(OUT_PAIR_EMB))
    summary.append(str(OUT_NEIGHBORS))

    OUT_SUMMARY.write_text("\n".join(summary), encoding="utf-8")
    print("Saved summary:", OUT_SUMMARY)

    print("============================================================")
    print("DONE")
    print("Next: open surgical_embedding_neighbors_pilot_50.csv")
    print("============================================================")


if __name__ == "__main__":
    main()