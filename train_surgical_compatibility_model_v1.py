from pathlib import Path
from collections import defaultdict
import json
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")
GRAPH_DIR = BASE_DIR / "surgical_training_graph_outputs"
OUT_DIR = BASE_DIR / "surgical_compatibility_model_outputs"
OUT_DIR.mkdir(exist_ok=True)

EDGES_CSV = GRAPH_DIR / "surgical_graph_edges_aggregated_with_years.csv"

PROBLEM_INDEX_CSV = GRAPH_DIR / "surgical_problem_embedding_index.csv"
SOLUTION_INDEX_CSV = GRAPH_DIR / "surgical_solution_embedding_index.csv"

PROBLEM_EMB_NPY = GRAPH_DIR / "surgical_problem_embeddings.npy"
SOLUTION_EMB_NPY = GRAPH_DIR / "surgical_solution_embeddings.npy"

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
rng = np.random.default_rng(RANDOM_STATE)

# Main historical split.
# Model sees historical graph up to CUTOFF_YEAR.
# Positive test edges are problem-solution pairs first appearing after cutoff.
CUTOFF_YEAR = 2015
FUTURE_START_YEAR = 2016
FUTURE_END_YEAR = 2022

# Training positives are historical edges up to cutoff.
# We sample negatives among pairs never observed in full corpus.
MAX_TRAIN_POSITIVES = 80000
MAX_TEST_POSITIVES = 30000

NEG_PER_POS_TRAIN = 4
NEG_PER_POS_TEST = 20

# For calculating candidate similarity to known solution-neighbourhood of a problem.
TOPK_NEIGHBOUR_SOLUTIONS = 5

OUT_METRICS = OUT_DIR / "compatibility_model_metrics.csv"
OUT_SUMMARY = OUT_DIR / "compatibility_model_summary.txt"
OUT_ROC = OUT_DIR / "compatibility_model_roc.png"
OUT_PR = OUT_DIR / "compatibility_model_precision_recall.png"
OUT_MODEL = OUT_DIR / "surgical_compatibility_model_v1.joblib"
OUT_FEATURES_JSON = OUT_DIR / "compatibility_model_feature_names.json"
OUT_TEST_PREDICTIONS = OUT_DIR / "compatibility_model_test_predictions.csv"


# ============================================================
# HELPERS
# ============================================================

def log1p_safe(x):
    try:
        return math.log1p(max(float(x), 0.0))
    except Exception:
        return 0.0


def parse_year(x):
    if pd.isna(x):
        return np.nan
    try:
        return int(float(str(x).strip()))
    except Exception:
        return np.nan


def normalize_matrix(x):
    x = x.astype(np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


def weighted_average(vectors, weights):
    if len(vectors) == 0:
        return None
    weights = np.asarray(weights, dtype=np.float32)
    weights = np.maximum(weights, 1e-6)
    weights = weights / weights.sum()
    return np.average(np.vstack(vectors), axis=0, weights=weights).astype(np.float32)


def cosine(a, b):
    return float(np.dot(a, b))


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def detect_weight_column(edges):
    candidates = ["weight", "n_pmids", "pmid_count", "count", "n", "frequency"]
    for c in candidates:
        if c in edges.columns:
            return c
    return None


def load_inputs():
    print("Loading files...")

    for p in [EDGES_CSV, PROBLEM_INDEX_CSV, SOLUTION_INDEX_CSV, PROBLEM_EMB_NPY, SOLUTION_EMB_NPY]:
        require_file(p)

    edges = pd.read_csv(EDGES_CSV, encoding="utf-8-sig", low_memory=False)
    problem_index = pd.read_csv(PROBLEM_INDEX_CSV, encoding="utf-8-sig", low_memory=False)
    solution_index = pd.read_csv(SOLUTION_INDEX_CSV, encoding="utf-8-sig", low_memory=False)

    problem_emb = normalize_matrix(np.load(PROBLEM_EMB_NPY))
    solution_emb = normalize_matrix(np.load(SOLUTION_EMB_NPY))

    print("edges:", edges.shape)
    print("problem_index:", problem_index.shape)
    print("solution_index:", solution_index.shape)
    print("problem_emb:", problem_emb.shape)
    print("solution_emb:", solution_emb.shape)

    needed_cols = {"problem_id", "solution_id"}
    missing = needed_cols - set(edges.columns)
    if missing:
        raise ValueError(f"Edges file is missing required columns: {missing}")

    if "first_year" not in edges.columns:
        raise ValueError("Edges file needs a first_year column for temporal validation.")

    weight_col = detect_weight_column(edges)
    if weight_col is None:
        print("No weight-like column found. Using weight = 1 for every edge.")
        edges["edge_weight_used"] = 1.0
    else:
        print(f"Using edge weight column: {weight_col}")
        edges["edge_weight_used"] = pd.to_numeric(edges[weight_col], errors="coerce").fillna(1.0)

    edges["first_year_num"] = edges["first_year"].apply(parse_year)
    edges = edges[
        edges["problem_id"].notna()
        & edges["solution_id"].notna()
        & edges["first_year_num"].notna()
    ].copy()

    edges["problem_id"] = edges["problem_id"].astype(str)
    edges["solution_id"] = edges["solution_id"].astype(str)
    edges["first_year_num"] = edges["first_year_num"].astype(int)
    edges["edge_weight_used"] = edges["edge_weight_used"].astype(float)

    # Keep only edges for which embeddings exist.
    problem_id_to_idx = {str(x): i for i, x in enumerate(problem_index["problem_id"].astype(str))}
    solution_id_to_idx = {str(x): i for i, x in enumerate(solution_index["solution_id"].astype(str))}

    before = len(edges)
    edges = edges[
        edges["problem_id"].isin(problem_id_to_idx)
        & edges["solution_id"].isin(solution_id_to_idx)
    ].copy()
    print(f"Edges with available embeddings: {len(edges):,}/{before:,}")

    return edges, problem_index, solution_index, problem_emb, solution_emb, problem_id_to_idx, solution_id_to_idx


def build_past_graph(edges, cutoff_year):
    past = edges[edges["first_year_num"] <= cutoff_year].copy()

    p_to_s = defaultdict(set)
    s_to_p = defaultdict(set)

    p_to_s_weight = defaultdict(dict)
    s_to_p_weight = defaultdict(dict)

    p_total_weight = defaultdict(float)
    s_total_weight = defaultdict(float)

    for row in past.itertuples(index=False):
        p = row.problem_id
        s = row.solution_id
        w = float(row.edge_weight_used)

        p_to_s[p].add(s)
        s_to_p[s].add(p)

        p_to_s_weight[p][s] = p_to_s_weight[p].get(s, 0.0) + w
        s_to_p_weight[s][p] = s_to_p_weight[s].get(p, 0.0) + w

        p_total_weight[p] += w
        s_total_weight[s] += w

    past_pairs = set(zip(past["problem_id"], past["solution_id"]))

    graph = {
        "past_edges": past,
        "p_to_s": p_to_s,
        "s_to_p": s_to_p,
        "p_to_s_weight": p_to_s_weight,
        "s_to_p_weight": s_to_p_weight,
        "p_total_weight": p_total_weight,
        "s_total_weight": s_total_weight,
        "past_pairs": past_pairs,
        "active_problems": set(p_to_s.keys()),
        "active_solutions": set(s_to_p.keys()),
    }

    return graph


def compute_problem_solution_centroids(graph, solution_emb, solution_id_to_idx):
    """
    For each surgical problem, calculate the weighted centroid of its historical solution embeddings.
    This is the learned historical solution neighbourhood of the problem.
    """
    centroids = {}

    for p, solutions in graph["p_to_s"].items():
        vectors = []
        weights = []

        for s in solutions:
            if s not in solution_id_to_idx:
                continue
            vectors.append(solution_emb[solution_id_to_idx[s]])
            weights.append(log1p_safe(graph["p_to_s_weight"][p].get(s, 1.0)))

        centroids[p] = weighted_average(vectors, weights)

    return centroids


def get_future_positive_pairs(edges, graph, start_year, end_year, max_pos=None):
    future = edges[
        (edges["first_year_num"] >= start_year)
        & (edges["first_year_num"] <= end_year)
    ].copy()

    # Problem must already exist before cutoff.
    # Candidate solution may be old or new; this is important for cold-start-like behaviour.
    future = future[future["problem_id"].isin(graph["active_problems"])].copy()

    # Edge must not have existed before cutoff.
    future["_pair"] = list(zip(future["problem_id"], future["solution_id"]))
    future = future[~future["_pair"].isin(graph["past_pairs"])].copy()

    future = future.drop_duplicates(subset=["problem_id", "solution_id"]).copy()

    if max_pos is not None and len(future) > max_pos:
        future = future.sample(n=max_pos, random_state=RANDOM_STATE).copy()

    return list(zip(future["problem_id"], future["solution_id"]))


def sample_negatives_for_problems(
    problems,
    candidate_solutions,
    all_observed_pairs,
    n_neg,
):
    negatives = set()
    problems = list(problems)
    candidate_solutions = list(candidate_solutions)

    attempts = 0
    max_attempts = n_neg * 200

    while len(negatives) < n_neg and attempts < max_attempts:
        attempts += 1
        p = problems[rng.integers(0, len(problems))]
        s = candidate_solutions[rng.integers(0, len(candidate_solutions))]

        if (p, s) in all_observed_pairs:
            continue

        negatives.add((p, s))

    if len(negatives) < n_neg:
        print(f"WARNING: sampled only {len(negatives):,}/{n_neg:,} negatives")

    return list(negatives)


def build_problem_context(graph):
    """
    Problem context:
    other problems that share at least one historical solution with p.
    Useful for graph-neighbourhood support.
    """
    context = {}
    for p, sols in graph["p_to_s"].items():
        ctx = set()
        for s in sols:
            ctx.update(graph["s_to_p"].get(s, set()))
        ctx.discard(p)
        context[p] = ctx
    return context


def compute_features(
    pairs,
    labels,
    graph,
    problem_context,
    problem_centroids,
    problem_emb,
    solution_emb,
    problem_id_to_idx,
    solution_id_to_idx,
):
    feature_names = [
        "problem_degree_log",
        "problem_total_weight_log",
        "solution_seen_before",
        "solution_degree_log",
        "solution_total_weight_log",
        "problem_solution_cosine",
        "candidate_to_problem_solution_centroid_cosine",
        "candidate_to_known_solution_max_cosine",
        "candidate_to_known_solution_topk_mean_cosine",
        "candidate_to_known_solution_weighted_mean_cosine",
        "three_hop_problem_solution_support_log",
        "three_hop_support_exists",
        "context_jaccard",
    ]

    rows = []
    kept_pairs = []
    kept_labels = []

    for (p, s), y in zip(pairs, labels):
        if p not in problem_id_to_idx or s not in solution_id_to_idx:
            continue

        p_idx = problem_id_to_idx[p]
        s_idx = solution_id_to_idx[s]

        p_vec = problem_emb[p_idx]
        s_vec = solution_emb[s_idx]

        p_solutions = graph["p_to_s"].get(p, set())

        p_deg = len(p_solutions)
        p_weight = graph["p_total_weight"].get(p, 0.0)

        s_seen = 1 if s in graph["active_solutions"] else 0
        s_deg = len(graph["s_to_p"].get(s, set()))
        s_weight = graph["s_total_weight"].get(s, 0.0)

        ps_cos = cosine(p_vec, s_vec)

        centroid = problem_centroids.get(p)
        if centroid is None:
            centroid_cos = 0.0
        else:
            centroid_cos = cosine(s_vec, centroid)

        # Candidate similarity to the historical known solutions of p.
        known_solution_sims = []
        known_solution_weights = []

        for old_s in p_solutions:
            if old_s not in solution_id_to_idx:
                continue
            old_vec = solution_emb[solution_id_to_idx[old_s]]
            sim = cosine(s_vec, old_vec)
            known_solution_sims.append(sim)
            known_solution_weights.append(log1p_safe(graph["p_to_s_weight"][p].get(old_s, 1.0)))

        if len(known_solution_sims) == 0:
            max_known_sim = 0.0
            topk_mean_sim = 0.0
            weighted_mean_sim = 0.0
        else:
            sims_arr = np.asarray(known_solution_sims, dtype=np.float32)
            weights_arr = np.asarray(known_solution_weights, dtype=np.float32)
            weights_arr = np.maximum(weights_arr, 1e-6)

            max_known_sim = float(np.max(sims_arr))
            k = min(TOPK_NEIGHBOUR_SOLUTIONS, len(sims_arr))
            topk_mean_sim = float(np.mean(np.sort(sims_arr)[-k:]))
            weighted_mean_sim = float(np.average(sims_arr, weights=weights_arr))

        # 3-hop graph support:
        # p -> historical solution -> related problem -> candidate solution
        p_ctx = problem_context.get(p, set())
        s_problems = graph["s_to_p"].get(s, set())

        if len(p_ctx) < len(s_problems):
            three_hop = len(p_ctx.intersection(s_problems))
        else:
            three_hop = len(s_problems.intersection(p_ctx))

        three_hop_exists = 1 if three_hop > 0 else 0

        union_size = len(p_ctx.union(s_problems)) if p_ctx or s_problems else 0
        context_jaccard = three_hop / union_size if union_size > 0 else 0.0

        rows.append([
            log1p_safe(p_deg),
            log1p_safe(p_weight),
            s_seen,
            log1p_safe(s_deg),
            log1p_safe(s_weight),
            ps_cos,
            centroid_cos,
            max_known_sim,
            topk_mean_sim,
            weighted_mean_sim,
            log1p_safe(three_hop),
            three_hop_exists,
            context_jaccard,
        ])

        kept_pairs.append((p, s))
        kept_labels.append(y)

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(kept_labels, dtype=np.int8)

    return X, y, kept_pairs, feature_names


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.035,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=RANDOM_STATE,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "val_roc_auc": float(roc_auc_score(y_val, val_prob)),
        "val_average_precision": float(average_precision_score(y_val, val_prob)),
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        "test_average_precision": float(average_precision_score(y_test, test_prob)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_train_pos": int(y_train.sum()),
        "n_val_pos": int(y_val.sum()),
        "n_test_pos": int(y_test.sum()),
    }

    return model, val_prob, test_prob, metrics


def plot_roc(y_test, test_prob):
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    auc = roc_auc_score(y_test, test_prob)

    plt.figure(figsize=(6.6, 5.8), dpi=300)
    plt.plot(fpr, tpr, linewidth=2, label=f"Compatibility model AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black", alpha=0.7, label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Surgical compatibility model ROC")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_ROC, dpi=300)
    plt.close()


def plot_pr(y_test, test_prob):
    precision, recall, _ = precision_recall_curve(y_test, test_prob)
    ap = average_precision_score(y_test, test_prob)

    plt.figure(figsize=(6.6, 5.8), dpi=300)
    plt.plot(recall, precision, linewidth=2, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Surgical compatibility model precision-recall")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PR, dpi=300)
    plt.close()


def main():
    edges, problem_index, solution_index, problem_emb, solution_emb, problem_id_to_idx, solution_id_to_idx = load_inputs()

    all_observed_pairs = set(zip(edges["problem_id"], edges["solution_id"]))

    print("\nBuilding past graph...")
    graph = build_past_graph(edges, CUTOFF_YEAR)
    problem_context = build_problem_context(graph)
    problem_centroids = compute_problem_solution_centroids(graph, solution_emb, solution_id_to_idx)

    print(f"Past graph cutoff <= {CUTOFF_YEAR}")
    print(f"Active problems: {len(graph['active_problems']):,}")
    print(f"Active solutions: {len(graph['active_solutions']):,}")
    print(f"Past edges: {len(graph['past_pairs']):,}")

    # ------------------------------------------------------------
    # Training positives: historical edges up to cutoff.
    # ------------------------------------------------------------

    train_pos_df = graph["past_edges"].drop_duplicates(subset=["problem_id", "solution_id"]).copy()

    # Keep only problems with at least one historical solution neighbourhood.
    train_pos_df = train_pos_df[train_pos_df["problem_id"].isin(graph["active_problems"])].copy()

    if MAX_TRAIN_POSITIVES is not None and len(train_pos_df) > MAX_TRAIN_POSITIVES:
        train_pos_df = train_pos_df.sample(n=MAX_TRAIN_POSITIVES, random_state=RANDOM_STATE).copy()

    train_pos_pairs = list(zip(train_pos_df["problem_id"], train_pos_df["solution_id"]))

    train_neg_pairs = sample_negatives_for_problems(
        problems=[p for p, s in train_pos_pairs],
        candidate_solutions=solution_id_to_idx.keys(),
        all_observed_pairs=all_observed_pairs,
        n_neg=len(train_pos_pairs) * NEG_PER_POS_TRAIN,
    )

    train_pairs_all = train_pos_pairs + train_neg_pairs
    y_train_all = np.asarray([1] * len(train_pos_pairs) + [0] * len(train_neg_pairs), dtype=np.int8)
    train_pairs_all, y_train_all = shuffle(train_pairs_all, y_train_all, random_state=RANDOM_STATE)

    # ------------------------------------------------------------
    # Future test positives: new problem-solution relations after cutoff.
    # Candidate solution can be previously unseen in the surgical graph.
    # ------------------------------------------------------------

    test_pos_pairs = get_future_positive_pairs(
        edges=edges,
        graph=graph,
        start_year=FUTURE_START_YEAR,
        end_year=FUTURE_END_YEAR,
        max_pos=MAX_TEST_POSITIVES,
    )

    test_neg_pairs = sample_negatives_for_problems(
        problems=[p for p, s in test_pos_pairs],
        candidate_solutions=solution_id_to_idx.keys(),
        all_observed_pairs=all_observed_pairs,
        n_neg=len(test_pos_pairs) * NEG_PER_POS_TEST,
    )

    test_pairs = test_pos_pairs + test_neg_pairs
    y_test = np.asarray([1] * len(test_pos_pairs) + [0] * len(test_neg_pairs), dtype=np.int8)
    test_pairs, y_test = shuffle(test_pairs, y_test, random_state=RANDOM_STATE)

    print("\nDataset sizes:")
    print(f"Train positives: {len(train_pos_pairs):,}")
    print(f"Train negatives: {len(train_neg_pairs):,}")
    print(f"Test positives, future edges: {len(test_pos_pairs):,}")
    print(f"Test negatives: {len(test_neg_pairs):,}")

    # ------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------

    print("\nComputing training features...")
    X_all, y_all, train_pairs_kept, feature_names = compute_features(
        pairs=train_pairs_all,
        labels=y_train_all,
        graph=graph,
        problem_context=problem_context,
        problem_centroids=problem_centroids,
        problem_emb=problem_emb,
        solution_emb=solution_emb,
        problem_id_to_idx=problem_id_to_idx,
        solution_id_to_idx=solution_id_to_idx,
    )

    print("Computing test features...")
    X_test, y_test, test_pairs_kept, _ = compute_features(
        pairs=test_pairs,
        labels=y_test,
        graph=graph,
        problem_context=problem_context,
        problem_centroids=problem_centroids,
        problem_emb=problem_emb,
        solution_emb=solution_emb,
        problem_id_to_idx=problem_id_to_idx,
        solution_id_to_idx=solution_id_to_idx,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    print("\nFeature matrix:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    # ------------------------------------------------------------
    # Train and validate
    # ------------------------------------------------------------

    model, val_prob, test_prob, metrics = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------

    joblib.dump(model, OUT_MODEL)
    OUT_FEATURES_JSON.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")

    plot_roc(y_test, test_prob)
    plot_pr(y_test, test_prob)

    test_pred_df = pd.DataFrame({
        "problem_id": [p for p, s in test_pairs_kept],
        "solution_id": [s for p, s in test_pairs_kept],
        "label": y_test,
        "internal_model_probability": test_prob,
    })

    # Do not interpret this as a public ranking. This is validation output only.
    test_pred_df.to_csv(OUT_TEST_PREDICTIONS, index=False, encoding="utf-8-sig")

    summary = []
    summary.append("Surgical compatibility model v1")
    summary.append("=" * 80)
    summary.append("")
    summary.append("Goal:")
    summary.append("Learn historical compatibility patterns between surgical problem nodes and technical solution nodes.")
    summary.append("No LLM plausibility scores are used.")
    summary.append("")
    summary.append(f"Cutoff year: <= {CUTOFF_YEAR}")
    summary.append(f"Future validation window: {FUTURE_START_YEAR}-{FUTURE_END_YEAR}")
    summary.append("")
    summary.append(f"Active historical problems: {len(graph['active_problems']):,}")
    summary.append(f"Active historical solutions: {len(graph['active_solutions']):,}")
    summary.append(f"Past observed pairs: {len(graph['past_pairs']):,}")
    summary.append("")
    summary.append(f"Train positives: {len(train_pos_pairs):,}")
    summary.append(f"Train negatives: {len(train_neg_pairs):,}")
    summary.append(f"Test positives: {len(test_pos_pairs):,}")
    summary.append(f"Test negatives: {len(test_neg_pairs):,}")
    summary.append("")
    for k, v in metrics.items():
        summary.append(f"{k}: {v}")
    summary.append("")
    summary.append("Features:")
    for f in feature_names:
        summary.append(f"- {f}")
    summary.append("")
    summary.append("Output files:")
    for p in [OUT_MODEL, OUT_METRICS, OUT_ROC, OUT_PR, OUT_TEST_PREDICTIONS, OUT_FEATURES_JSON]:
        summary.append(str(p))

    OUT_SUMMARY.write_text("\n".join(summary), encoding="utf-8")

    print("\nSaved:")
    print(OUT_SUMMARY)
    print(OUT_METRICS)
    print(OUT_ROC)
    print(OUT_PR)
    print(OUT_MODEL)
    print("\nDONE")


if __name__ == "__main__":
    main()