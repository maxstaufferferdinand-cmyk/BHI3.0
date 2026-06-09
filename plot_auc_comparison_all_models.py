from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Max Stauffer\BreakingRules")

OUT_DIR = BASE_DIR / "auc_comparison_outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_PNG = OUT_DIR / "roc_auc_comparison_all_models.png"
OUT_PDF = OUT_DIR / "roc_auc_comparison_all_models.pdf"
OUT_CSV = OUT_DIR / "roc_auc_comparison_all_models_summary.csv"

# Current model output
CURRENT_DATASET = (
    BASE_DIR
    / "cluster_edge_strength_model_outputs"
    / "cluster_edge_strength_top20_training_dataset.csv"
)

# Manual AUCs from our previous runs.
# These will be drawn as schematic ROC curves if no raw predictions exist.
MANUAL_AUCS = [
    {
        "name": "Future edge prediction v1",
        "auc": 0.5118,
        "type": "schematic",
        "note": "Temporal future-edge test; near random",
        "color": "#D55E00",
        "linestyle": "--",
    },
    {
        "name": "Random historical validation v1",
        "auc": 0.999996,
        "type": "schematic",
        "note": "Random validation; likely leakage/trivial recognition",
        "color": "#999999",
        "linestyle": ":",
    },
]

# Real ROC from current model, if dataset is present.
REAL_MODELS = [
    {
        "name": "Cluster-edge strength top20 model",
        "path": CURRENT_DATASET,
        "label_col": "label",
        "pred_col": "ml_predicted_probability_strong",
        "color": "#0072B2",
        "linestyle": "-",
        "note": "Top 20% vs bottom 50%; middle 30% excluded",
    }
]


# ============================================================
# HELPERS
# ============================================================

def schematic_roc_from_auc(auc, n=500):
    """
    Creates a smooth schematic ROC curve with approximately the requested AUC.
    This is only for models where we have AUC but not y_true/pred.
    
    We use TPR = FPR^alpha.
    Integral from 0 to 1 of x^alpha dx = 1/(alpha + 1) = AUC.
    Therefore alpha = 1/AUC - 1.
    """
    auc = float(auc)
    auc = min(max(auc, 0.500001), 0.999999)

    fpr = np.linspace(0, 1, n)
    alpha = (1.0 / auc) - 1.0
    tpr = fpr ** alpha

    return fpr, tpr


def load_real_roc(model_cfg):
    path = Path(model_cfg["path"])
    if not path.exists():
        print(f"Missing file, skipping real ROC: {path}")
        return None

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    label_col = model_cfg["label_col"]
    pred_col = model_cfg["pred_col"]

    if label_col not in df.columns or pred_col not in df.columns:
        print(f"Missing required columns in {path}")
        print(f"Needed: {label_col}, {pred_col}")
        return None

    tmp = df[[label_col, pred_col]].dropna().copy()
    tmp[label_col] = pd.to_numeric(tmp[label_col], errors="coerce")
    tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
    tmp = tmp.dropna()

    if tmp.empty:
        print(f"No non-missing predictions in {path}")
        return None

    y_true = tmp[label_col].astype(int).values
    y_pred = tmp[pred_col].astype(float).values

    if len(np.unique(y_true)) < 2:
        print(f"Only one class present in {path}")
        return None

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return {
        "name": model_cfg["name"],
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "color": model_cfg["color"],
        "linestyle": model_cfg["linestyle"],
        "note": model_cfg.get("note", ""),
        "type": "real",
        "n": len(tmp),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    curves = []

    # Real ROC curves
    for cfg in REAL_MODELS:
        result = load_real_roc(cfg)
        if result is not None:
            curves.append(result)

    # Manual/schematic ROC curves
    for cfg in MANUAL_AUCS:
        fpr, tpr = schematic_roc_from_auc(cfg["auc"])
        curves.append({
            "name": cfg["name"],
            "fpr": fpr,
            "tpr": tpr,
            "auc": cfg["auc"],
            "color": cfg["color"],
            "linestyle": cfg["linestyle"],
            "note": cfg.get("note", ""),
            "type": "schematic",
            "n": np.nan,
        })

    if not curves:
        raise RuntimeError("No curves available to plot.")

    # Sort: lower AUC first, strongest last so it is visually on top
    curves = sorted(curves, key=lambda x: x["auc"])

    plt.figure(figsize=(7.2, 6.4), dpi=300)

    # Chance line
    plt.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle="-",
        linewidth=1.0,
        alpha=0.45,
        label="Chance (AUC = 0.500)",
    )

    for c in curves:
        label = f"{c['name']} (AUC = {c['auc']:.3f})"
        if c["type"] == "schematic":
            label += " [schematic]"

        plt.plot(
            c["fpr"],
            c["tpr"],
            color=c["color"],
            linestyle=c["linestyle"],
            linewidth=2.4,
            label=label,
        )

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC-AUC comparison across model stages")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, loc="lower right", fontsize=8.5)
    plt.tight_layout()

    plt.savefig(OUT_PNG, dpi=300)
    plt.savefig(OUT_PDF)
    plt.close()

    summary = pd.DataFrame([
        {
            "model": c["name"],
            "auc": c["auc"],
            "curve_type": c["type"],
            "n_with_predictions": c["n"],
            "note": c["note"],
        }
        for c in curves
    ]).sort_values("auc", ascending=False)

    summary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("Saved:")
    print(OUT_PNG)
    print(OUT_PDF)
    print(OUT_CSV)
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()