import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


PREDICTIONS_DIR = "predictions"
RESULTS_DIR = "results"
SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, "all_models_comparison.csv")
SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR, "all_models_comparison.json")
METRICS_PLOT_PATH = os.path.join(RESULTS_DIR, "all_models_metric_comparison.png")
RANKING_PLOT_PATH = os.path.join(RESULTS_DIR, "all_models_f1_ranking.png")
DISAGREEMENT_PLOT_PATH = os.path.join(RESULTS_DIR, "all_models_prediction_disagreement.png")

LEGACY_ALIAS = {
    "bert_untrained": "bert_base_untrained",
    "bert_finetuned_loaded": "bert_109m_trained",
    "bert_109m_trained_loaded": "bert_109m_trained",
    "legal_bert_109m_trained_loaded": "legal_bert_109m_trained",
}

LEGACY_SKIP = {
    "bert",
}


def infer_family(model_name: str) -> str:
    name = model_name.lower()
    if "legal_bert" in name or "lbert" in name:
        return "legal_bert"
    if "bert" in name:
        return "bert"
    if "lstm" in name:
        return "lstm"
    if "gru" in name:
        return "gru"
    return "other"


def infer_param_size(model_name: str) -> str:
    name = model_name.lower()
    if "109m" in name:
        return "109M"
    if "57m" in name:
        return "57M"
    if "base" in name:
        return "base"
    return "n/a"


def infer_training_status(model_name: str) -> str:
    name = model_name.lower()
    if "untrained" in name:
        return "untrained"
    if "trained" in name or "finetuned" in name or "loaded" in name:
        return "trained"
    return "unknown"


def safe_roc_auc(labels: np.ndarray, probs: Optional[np.ndarray]) -> float:
    if probs is None:
        return float("nan")
    if len(labels) != len(probs):
        return float("nan")
    if np.unique(labels).size < 2:
        return float("nan")
    if not np.isfinite(probs).all():
        return float("nan")
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")


def discover_model_prefixes() -> List[str]:
    if not os.path.isdir(PREDICTIONS_DIR):
        return []
    prefixes = []
    for filename in os.listdir(PREDICTIONS_DIR):
        if filename.endswith("_preds.npy"):
            prefixes.append(filename[: -len("_preds.npy")])
    return sorted(set(prefixes))


def canonicalize_prefix(prefix: str) -> Tuple[Optional[str], int, Optional[str]]:
    """
    Returns (canonical_name, priority, reason)
    lower priority is preferred when deduplicating.
    """
    if prefix in LEGACY_SKIP:
        return None, 99, "legacy generic name"

    if prefix in LEGACY_ALIAS:
        return LEGACY_ALIAS[prefix], 2, f"legacy alias -> {LEGACY_ALIAS[prefix]}"

    if prefix.endswith("_loaded"):
        return prefix[: -len("_loaded")], 1, "loaded suffix removed"

    return prefix, 0, None


def choose_canonical_prefixes(prefixes: List[str]) -> Tuple[List[str], List[str], List[str]]:
    chosen: Dict[str, Tuple[str, int]] = {}
    skipped: List[str] = []
    deduped: List[str] = []

    for prefix in sorted(prefixes):
        canonical, priority, reason = canonicalize_prefix(prefix)
        if canonical is None:
            skipped.append(f"{prefix} ({reason})")
            continue

        existing = chosen.get(canonical)
        if existing is None:
            chosen[canonical] = (prefix, priority)
            if reason is not None:
                deduped.append(f"{prefix} ({reason})")
            continue

        kept_prefix, kept_priority = existing
        if priority < kept_priority:
            deduped.append(f"{kept_prefix} (duplicate of {canonical})")
            chosen[canonical] = (prefix, priority)
        else:
            deduped.append(f"{prefix} (duplicate of {canonical})")

    selected = [source for source, _ in chosen.values()]
    return sorted(selected), skipped, deduped


def load_arrays(prefix: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    preds_path = os.path.join(PREDICTIONS_DIR, f"{prefix}_preds.npy")
    labels_path = os.path.join(PREDICTIONS_DIR, f"{prefix}_labels.npy")
    probs_path = os.path.join(PREDICTIONS_DIR, f"{prefix}_probs.npy")

    preds = np.asarray(np.load(preds_path)).reshape(-1).astype(int)
    labels = np.asarray(np.load(labels_path)).reshape(-1).astype(int)

    probs = None
    if os.path.exists(probs_path):
        loaded_probs = np.asarray(np.load(probs_path), dtype=np.float64).reshape(-1)
        if len(loaded_probs) == len(preds):
            probs = loaded_probs

    return preds, labels, probs


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: Optional[np.ndarray]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": safe_roc_auc(labels, probs),
    }


def save_metric_comparison_plot(df: pd.DataFrame) -> None:
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    plot_df = df[["model"] + metric_cols].copy()
    melted = plot_df.melt(id_vars="model", var_name="metric", value_name="score")

    fig_w = max(12, 0.8 * len(df) + 6)
    plt.figure(figsize=(fig_w, 6))
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("All Models - Metric Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(METRICS_PLOT_PATH, dpi=150)
    plt.close()


def save_f1_ranking_plot(df: pd.DataFrame) -> None:
    rank_df = df.sort_values("f1", ascending=False).copy()
    rank_df["display"] = rank_df["model"]

    plt.figure(figsize=(10, max(4, 0.45 * len(rank_df) + 1)))
    sns.barplot(data=rank_df, y="display", x="f1", color="#2f6f9f")
    plt.xlim(0, 1)
    plt.xlabel("F1 score")
    plt.ylabel("Model")
    plt.title("All Models - F1 Ranking")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(RANKING_PLOT_PATH, dpi=150)
    plt.close()


def save_disagreement_heatmap(model_runs: Dict[str, Dict[str, np.ndarray]]) -> None:
    model_names = sorted(model_runs.keys())
    if len(model_names) < 2:
        return

    matrix = np.full((len(model_names), len(model_names)), np.nan, dtype=np.float64)
    for i, m1 in enumerate(model_names):
        preds1 = model_runs[m1]["preds"]
        labels1 = model_runs[m1]["labels"]
        for j, m2 in enumerate(model_names):
            preds2 = model_runs[m2]["preds"]
            labels2 = model_runs[m2]["labels"]
            if i == j:
                matrix[i, j] = 0.0
                continue
            if len(preds1) != len(preds2) or len(labels1) != len(labels2):
                continue
            if not np.array_equal(labels1, labels2):
                continue
            matrix[i, j] = float(np.mean(preds1 != preds2))

    plt.figure(figsize=(max(8, 0.6 * len(model_names) + 2), max(6, 0.55 * len(model_names) + 1)))
    sns.heatmap(
        matrix,
        xticklabels=model_names,
        yticklabels=model_names,
        annot=True,
        fmt=".3f",
        cmap="magma_r",
        vmin=0.0,
        vmax=1.0,
        mask=np.isnan(matrix),
    )
    plt.title("Prediction Disagreement Rate (pairwise)")
    plt.xlabel("Model")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(DISAGREEMENT_PLOT_PATH, dpi=150)
    plt.close()


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    prefixes = discover_model_prefixes()
    if not prefixes:
        raise FileNotFoundError("No model prediction files found in predictions/ (expected *_preds.npy).")
    prefixes, skipped_legacy, skipped_duplicates = choose_canonical_prefixes(prefixes)

    rows = []
    model_runs: Dict[str, Dict[str, np.ndarray]] = {}
    skipped = []

    for prefix in prefixes:
        labels_path = os.path.join(PREDICTIONS_DIR, f"{prefix}_labels.npy")
        preds_path = os.path.join(PREDICTIONS_DIR, f"{prefix}_preds.npy")
        if not os.path.exists(labels_path) or not os.path.exists(preds_path):
            skipped.append(prefix)
            continue

        try:
            preds, labels, probs = load_arrays(prefix)
        except Exception as exc:
            skipped.append(f"{prefix} ({exc})")
            continue

        if len(preds) != len(labels):
            skipped.append(f"{prefix} (pred/label length mismatch)")
            continue

        metrics = compute_metrics(labels, preds, probs)
        rows.append(
            {
                "model": prefix,
                "family": infer_family(prefix),
                "param_size": infer_param_size(prefix),
                "training_status": infer_training_status(prefix),
                "samples": int(len(labels)),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )
        model_runs[prefix] = {"preds": preds, "labels": labels}

    if not rows:
        raise RuntimeError("No valid model runs found from predictions/*.npy files.")

    df = pd.DataFrame(rows).sort_values(
        by=["f1", "roc_auc", "accuracy"], ascending=[False, False, False]
    )
    df.to_csv(SUMMARY_CSV_PATH, index=False)

    json_records = df.where(pd.notnull(df), None).to_dict(orient="records")
    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(json_records, f, indent=2)

    save_metric_comparison_plot(df)
    save_f1_ranking_plot(df)
    save_disagreement_heatmap(model_runs)

    print("\n=== MODEL COMPARISON (sorted by F1) ===")
    print(
        df[
            [
                "model",
                "family",
                "param_size",
                "training_status",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved CSV          : {SUMMARY_CSV_PATH}")
    print(f"Saved JSON         : {SUMMARY_JSON_PATH}")
    print(f"Saved metrics plot : {METRICS_PLOT_PATH}")
    print(f"Saved F1 ranking   : {RANKING_PLOT_PATH}")
    if len(model_runs) >= 2:
        print(f"Saved disagreement : {DISAGREEMENT_PLOT_PATH}")
    if skipped:
        print("\nSkipped entries:")
        for item in skipped:
            print(f"  - {item}")
    if skipped_legacy:
        print("\nSkipped legacy names:")
        for item in skipped_legacy:
            print(f"  - {item}")
    if skipped_duplicates:
        print("\nSkipped duplicate aliases:")
        for item in skipped_duplicates:
            print(f"  - {item}")


if __name__ == "__main__":
    main()