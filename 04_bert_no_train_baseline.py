# ============================================================
# BERT NO-TRAINING BASELINE
# bert-base-uncased | raw clause text | no text preprocessing
# ============================================================

import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
SEED = 42
BERT_MODEL = "bert-base-uncased"
MODEL_NAME = "bert_base"
MAX_LEN = 140
BATCH_SIZE = 32
ATTN_IMPLEMENTATION = "eager"

MODELS_DIR = "models"
PREDICTIONS_DIR = "predictions"
RESULTS_DIR = "results"

UNTRAINED_PREFIX = f"{MODEL_NAME}_untrained"
UNTRAINED_MODEL_PATH = os.path.join(MODELS_DIR, f"{UNTRAINED_PREFIX}.pt")
FINE_TUNED_MODEL_CANDIDATES = [
    ("bert_109m_trained", os.path.join(MODELS_DIR, "bert_109m_trained_best.pt")),
    ("bert_57m_trained", os.path.join(MODELS_DIR, "bert_57m_trained_best.pt")),
    ("bert_trained_legacy", os.path.join(MODELS_DIR, "bert_best.pt")),
]

RISKY_LABELS = {
    "Indemnification",
    "Termination For Convenience",
    "Uncapped Liability",
    "Ip Ownership Assignment",
    "Non-Compete",
    "Exclusivity",
    "Auto-Renewal",
    "Change Of Control",
    "Liquidated Damages",
    "Anti-Assignment",
    "Most Favored Nation",
    "Warranty Duration",
    "Cap On Liability",
    "Audit Rights",
    "Price Restrictions",
    "Rofr/Rofo/Rofn",
    "Irrevocable Or Perpetual License",
    "Joint Ip Ownership",
    "No-Solicit Of Employees",
    "Unlimited/All-You-Can-Eat-License",
}


# ============================================================
# SETUP
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

IS_ROCM = getattr(torch.version, "hip", None) is not None
torch.backends.cudnn.deterministic = not IS_ROCM
torch.backends.cudnn.benchmark = IS_ROCM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Seed        : {SEED}")
print(f"Device      : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        print("Attention   : eager/math SDPA")

print("\nBaseline note:")
print("  bert-base-uncased has no trained risk-classification head.")
print("  This script evaluates its randomly initialized classifier head.")
print("  Use it as an untrained-BERT sanity baseline, not as a legal-risk model.")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DATA
# ============================================================
print("\nLoading dataset...")
ds = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_clause_classification")
df = ds["train"].to_pandas()

df["risk"] = df["label"].apply(lambda x: 1 if x in RISKY_LABELS else 0)

# Raw clause text only. No stemming, lowercasing, stopword removal, or cleaning.
df["text"] = df["clause"].astype(str)
df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)

train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df["risk"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["risk"]
)

print(f"Dataset     : {df.shape}")
print(f"Train       : {len(train_df)} | risk ratio: {train_df['risk'].mean():.3f}")
print(f"Val         : {len(val_df)} | risk ratio: {val_df['risk'].mean():.3f}")
print(f"Test        : {len(test_df)} | risk ratio: {test_df['risk'].mean():.3f}")


# ============================================================
# TOKENIZATION
# ============================================================
print(f"\nLoading tokenizer: {BERT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)


def encode_texts(texts):
    encoding = tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]


print("Tokenizing test clauses...")
X_test_ids, X_test_mask = encode_texts(test_df["text"])
y_test = torch.tensor(test_df["risk"].values, dtype=torch.long)


class ContractDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


test_loader = DataLoader(
    ContractDataset(X_test_ids, X_test_mask, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(f"Test batches: {len(test_loader)}")


# ============================================================
# METRICS / OUTPUT HELPERS
# ============================================================
def compute_metrics(labels, preds, probs):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs, dtype=np.float64)

    if np.unique(labels).size < 2 or not np.isfinite(probs).all():
        auc = None
    else:
        auc = float(roc_auc_score(labels, probs))

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": auc,
    }


def rounded_metrics(metrics):
    return {
        key: None if value is None else round(float(value), 4)
        for key, value in metrics.items()
    }


def save_metrics(prefix, metrics):
    path = os.path.join(PREDICTIONS_DIR, f"{prefix}_metrics.json")
    with open(path, "w") as f:
        json.dump(rounded_metrics(metrics), f, indent=2)
    return path


def save_predictions(prefix, preds, probs, labels):
    np.save(os.path.join(PREDICTIONS_DIR, f"{prefix}_preds.npy"), np.asarray(preds))
    np.save(os.path.join(PREDICTIONS_DIR, f"{prefix}_probs.npy"), np.asarray(probs))
    np.save(os.path.join(PREDICTIONS_DIR, f"{prefix}_labels.npy"), np.asarray(labels))


def save_model_state(model, path):
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state_dict, path)
    return path


def save_confusion_matrix(prefix, labels, preds, title):
    cm = confusion_matrix(labels, preds)
    path = os.path.join(RESULTS_DIR, f"{prefix}_confusion_matrix.png")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Risky", "Risky"],
        yticklabels=["Not Risky", "Risky"],
        annot_kws={"size": 14},
    )
    plt.title(title, fontsize=13)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_confusion_matrix_grid(rows, filename, title):
    path = os.path.join(RESULTS_DIR, filename)
    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 5))
    if len(rows) == 1:
        axes = [axes]

    for ax, (name, labels, preds) in zip(axes, rows):
        cm = confusion_matrix(labels, preds)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Risky", "Risky"],
            yticklabels=["Not Risky", "Risky"],
            annot_kws={"size": 13},
            ax=ax,
        )
        ax.set_title(name)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_probability_histogram(prefix, labels, probs, title):
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=np.float64)
    path = os.path.join(RESULTS_DIR, f"{prefix}_probability_distribution.png")

    plt.figure(figsize=(7, 5))
    sns.histplot(
        probs[labels == 0],
        bins=30,
        stat="density",
        color="#2f6f9f",
        alpha=0.55,
        label="Not Risky",
    )
    sns.histplot(
        probs[labels == 1],
        bins=30,
        stat="density",
        color="#c76b29",
        alpha=0.55,
        label="Risky",
    )
    plt.xlim(0, 1)
    plt.xlabel("Predicted risky probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_metrics_bar_chart(rows, filename, title):
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    path = os.path.join(RESULTS_DIR, filename)

    x = np.arange(len(metric_keys))
    width = min(0.8 / len(rows), 0.35)

    plt.figure(figsize=(9, 5))
    for idx, (name, metrics) in enumerate(rows):
        values = [
            np.nan if metrics[key] is None else float(metrics[key])
            for key in metric_keys
        ]
        offset = (idx - (len(rows) - 1) / 2) * width
        plt.bar(x + offset, values, width, label=name)

    plt.ylim(0, 1)
    plt.xticks(x, metric_labels)
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_probability_scatter(
    untrained_prefix, fine_prefix, untrained_name, fine_name, untrained_probs, fine_probs, labels
):
    labels = np.asarray(labels)
    path = os.path.join(
        RESULTS_DIR, f"{untrained_prefix}_vs_{fine_prefix}_probabilities.png"
    )

    plt.figure(figsize=(6, 6))
    plt.scatter(
        np.asarray(untrained_probs)[labels == 0],
        np.asarray(fine_probs)[labels == 0],
        s=16,
        alpha=0.55,
        label="Not Risky",
        color="#2f6f9f",
    )
    plt.scatter(
        np.asarray(untrained_probs)[labels == 1],
        np.asarray(fine_probs)[labels == 1],
        s=16,
        alpha=0.55,
        label="Risky",
        color="#c76b29",
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(f"{untrained_name} risky probability")
    plt.ylabel(f"{fine_name} risky probability")
    plt.title(f"{untrained_name} vs {fine_name} Probability Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_roc_curve(prefix, labels, probs, title):
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=np.float64)
    path = os.path.join(RESULTS_DIR, f"{prefix}_roc_curve.png")

    if np.unique(labels).size < 2 or not np.isfinite(probs).all():
        return None

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Random")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_roc_curve_comparison(rows, filename, title):
    path = os.path.join(RESULTS_DIR, filename)

    plt.figure(figsize=(6, 5))
    plotted = False
    for name, labels, probs in rows:
        labels = np.asarray(labels)
        probs = np.asarray(probs, dtype=np.float64)
        if np.unique(labels).size < 2 or not np.isfinite(probs).all():
            continue

        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.4f})")
        plotted = True

    if not plotted:
        plt.close()
        return None

    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Random")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def print_metrics_table(rows):
    print(f"\n{'Model':<24} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    for name, metrics in rows:
        auc = metrics["roc_auc"]
        auc_text = "nan" if auc is None else f"{auc:.4f}"
        print(
            f"{name:<24} "
            f"{metrics['accuracy']:>8.4f} "
            f"{metrics['precision']:>8.4f} "
            f"{metrics['recall']:>8.4f} "
            f"{metrics['f1']:>8.4f} "
            f"{auc_text:>8}"
        )


def load_bert_classifier():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL,
        num_labels=2,
        ignore_mismatched_sizes=True,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    return model.to(DEVICE)


def resolve_fine_tuned_checkpoint():
    for tag, checkpoint_path in FINE_TUNED_MODEL_CANDIDATES:
        if os.path.exists(checkpoint_path):
            return tag, checkpoint_path
    return None, None


def evaluate_model(model, loader, desc):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()

            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"{desc} produced NaN/Inf logits")

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.asarray(all_preds), np.asarray(all_labels), np.asarray(all_probs)


def save_comparison(untrained_prefix, fine_prefix, untrained, fine_tuned):
    untrained_metrics, untrained_preds, untrained_probs = untrained
    fine_metrics, fine_preds, fine_probs = fine_tuned

    agreement = float(np.mean(untrained_preds == fine_preds))
    comparison = {
        "untrained": rounded_metrics(untrained_metrics),
        "fine_tuned": rounded_metrics(fine_metrics),
        "fine_tuned_minus_untrained": {
            key: (
                None
                if untrained_metrics[key] is None or fine_metrics[key] is None
                else round(float(fine_metrics[key] - untrained_metrics[key]), 4)
            )
            for key in untrained_metrics
        },
        "prediction_agreement": round(agreement, 4),
        "prediction_disagreements": int(np.sum(untrained_preds != fine_preds)),
    }

    json_path = os.path.join(
        PREDICTIONS_DIR, f"{untrained_prefix}_vs_{fine_prefix}_comparison.json"
    )
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2)

    comparison_columns = [
        column
        for column in ["file_name", "pages", "label", "risk", "text"]
        if column in test_df.columns
    ]
    compare_df = test_df[comparison_columns].copy()
    compare_df["untrained_pred"] = untrained_preds
    compare_df["untrained_risky_prob"] = untrained_probs
    compare_df["fine_tuned_pred"] = fine_preds
    compare_df["fine_tuned_risky_prob"] = fine_probs
    compare_df["models_disagree"] = untrained_preds != fine_preds

    csv_path = os.path.join(
        PREDICTIONS_DIR, f"{untrained_prefix}_vs_{fine_prefix}_test.csv"
    )
    compare_df.to_csv(csv_path, index=False)

    print(f"\nPrediction agreement: {agreement:.4f}")
    print(f"Disagreements       : {comparison['prediction_disagreements']}")
    print(f"Comparison JSON     : {json_path}")
    print(f"Comparison CSV      : {csv_path}")


# ============================================================
# UNTRAINED BERT BASELINE
# ============================================================
print("\nEvaluating untrained bert-base-uncased classifier...")
untrained_model = load_bert_classifier()
untrained_model_path = save_model_state(untrained_model, UNTRAINED_MODEL_PATH)
untrained_preds, test_labels, untrained_probs = evaluate_model(
    untrained_model, test_loader, "Untrained BERT"
)
untrained_metrics = compute_metrics(test_labels, untrained_preds, untrained_probs)

save_predictions(UNTRAINED_PREFIX, untrained_preds, untrained_probs, test_labels)
untrained_metrics_path = save_metrics(UNTRAINED_PREFIX, untrained_metrics)
untrained_cm_path = save_confusion_matrix(
    UNTRAINED_PREFIX,
    test_labels,
    untrained_preds,
    "Untrained BERT - Confusion Matrix",
)
untrained_prob_path = save_probability_histogram(
    UNTRAINED_PREFIX,
    test_labels,
    untrained_probs,
    "Untrained BERT - Risk Probability Distribution",
)
untrained_metrics_plot_path = save_metrics_bar_chart(
    [("Untrained BERT", untrained_metrics)],
    f"{UNTRAINED_PREFIX}_metrics.png",
    "Untrained BERT Metrics",
)
untrained_roc_path = save_roc_curve(
    UNTRAINED_PREFIX,
    test_labels,
    untrained_probs,
    "Untrained BERT - ROC Curve",
)

print_metrics_table([("Untrained BERT", untrained_metrics)])
print("\nClassification report - untrained BERT")
print(
    classification_report(
        test_labels,
        untrained_preds,
        target_names=["Not Risky", "Risky"],
        zero_division=0,
    )
)
print(f"Saved metrics      : {untrained_metrics_path}")
print(f"Saved confusion    : {untrained_cm_path}")
print(f"Saved probabilities: {untrained_prob_path}")
print(f"Saved metric plot  : {untrained_metrics_plot_path}")
print(f"Saved ROC curve    : {untrained_roc_path}")
print(f"Saved model        : {untrained_model_path}")


# ============================================================
# OPTIONAL FINE-TUNED MODEL COMPARISON
# ============================================================
fine_tuned_tag, fine_tuned_checkpoint = resolve_fine_tuned_checkpoint()
if fine_tuned_checkpoint is not None:
    fine_tuned_prefix = f"{fine_tuned_tag}_loaded"
    fine_tuned_title = fine_tuned_tag.replace("_", " ").title()
    print(f"\nEvaluating fine-tuned checkpoint: {fine_tuned_checkpoint}")
    del untrained_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fine_tuned_model = load_bert_classifier()
    fine_tuned_state = torch.load(
        fine_tuned_checkpoint,
        map_location=DEVICE,
        weights_only=True,
    )
    fine_tuned_model.load_state_dict(fine_tuned_state)

    fine_preds, fine_labels, fine_probs = evaluate_model(
        fine_tuned_model, test_loader, "Fine-tuned BERT"
    )
    if not np.array_equal(test_labels, fine_labels):
        raise ValueError("Fine-tuned labels do not match untrained baseline labels")

    fine_metrics = compute_metrics(fine_labels, fine_preds, fine_probs)
    save_predictions(fine_tuned_prefix, fine_preds, fine_probs, fine_labels)
    fine_metrics_path = save_metrics(fine_tuned_prefix, fine_metrics)
    fine_cm_path = save_confusion_matrix(
        fine_tuned_prefix,
        fine_labels,
        fine_preds,
        f"{fine_tuned_title} - Confusion Matrix",
    )
    fine_prob_path = save_probability_histogram(
        fine_tuned_prefix,
        fine_labels,
        fine_probs,
        f"{fine_tuned_title} - Risk Probability Distribution",
    )
    comparison_metrics_plot_path = save_metrics_bar_chart(
        [
            ("Untrained BERT", untrained_metrics),
            (fine_tuned_title, fine_metrics),
        ],
        f"{UNTRAINED_PREFIX}_vs_{fine_tuned_prefix}_metrics.png",
        f"Untrained BERT vs {fine_tuned_title} Metrics",
    )
    comparison_cm_path = save_confusion_matrix_grid(
        [
            ("Untrained BERT", test_labels, untrained_preds),
            (fine_tuned_title, fine_labels, fine_preds),
        ],
        f"{UNTRAINED_PREFIX}_vs_{fine_tuned_prefix}_confusion_matrices.png",
        "Confusion Matrix Comparison",
    )
    comparison_prob_path = save_probability_scatter(
        UNTRAINED_PREFIX,
        fine_tuned_prefix,
        "Untrained BERT",
        fine_tuned_title,
        untrained_probs,
        fine_probs,
        test_labels,
    )
    fine_roc_path = save_roc_curve(
        fine_tuned_prefix,
        fine_labels,
        fine_probs,
        f"{fine_tuned_title} - ROC Curve",
    )
    comparison_roc_path = save_roc_curve_comparison(
        [
            ("Untrained BERT", test_labels, untrained_probs),
            (fine_tuned_title, fine_labels, fine_probs),
        ],
        f"{UNTRAINED_PREFIX}_vs_{fine_tuned_prefix}_roc_curve.png",
        f"Untrained BERT vs {fine_tuned_title} ROC Curve Comparison",
    )

    print_metrics_table(
        [
            ("Untrained BERT", untrained_metrics),
            (fine_tuned_title, fine_metrics),
        ]
    )
    print("\nClassification report - fine-tuned BERT")
    print(
        classification_report(
            fine_labels,
            fine_preds,
            target_names=["Not Risky", "Risky"],
            zero_division=0,
        )
    )
    print(f"Saved fine-tuned metrics   : {fine_metrics_path}")
    print(f"Saved fine-tuned confusion : {fine_cm_path}")
    print(f"Saved fine-tuned probs     : {fine_prob_path}")
    print(f"Saved comparison metrics   : {comparison_metrics_plot_path}")
    print(f"Saved comparison confusion : {comparison_cm_path}")
    print(f"Saved probability scatter  : {comparison_prob_path}")
    print(f"Saved fine-tuned ROC       : {fine_roc_path}")
    print(f"Saved comparison ROC       : {comparison_roc_path}")

    save_comparison(
        UNTRAINED_PREFIX,
        fine_tuned_prefix,
        (untrained_metrics, untrained_preds, untrained_probs),
        (fine_metrics, fine_preds, fine_probs),
    )
else:
    candidate_paths = ", ".join(path for _, path in FINE_TUNED_MODEL_CANDIDATES)
    print(f"\nFine-tuned checkpoint not found. Looked for: {candidate_paths}")
    print("Only the untrained BERT baseline was evaluated.")

print("\nDone.")
