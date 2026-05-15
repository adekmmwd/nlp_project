# ============================================================
# LSTM MODEL — Legal Contract Risk Classifier
# ============================================================
# Binary classification: risky clause (1) vs safe clause (0)
# Uses pre-tokenized sequences from 02_tokenization.py
# ============================================================

# %% Cell 1 — Imports
import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

print(f"✅ Imports complete")
print(f"   PyTorch version : {torch.__version__}")
print(f"   CUDA available  : {torch.cuda.is_available()}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using device    : {DEVICE}")

MODEL_NAME = "lstm"
MODELS_DIR = "models"
RESULTS_DIR = "results"
PREDICTIONS_DIR = "predictions"
MODEL_CHECKPOINT_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}_best.pt")
TRAINING_CURVES_PATH = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_training_curves.png")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_confusion_matrix.png")
METRICS_PATH = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_test_metrics.json")
PREDS_PATH = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_preds.npy")
PROBS_PATH = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_probs.npy")
LABELS_PATH = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_labels.npy")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# %% Cell 2 — Hyperparameters
with open("tokens/config.json") as f:
    cfg = json.load(f)

VOCAB_SIZE    = cfg["vocab_size"]      # 6875
MAX_LEN       = cfg["max_len"]         # 140
BATCH_SIZE    = cfg["batch_size"]      # 32

# Model hyperparameters
EMBED_DIM     = 128    # embedding dimension
HIDDEN_DIM    = 256    # LSTM hidden units per direction
NUM_LAYERS    = 2      # stacked LSTM layers
DROPOUT       = 0.4    # dropout rate
BIDIRECTIONAL = True   # use BiLSTM for richer context

# Training
EPOCHS        = 15
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 4      # early stopping patience

print(f"\n{'='*50}")
print(f"  Model Configuration")
print(f"{'='*50}")
print(f"  Vocab size     : {VOCAB_SIZE}")
print(f"  Sequence len   : {MAX_LEN}")
print(f"  Embed dim      : {EMBED_DIM}")
print(f"  Hidden dim     : {HIDDEN_DIM} ({'Bi' if BIDIRECTIONAL else 'Uni'}directional)")
print(f"  LSTM layers    : {NUM_LAYERS}")
print(f"  Dropout        : {DROPOUT}")
print(f"  Batch size     : {BATCH_SIZE}")
print(f"  Epochs         : {EPOCHS}")
print(f"  Learning rate  : {LR}")
print(f"{'='*50}\n")


# ============================================================
# DATASET
# ============================================================

# %% Cell 3 — Load pre-tokenized data
X_train = np.load("tokens/X_train_lstm.npy")
X_val   = np.load("tokens/X_val_lstm.npy")
X_test  = np.load("tokens/X_test_lstm.npy")
y_train = np.load("tokens/y_train.npy")
y_val   = np.load("tokens/y_val.npy")
y_test  = np.load("tokens/y_test.npy")

# Load class weights for imbalance handling
class_weights_np = np.load("splits/class_weights.npy")   # [w0, w1]

print(f"Data shapes:")
print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_val  : {X_val.shape}  y_val  : {y_val.shape}")
print(f"  X_test : {X_test.shape}  y_test : {y_test.shape}")

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
print(f"\nClass distribution (train):")
print(f"  Not risky (0) : {n_neg}  ({n_neg/len(y_train)*100:.1f}%)")
print(f"  Risky     (1) : {n_pos}  ({n_pos/len(y_train)*100:.1f}%)")
print(f"  Class weights : {class_weights_np}")


# %% Cell 4 — PyTorch Dataset
class ClauseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# %% Cell 5 — DataLoaders with weighted sampling for train set
train_ds = ClauseDataset(X_train, y_train)
val_ds   = ClauseDataset(X_val,   y_val)
test_ds  = ClauseDataset(X_test,  y_test)

# WeightedRandomSampler oversamples minority class during training
sample_weights = np.where(y_train == 1,
                          class_weights_np[1],
                          class_weights_np[0])
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float32),
    num_samples=len(train_ds),
    replacement=True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\n✅ DataLoaders ready")
print(f"  Train batches : {len(train_loader)}")
print(f"  Val batches   : {len(val_loader)}")
print(f"  Test batches  : {len(test_loader)}")


# ============================================================
# MODEL DEFINITION
# ============================================================

# %% Cell 6 — BiLSTM Classifier
class LSTMClassifier(nn.Module):
    """
    Bidirectional multi-layer LSTM for binary clause risk classification.

    Architecture:
        Embedding → Dropout → BiLSTM → Attention pooling → FC head → Sigmoid

    Attention pooling lets the model focus on the most risk-relevant
    words regardless of their position in the sequence, outperforming
    a simple last-hidden-state approach.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers, dropout, bidirectional=True):
        super().__init__()

        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim     = hidden_dim

        # ── Embedding ──────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0          # PAD token gets zero gradient
        )
        self.embed_dropout = nn.Dropout(dropout)

        # ── BiLSTM ─────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # ── Attention pooling ──────────────────────────────────
        lstm_out_dim     = hidden_dim * self.num_directions
        self.attention   = nn.Linear(lstm_out_dim, 1)

        # ── Classifier head ────────────────────────────────────
        self.fc1       = nn.Linear(lstm_out_dim, 64)
        self.fc2       = nn.Linear(64, 1)
        self.relu      = nn.ReLU()
        self.fc_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len)

        emb = self.embed_dropout(self.embedding(x))      # (B, T, E)

        lstm_out, _ = self.lstm(emb)                     # (B, T, H*dirs)
        lstm_out    = self.lstm_dropout(lstm_out)

        # Attention: score each time-step then weighted sum
        attn_weights = torch.softmax(
            self.attention(lstm_out), dim=1)             # (B, T, 1)
        context = (attn_weights * lstm_out).sum(dim=1)  # (B, H*dirs)

        out   = self.relu(self.fc1(context))             # (B, 64)
        out   = self.fc_dropout(out)
        logit = self.fc2(out)                            # (B, 1)
        return logit.squeeze(1)                          # (B,)


model = LSTMClassifier(
    vocab_size=VOCAB_SIZE + 2,   # +2 for PAD (0) and OOV (1)
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n✅ Model built — {total_params:,} trainable parameters")
print(model)


# ============================================================
# TRAINING SETUP
# ============================================================

# %% Cell 7 — Loss, optimizer, scheduler
pos_weight = torch.tensor(
    [class_weights_np[1] / class_weights_np[0]],
    dtype=torch.float32
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

print(f"✅ Optimizer : Adam  (lr={LR}, wd={WEIGHT_DECAY})")
print(f"   Loss      : BCEWithLogitsLoss (pos_weight={pos_weight.item():.4f})")
print(f"   Scheduler : ReduceLROnPlateau (patience=2, factor=0.5)")


# ============================================================
# TRAINING LOOP
# ============================================================

# %% Cell 8 — Epoch helper
def run_epoch(loader, model, criterion, optimizer=None, threshold=0.5):
    """One pass. Train if optimizer provided, else evaluate."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs  = torch.sigmoid(logits).detach().cpu().numpy()
            preds  = (probs >= threshold).astype(int)
            labels = y_batch.cpu().numpy().astype(int)

            total_loss  += loss.item() * len(y_batch)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / len(loader.dataset)
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    auc  = roc_auc_score(all_labels, all_probs)

    return avg_loss, acc, prec, rec, f1, auc, all_labels, all_preds, all_probs


# %% Cell 9 — Main training loop
history = {k: [] for k in [
    "train_loss", "val_loss",
    "train_f1",   "val_f1",
    "train_acc",  "val_acc",
    "train_auc",  "val_auc",
]}

best_val_f1  = -1.0
best_epoch   = -1
patience_ctr = 0
best_weights = None

print(f"\n{'='*75}")
print(f"  Training BiLSTM — max {EPOCHS} epochs  |  early-stop patience={PATIENCE}")
print(f"{'='*75}")

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_prec, tr_rec, tr_f1, tr_auc, _, _, _ = \
        run_epoch(train_loader, model, criterion, optimizer)

    vl_loss, vl_acc, vl_prec, vl_rec, vl_f1, vl_auc, _, _, _ = \
        run_epoch(val_loader, model, criterion)

    scheduler.step(vl_f1)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_f1"].append(tr_f1)
    history["val_f1"].append(vl_f1)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)
    history["train_auc"].append(tr_auc)
    history["val_auc"].append(vl_auc)

    marker = ""
    if vl_f1 > best_val_f1:
        best_val_f1  = vl_f1
        best_epoch   = epoch
        patience_ctr = 0
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        marker = "  ← best"
    else:
        patience_ctr += 1

    print(
        f"  Ep {epoch:02d}/{EPOCHS}"
        f"  | Train loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} auc={tr_auc:.4f}"
        f"  | Val   loss={vl_loss:.4f} acc={vl_acc:.4f} f1={vl_f1:.4f} auc={vl_auc:.4f}"
        + marker
    )

    if patience_ctr >= PATIENCE:
        print(f"\n  ⏹  Early stopping at epoch {epoch}.")
        break

print(f"\n✅ Best model: epoch {best_epoch}  (val F1 = {best_val_f1:.4f})")


# ============================================================
# EVALUATION
# ============================================================

# %% Cell 10 — Test-set evaluation
model.load_state_dict(best_weights)

te_loss, te_acc, te_prec, te_rec, te_f1, te_auc, \
    test_labels, test_preds, test_probs = \
    run_epoch(test_loader, model, criterion)

print(f"\n{'='*50}")
print(f"  TEST SET RESULTS  (best epoch = {best_epoch})")
print(f"{'='*50}")
print(f"  Loss      : {te_loss:.4f}")
print(f"  Accuracy  : {te_acc:.4f}")
print(f"  Precision : {te_prec:.4f}")
print(f"  Recall    : {te_rec:.4f}")
print(f"  F1        : {te_f1:.4f}")
print(f"  ROC-AUC   : {te_auc:.4f}")
print(f"\n{classification_report(test_labels, test_preds, target_names=['Safe (0)', 'Risky (1)'])}")


# ============================================================
# PLOTS
# ============================================================

# %% Cell 11 — Training curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("BiLSTM Training History", fontsize=14, fontweight="bold")
eps = range(1, len(history["train_loss"]) + 1)

metrics = [("Loss", "train_loss", "val_loss"),
           ("F1 Score", "train_f1", "val_f1"),
           ("ROC-AUC",  "train_auc", "val_auc")]

for ax, (title, tr_key, vl_key) in zip(axes, metrics):
    ax.plot(eps, history[tr_key], label="Train")
    ax.plot(eps, history[vl_key], label="Val")
    ax.axvline(best_epoch, color="green", linestyle="--",
               alpha=0.7, label=f"Best (ep {best_epoch})")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()

plt.tight_layout()
plt.savefig(TRAINING_CURVES_PATH, dpi=150, bbox_inches="tight")
print(f"✅ Saved: {TRAINING_CURVES_PATH}")


# %% Cell 12 — Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Safe (0)", "Risky (1)"],
            yticklabels=["Safe (0)", "Risky (1)"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — Test Set\n"
             f"Acc={te_acc:.3f}  F1={te_f1:.3f}  AUC={te_auc:.3f}")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
print(f"✅ Saved: {CONFUSION_MATRIX_PATH}")


# ============================================================
# SAVE MODEL
# ============================================================

# %% Cell 13 — Save checkpoint
torch.save({
    "model_state_dict" : best_weights,
    "model_config" : {
        "vocab_size"    : VOCAB_SIZE + 2,
        "embed_dim"     : EMBED_DIM,
        "hidden_dim"    : HIDDEN_DIM,
        "num_layers"    : NUM_LAYERS,
        "dropout"       : DROPOUT,
        "bidirectional" : BIDIRECTIONAL,
    },
    "best_epoch"   : best_epoch,
    "val_f1"       : best_val_f1,
    "test_metrics" : {
        "loss"      : te_loss,
        "accuracy"  : te_acc,
        "precision" : te_prec,
        "recall"    : te_rec,
        "f1"        : te_f1,
        "roc_auc"   : te_auc,
    }
}, MODEL_CHECKPOINT_PATH)

np.save(PREDS_PATH, np.asarray(test_preds))
np.save(PROBS_PATH, np.asarray(test_probs))
np.save(LABELS_PATH, np.asarray(test_labels))
with open(METRICS_PATH, "w") as f:
    json.dump(
        {
            "model": MODEL_NAME,
            "best_epoch": int(best_epoch),
            "val_f1": round(float(best_val_f1), 4),
            "test_loss": round(float(te_loss), 4),
            "test_accuracy": round(float(te_acc), 4),
            "test_precision": round(float(te_prec), 4),
            "test_recall": round(float(te_rec), 4),
            "test_f1": round(float(te_f1), 4),
            "test_roc_auc": round(float(te_auc), 4),
        },
        f,
        indent=2,
    )

print(f"\n✅ Checkpoint saved → {MODEL_CHECKPOINT_PATH}")
print(f"✅ Metrics saved    → {METRICS_PATH}")
print(f"\n{'='*50}")
print(f"  DONE — Summary")
print(f"{'='*50}")
print(f"  Best epoch  : {best_epoch}")
print(f"  Val   F1    : {best_val_f1:.4f}")
print(f"  Test  F1    : {te_f1:.4f}")
print(f"  Test  AUC   : {te_auc:.4f}")
print(f"{'='*50}")
