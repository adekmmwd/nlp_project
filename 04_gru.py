# ============================================================
# GRU MODEL — Legal Contract Risk Classifier
# ============================================================
# Binary classification: risky clause (1) vs safe clause (0)
# Uses pre-tokenized sequences from 02_tokenization.py
# ============================================================

# %% Cell 1 — Imports
import os
import json
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

MODEL_NAME = "gru"
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
# GRU trains faster than LSTM (no cell state), so we can afford
# a slightly wider hidden dim for the same training time budget.
EMBED_DIM     = 128    # embedding dimension
HIDDEN_DIM    = 256    # GRU hidden units per direction
NUM_LAYERS    = 2      # stacked GRU layers
DROPOUT       = 0.4    # dropout rate
BIDIRECTIONAL = True   # use BiGRU for richer context

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
print(f"  GRU layers     : {NUM_LAYERS}")
print(f"  Dropout        : {DROPOUT}")
print(f"  Batch size     : {BATCH_SIZE}")
print(f"  Epochs         : {EPOCHS}")
print(f"  Learning rate  : {LR}")
print(f"{'='*50}\n")


# ============================================================
# DATASET
# ============================================================

# %% Cell 3 — Load pre-tokenized data
X_train = np.load("tokens/X_train_lstm.npy")   # same sequences as LSTM
X_val   = np.load("tokens/X_val_lstm.npy")
X_test  = np.load("tokens/X_test_lstm.npy")
y_train = np.load("tokens/y_train.npy")
y_val   = np.load("tokens/y_val.npy")
y_test  = np.load("tokens/y_test.npy")

class_weights_np = np.load("splits/class_weights.npy")   # [w0, w1]

print(f"Data shapes:")
print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_val  : {X_val.shape}  y_val  : {y_val.shape}")
print(f"  X_test : {X_test.shape}  y_test : {y_test.shape}")

n_pos = int(y_train.sum())
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


# %% Cell 5 — DataLoaders
train_ds = ClauseDataset(X_train, y_train)
val_ds   = ClauseDataset(X_val,   y_val)
test_ds  = ClauseDataset(X_test,  y_test)

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

# %% Cell 6 — Multi-head attention over GRU outputs
class MultiHeadSelfAttention(nn.Module):
    """
    Lightweight multi-head attention applied over GRU time-step outputs.

    Instead of a single scalar score per token, each head learns a
    different aspect of risk (e.g. one head for liability language,
    another for negation patterns).  The heads' context vectors are
    concatenated before the FC head.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale    = self.head_dim ** -0.5

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        def split_heads(t):
            return t.view(B, T, H, Dh).transpose(1, 2)   # (B, H, T, Dh)

        Q = split_heads(self.q(x))
        K = split_heads(self.k(x))
        V = split_heads(self.v(x))

        scores  = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        weights = torch.softmax(scores, dim=-1)                       # (B, H, T, T)

        ctx = torch.matmul(weights, V)                   # (B, H, T, Dh)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)          # (B, T, D)
        return self.out_proj(ctx)                         # (B, T, D)


class GRUClassifier(nn.Module):
    """
    Bidirectional multi-layer GRU with multi-head self-attention pooling.

    Architecture:
        Embedding → Dropout
          → BiGRU (2 layers) → Dropout
            → Multi-Head Self-Attention
              → Mean + Max pooling (concatenated) → LayerNorm
                → FC(1024→256) → GELU → Dropout
                  → FC(256→64) → GELU → Dropout
                    → FC(64→1) → Sigmoid

    Design notes
    ────────────
    • GRU vs LSTM: GRU has no separate cell state — it merges the
      "forget" and "input" gates into a single update gate.  This
      reduces parameter count (~25 % fewer weights per layer) while
      matching LSTM performance on most NLP tasks.

    • Mean + Max pooling: mean pooling captures the overall "tone" of
      a clause; max pooling retains the single most risk-salient
      feature per dimension.  Concatenating both gives richer
      representations than either alone.

    • Multi-head attention refines the GRU output before pooling,
      letting the model focus on contextually important tokens
      (e.g. "unlimited", "waive", "indemnify") from multiple angles.

    • GELU activation: smoother gradient flow than ReLU, common in
      modern NLP classifiers.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers, dropout, num_heads=4, bidirectional=True):
        super().__init__()

        self.num_directions = 2 if bidirectional else 1
        gru_out_dim         = hidden_dim * self.num_directions   # 512

        # ── Embedding ──────────────────────────────────────────
        self.embedding    = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # ── BiGRU ──────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.gru_dropout = nn.Dropout(dropout)

        # ── Multi-head attention ───────────────────────────────
        self.attention   = MultiHeadSelfAttention(gru_out_dim, num_heads)
        self.layer_norm  = nn.LayerNorm(gru_out_dim * 2)   # after concat pooling

        # ── Classifier head ────────────────────────────────────
        self.fc1      = nn.Linear(gru_out_dim * 2, 256)
        self.fc2      = nn.Linear(256, 64)
        self.fc3      = nn.Linear(64, 1)
        self.gelu     = nn.GELU()
        self.drop1    = nn.Dropout(dropout)
        self.drop2    = nn.Dropout(dropout / 2)   # lighter dropout near output

    def forward(self, x):
        # x: (B, T)

        emb = self.embed_dropout(self.embedding(x))  # (B, T, E)

        gru_out, _ = self.gru(emb)                   # (B, T, H*dirs)
        gru_out    = self.gru_dropout(gru_out)

        # Multi-head attention refines token representations
        attn_out = self.attention(gru_out)            # (B, T, H*dirs)

        # Mean + Max pooling over sequence dimension
        mean_pool = attn_out.mean(dim=1)              # (B, H*dirs)
        max_pool  = attn_out.max(dim=1).values        # (B, H*dirs)
        pooled    = torch.cat([mean_pool, max_pool], dim=1)  # (B, H*dirs*2)
        pooled    = self.layer_norm(pooled)

        # Classifier
        out = self.drop1(self.gelu(self.fc1(pooled)))  # (B, 256)
        out = self.drop2(self.gelu(self.fc2(out)))     # (B, 64)
        return self.fc3(out).squeeze(1)                # (B,)


model = GRUClassifier(
    vocab_size=VOCAB_SIZE + 2,   # +2 for PAD (0) and OOV (1)
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    num_heads=4,
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

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)

# Cosine annealing gives a smooth LR curve, often better for GRUs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5
)

print(f"✅ Optimizer : AdamW  (lr={LR}, wd={WEIGHT_DECAY})")
print(f"   Loss      : BCEWithLogitsLoss (pos_weight={pos_weight.item():.4f})")
print(f"   Scheduler : CosineAnnealingLR (T_max={EPOCHS}, eta_min=1e-5)")


# ============================================================
# TRAINING LOOP
# ============================================================

# %% Cell 8 — Epoch helper
def run_epoch(loader, model, criterion, optimizer=None, threshold=0.5):
    """One pass. Trains if optimizer is given, else evaluates."""
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


# %% Cell 9 — Main training loop with early stopping
history = {k: [] for k in [
    "train_loss", "val_loss",
    "train_f1",   "val_f1",
    "train_acc",  "val_acc",
    "train_auc",  "val_auc",
    "lr"
]}

best_val_f1  = -1.0
best_epoch   = -1
patience_ctr = 0
best_weights = None

print(f"\n{'='*75}")
print(f"  Training BiGRU — max {EPOCHS} epochs  |  early-stop patience={PATIENCE}")
print(f"{'='*75}")

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_prec, tr_rec, tr_f1, tr_auc, _, _, _ = \
        run_epoch(train_loader, model, criterion, optimizer)

    vl_loss, vl_acc, vl_prec, vl_rec, vl_f1, vl_auc, _, _, _ = \
        run_epoch(val_loader, model, criterion)

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_f1"].append(tr_f1)
    history["val_f1"].append(vl_f1)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)
    history["train_auc"].append(tr_auc)
    history["val_auc"].append(vl_auc)
    history["lr"].append(current_lr)

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
        f"  Ep {epoch:02d}/{EPOCHS}  lr={current_lr:.2e}"
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

# %% Cell 10 — Test-set evaluation with best weights
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

# %% Cell 11 — Training curves (4-panel, includes LR schedule)
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("BiGRU Training History", fontsize=14, fontweight="bold")
eps = range(1, len(history["train_loss"]) + 1)

panels = [
    (axes[0, 0], "Loss",    "train_loss", "val_loss"),
    (axes[0, 1], "F1 Score","train_f1",   "val_f1"),
    (axes[1, 0], "ROC-AUC", "train_auc",  "val_auc"),
]

for ax, title, tr_key, vl_key in panels:
    ax.plot(eps, history[tr_key], label="Train", linewidth=1.8)
    ax.plot(eps, history[vl_key], label="Val",   linewidth=1.8)
    ax.axvline(best_epoch, color="green", linestyle="--",
               alpha=0.7, label=f"Best (ep {best_epoch})")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

# Learning-rate schedule
axes[1, 1].plot(eps, history["lr"], color="darkorange", linewidth=1.8)
axes[1, 1].set_title("Learning Rate Schedule", fontweight="bold")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("LR")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(TRAINING_CURVES_PATH, dpi=150, bbox_inches="tight")
print(f"✅ Saved: {TRAINING_CURVES_PATH}")


# %% Cell 12 — Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=["Safe (0)", "Risky (1)"],
    yticklabels=["Safe (0)", "Risky (1)"],
    ax=ax
)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(
    f"Confusion Matrix — Test Set\n"
    f"Acc={te_acc:.3f}  F1={te_f1:.3f}  AUC={te_auc:.3f}"
)
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
        "num_heads"     : 4,
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