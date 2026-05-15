# ============================================================
# 05_lbert.py
# BERT — BINARY RISK CLASSIFICATION
# bert-base-uncased | raw clause text | no preprocessing
# ============================================================

# %% Imports
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


# ============================================================
# SEED — makes every run identical
# ============================================================
SEED = 42
IS_ROCM = getattr(torch.version, "hip", None) is not None
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = not IS_ROCM
torch.backends.cudnn.benchmark = IS_ROCM
print(f"✅ Seed set to {SEED}")
if IS_ROCM:
    print("ROCm detected : using non-deterministic cuDNN kernels for stability")


# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    # ROCm quirk — absorb the first CUDA call error
    print("\nWarming up GPU...")
    try:
        torch.tensor([1.0]).cuda()
    except RuntimeError:
        pass
    x = torch.tensor([1.0]).cuda()
    print(f"GPU ready    : {x.device}")

    # ROCm can produce non-finite values with fused SDPA kernels on some cards.
    # Keep BERT on the stable math/eager attention path.
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        print("Attention    : eager/math SDPA")


# ============================================================
# HYPERPARAMETERS — all in one place
# ============================================================
BERT_MODEL = "nlpaueb/legal-bert-base-uncased"
SAFE_MODEL_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "models--nlpaueb--legal-bert-base-uncased/"
    "snapshots/418b58a010be042386a2ad50f350a9881ef3c578"
)
SAFE_MODEL_REVISION = "refs/pr/1"
MAX_LEN = 140  # 95th percentile of clause lengths
BATCH_SIZE = 16  # safe for 16GB VRAM
EPOCHS = 8  # BERT converges in 3-4 epochs
LR = 2e-5  # standard BERT fine-tuning LR
BERT_LR = 5e-6 if IS_ROCM else LR  # conservative LR for partial ROCm fine-tuning
WARMUP_RATIO = 0.1  # 10% of steps for LR warmup
CLIP_GRAD = 1.0  # gradient clipping threshold
ATTN_IMPLEMENTATION = "eager"  # stable on ROCm; avoids fused SDPA NaNs
ADAM_EPS = 1e-6  # slightly more stable than 1e-8 on ROCm
ADAM_FOREACH = False if IS_ROCM else None
MAX_GRAD_WARNINGS = 3
UNFREEZE_LAST_BERT_LAYERS = None
TRAIN_BERT_POOLER = IS_ROCM
CHECK_PARAM_UPDATES = IS_ROCM
KEEP_BERT_EVAL_DURING_TRAIN = IS_ROCM

MODEL_NAME = "legal_bert_109m_trained"
MODELS_DIR = "models"
RESULTS_DIR = "results"
PREDICTIONS_DIR = "predictions"
MODEL_CHECKPOINT_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}_best.pt")
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_confusion_matrix.png")
PREDS_PATH = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_preds.npy")
PROBS_PATH = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_probs.npy")
LABELS_PATH = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_labels.npy")
METRICS_PATH = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_metrics.json")

print(f"\nHyperparameters:")
print(f"  Model      : {BERT_MODEL}")
print(f"  Safe load  : safetensors")
print(f"  Save tag   : {MODEL_NAME}")
print(f"  MAX_LEN    : {MAX_LEN}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Epochs     : {EPOCHS}")
print(f"  Head LR    : {LR}")
print(f"  BERT LR    : {BERT_LR}")
print(f"  Attention  : {ATTN_IMPLEMENTATION}")
print(f"  Adam eps   : {ADAM_EPS}")
if UNFREEZE_LAST_BERT_LAYERS is None:
    print("  Train BERT : all layers")
else:
    print(f"  Train BERT : last {UNFREEZE_LAST_BERT_LAYERS} layers + pooler")
print(f"  BERT dropout off: {KEEP_BERT_EVAL_DURING_TRAIN}")


# ============================================================
# LOAD DATASET
# ============================================================
print("\nLoading dataset...")
ds = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_clause_classification")
df = ds["train"].to_pandas()
print(f"Dataset shape : {df.shape}")


# ============================================================
# RISK LABELS
# ============================================================
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

df["risk"] = df["label"].apply(lambda x: 1 if x in RISKY_LABELS else 0)

# raw text only — no preprocessing, BERT handles it
df["text"] = df["clause"].astype(str)
df = df[df["text"].str.strip().str.len() > 0]

counts = df["risk"].value_counts()
print(f"\nNot Risky : {counts[0]} ({counts[0] / len(df) * 100:.1f}%)")
print(f"Risky     : {counts[1]} ({counts[1] / len(df) * 100:.1f}%)")


# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df["risk"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["risk"]
)

print(f"\nTrain : {len(train_df)} | risk ratio: {train_df['risk'].mean():.3f}")
print(f"Val   : {len(val_df)} | risk ratio: {val_df['risk'].mean():.3f}")
print(f"Test  : {len(test_df)} | risk ratio: {test_df['risk'].mean():.3f}")


# ============================================================
# CLASS WEIGHTS
# ============================================================
class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=np.array([0, 1]), y=train_df["risk"].values
)
class_weights = torch.tensor(class_weights_arr, dtype=torch.float32).to(DEVICE)
print(f"\nClass weights : {class_weights_arr}")
print(
    f"  Model penalised {class_weights_arr[1] / class_weights_arr[0]:.1f}x more for missing a risky clause"
)


# ============================================================
# TOKENIZER
# ============================================================
print(f"\nLoading tokenizer: {BERT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
print(f"✅ Tokenizer loaded | vocab size: {tokenizer.vocab_size:,}")


# ============================================================
# TOKENIZATION
# ============================================================
def encode_texts(texts):
    """
    Converts raw clause text into BERT input format.
    Returns input_ids and attention_mask as tensors.
    padding='max_length' pads all sequences to exactly MAX_LEN.
    truncation=True cuts sequences longer than MAX_LEN.
    """
    encoding = tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]


print("\nTokenizing datasets...")
X_train_ids, X_train_mask = encode_texts(train_df["text"])
X_val_ids, X_val_mask = encode_texts(val_df["text"])
X_test_ids, X_test_mask = encode_texts(test_df["text"])

y_train = torch.tensor(train_df["risk"].values, dtype=torch.long)
y_val = torch.tensor(val_df["risk"].values, dtype=torch.long)
y_test = torch.tensor(test_df["risk"].values, dtype=torch.long)

print(f"✅ Tokenization complete")
print(f"   Train : {X_train_ids.shape}")
print(f"   Val   : {X_val_ids.shape}")
print(f"   Test  : {X_test_ids.shape}")


# ============================================================
# DATASET CLASS
# ============================================================
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


# ============================================================
# DATASETS + DATALOADERS
# ============================================================
train_dataset = ContractDataset(X_train_ids, X_train_mask, y_train)
val_dataset = ContractDataset(X_val_ids, X_val_mask, y_val)
test_dataset = ContractDataset(X_test_ids, X_test_mask, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n✅ DataLoaders ready")
print(f"   Train batches : {len(train_loader)}")
print(f"   Val batches   : {len(val_loader)}")
print(f"   Test batches  : {len(test_loader)}")


# ============================================================
# LOAD MODEL
# ============================================================
print(f"\nLoading {BERT_MODEL}...")
model_source = BERT_MODEL
model_kwargs = {}

if os.path.exists(os.path.join(SAFE_MODEL_SNAPSHOT, "model.safetensors")):
    model_source = SAFE_MODEL_SNAPSHOT
    print(f"Using local safetensors snapshot: {SAFE_MODEL_SNAPSHOT}")
else:
    model_kwargs["revision"] = SAFE_MODEL_REVISION
    print(f"Using safetensors revision: {SAFE_MODEL_REVISION}")

model_config = AutoConfig.from_pretrained(BERT_MODEL, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(
    model_source,
    config=model_config,
    ignore_mismatched_sizes=True,
    attn_implementation=ATTN_IMPLEMENTATION,
    use_safetensors=True,
    **model_kwargs,
)
model = model.to(DEVICE)

if UNFREEZE_LAST_BERT_LAYERS is not None:
    for param in model.base_model.parameters():
        param.requires_grad = False

    encoder_layers = model.base_model.encoder.layer
    for layer in encoder_layers[-UNFREEZE_LAST_BERT_LAYERS:]:
        for param in layer.parameters():
            param.requires_grad = True

    if TRAIN_BERT_POOLER and getattr(model.base_model, "pooler", None) is not None:
        for param in model.base_model.pooler.parameters():
            param.requires_grad = True

    print(
        f"Training mode : last {UNFREEZE_LAST_BERT_LAYERS} BERT layers "
        "+ pooler + classifier head"
    )

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model loaded")
print(f"   Total params     : {total_params:,}")
print(f"   Trainable params : {trainable_params:,}")


# ============================================================
# LOSS — weighted to handle class imbalance
# ============================================================
def weighted_cross_entropy_loss(logits, labels):
    log_probs = torch.log_softmax(logits.float(), dim=1)
    sample_weights = class_weights[labels]
    nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    return (nll * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)


# ============================================================
# OPTIMIZER — AdamW with weight decay
# ============================================================
base_param_ids = {id(param) for param in model.base_model.parameters()}
bert_params = [param for param in model.base_model.parameters() if param.requires_grad]
head_params = [
    param
    for param in model.parameters()
    if param.requires_grad and id(param) not in base_param_ids
]

optimizer_groups = []
if bert_params:
    optimizer_groups.append({"params": bert_params, "lr": BERT_LR})
if head_params:
    optimizer_groups.append({"params": head_params, "lr": LR})

optimizer = AdamW(
    optimizer_groups,
    weight_decay=0.01,
    eps=ADAM_EPS,
    foreach=ADAM_FOREACH,
)


# ============================================================
# SCHEDULER — warmup then linear decay
# ============================================================
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

print(f"\nOptimizer + Scheduler ready")
print(f"   Total steps  : {total_steps:,}")
print(f"   Warmup steps : {warmup_steps:,}")


# ============================================================
# SANITY CHECK — make sure loss is not NaN before training
# ============================================================
print("\nSanity checking loss on one batch...")
model.train()
_batch = next(iter(train_loader))
_ids = _batch["input_ids"].to(DEVICE)
_mask = _batch["attention_mask"].to(DEVICE)
_labels = _batch["labels"].to(DEVICE)
_outputs = model(input_ids=_ids, attention_mask=_mask)
_loss = weighted_cross_entropy_loss(_outputs.logits, _labels)
print(f"Sanity loss : {_loss.item():.4f}")
assert torch.isfinite(_outputs.logits).all(), "❌ Logits are not finite"
assert torch.isfinite(_loss), "❌ Loss is not finite — check class weights dtype"
print("✅ Loss is valid\n")


# ============================================================
# NUMERIC HELPERS
# ============================================================
def require_finite_tensor(name, tensor, batch_idx=None):
    if torch.isfinite(tensor).all():
        return

    finite_values = tensor.detach()[torch.isfinite(tensor.detach())]
    location = f" at batch {batch_idx}" if batch_idx is not None else ""

    if finite_values.numel() == 0:
        stats = "no finite values"
    else:
        stats = (
            f"finite range "
            f"[{finite_values.min().item():.4g}, {finite_values.max().item():.4g}]"
        )

    raise FloatingPointError(f"{name} contains NaN/Inf{location} ({stats})")


def positive_class_probs(logits, batch_idx=None):
    require_finite_tensor("Logits", logits, batch_idx)
    probs = torch.softmax(logits.detach().float(), dim=1)[:, 1]
    require_finite_tensor("Probabilities", probs, batch_idx)
    return probs


def safe_roc_auc(labels, probs, split_name):
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=np.float64)

    finite_mask = np.isfinite(probs)
    if not finite_mask.all():
        bad_count = int((~finite_mask).sum())
        raise FloatingPointError(
            f"{split_name} probabilities contain {bad_count} NaN/Inf values"
        )

    if np.unique(labels).size < 2:
        return float("nan")

    return roc_auc_score(labels, probs)


def sanitize_nonfinite_gradients(model):
    bad_tensors = 0
    bad_values = 0
    examples = []

    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue

        finite_mask = torch.isfinite(grad)
        if finite_mask.all():
            continue

        nonfinite_count = int((~finite_mask).sum().item())
        bad_tensors += 1
        bad_values += nonfinite_count

        if len(examples) < 5:
            examples.append(f"{name}: {nonfinite_count}/{grad.numel()} non-finite")

        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    return bad_tensors, bad_values, examples


def clone_trainable_parameters(model):
    return [
        (param, param.detach().clone())
        for param in model.parameters()
        if param.requires_grad
    ]


def restore_trainable_parameters(snapshot):
    with torch.no_grad():
        for param, saved in snapshot:
            param.copy_(saved)


def first_nonfinite_parameter(model, trainable_only=False):
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        if torch.isfinite(param).all():
            continue

        nonfinite_count = int((~torch.isfinite(param)).sum().item())
        return name, nonfinite_count, param.numel()

    return None


def clear_optimizer_state(optimizer):
    optimizer.state.clear()


# ============================================================
# TRAIN FUNCTION
# ============================================================
def train_epoch(model, loader):
    model.train()
    if KEEP_BERT_EVAL_DURING_TRAIN:
        # Keep BERT dropout disabled on ROCm while still training its weights.
        model.base_model.eval()

    total_loss = 0
    successful_batches = 0
    skipped_batches = 0
    nonfinite_grad_batches = 0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(pbar, start=1):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        require_finite_tensor("Training logits", logits, batch_idx)
        loss = weighted_cross_entropy_loss(logits, labels)
        require_finite_tensor("Training loss", loss, batch_idx)

        probs = positive_class_probs(logits, batch_idx)
        preds = torch.argmax(logits.detach(), dim=1)

        loss.backward()

        bad_tensors, bad_values, examples = sanitize_nonfinite_gradients(model)
        batch_had_grad_fix = bad_tensors > 0
        if bad_tensors:
            nonfinite_grad_batches += 1
            if nonfinite_grad_batches <= MAX_GRAD_WARNINGS:
                print(
                    f"\n⚠️  Sanitized {bad_values} non-finite gradient values "
                    f"in {bad_tensors} tensors at batch {batch_idx}"
                )
                for example in examples:
                    print(f"   {example}")

        # gradient clipping — prevents exploding gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CLIP_GRAD, error_if_nonfinite=False
        )
        if not torch.isfinite(grad_norm):
            skipped_batches += 1
            if not batch_had_grad_fix:
                nonfinite_grad_batches += 1
            if nonfinite_grad_batches <= MAX_GRAD_WARNINGS:
                print(
                    f"\n⚠️  Skipping batch {batch_idx}: "
                    f"gradient norm is {grad_norm.item()}"
                )
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "skipped": skipped_batches})
            continue

        param_snapshot = (
            clone_trainable_parameters(model) if CHECK_PARAM_UPDATES else None
        )
        optimizer.step()

        bad_param = first_nonfinite_parameter(model, trainable_only=True)
        if bad_param is not None:
            skipped_batches += 1
            name, bad_count, total_count = bad_param
            if param_snapshot is not None:
                restore_trainable_parameters(param_snapshot)
                clear_optimizer_state(optimizer)
            print(
                f"\n⚠️  Skipping batch {batch_idx}: optimizer produced "
                f"{bad_count}/{total_count} non-finite values in {name}"
            )
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "skipped": skipped_batches})
            continue

        scheduler.step()

        total_loss += loss.item()
        successful_batches += 1
        postfix = {"loss": f"{loss.item():.4f}"}
        if nonfinite_grad_batches:
            postfix["grad_fix"] = nonfinite_grad_batches
        if skipped_batches:
            postfix["skipped"] = skipped_batches
        pbar.set_postfix(postfix)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    if successful_batches == 0:
        raise FloatingPointError("No training batches completed successfully")

    if nonfinite_grad_batches:
        print(
            f"⚠️  Sanitized non-finite gradients on "
            f"{nonfinite_grad_batches}/{successful_batches} training batches"
        )
    if skipped_batches:
        print(f"⚠️  Skipped {skipped_batches} batches with unusable gradient norms")

    avg_loss = total_loss / successful_batches
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = safe_roc_auc(all_labels, all_probs, "Training")

    return avg_loss, acc, precision, recall, f1, roc_auc


# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate(model, loader):
    model.eval()
    bad_param = first_nonfinite_parameter(model)
    if bad_param is not None:
        name, bad_count, total_count = bad_param
        raise FloatingPointError(
            f"Model parameter {name} contains {bad_count}/{total_count} "
            "NaN/Inf values before evaluation"
        )

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False)

        for batch_idx, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            require_finite_tensor("Evaluation logits", logits, batch_idx)
            loss = weighted_cross_entropy_loss(logits, labels)
            require_finite_tensor("Evaluation loss", loss, batch_idx)

            total_loss += loss.item()

            probs = positive_class_probs(logits, batch_idx)
            preds = torch.argmax(logits.detach(), dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = safe_roc_auc(all_labels, all_probs, "Evaluation")

    return (
        avg_loss,
        acc,
        precision,
        recall,
        f1,
        roc_auc,
        all_preds,
        all_labels,
        all_probs,
    )


# ============================================================
# TRAINING LOOP
# ============================================================
print("🚀 Starting training...\n")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

best_val_f1 = -1.0
history = []

for epoch in range(EPOCHS):
    print(f"\n{'=' * 55}")
    print(f"  EPOCH {epoch + 1} / {EPOCHS}")
    print(f"{'=' * 55}")

    train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_epoch(
        model, train_loader
    )

    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, _, _, _ = evaluate(
        model, val_loader
    )

    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "train_auc": train_auc,
            "val_auc": val_auc,
        }
    )

    # print results table
    print(
        f"\n{'':>8} {'Loss':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}"
    )
    print(
        f"{'Train':>8} {train_loss:>8.4f} {train_acc:>8.4f} {train_prec:>8.4f} {train_rec:>8.4f} {train_f1:>8.4f} {train_auc:>8.4f}"
    )
    print(
        f"{'Val':>8} {val_loss:>8.4f} {val_acc:>8.4f} {val_prec:>8.4f} {val_rec:>8.4f} {val_f1:>8.4f} {val_auc:>8.4f}"
    )

    # save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
        print(f"\n  ✅ Best model saved (val F1={val_f1:.4f})")


# ============================================================
# PLOT TRAINING HISTORY
# ============================================================
epochs_range = [h["epoch"] for h in history]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(
    epochs_range, [h["train_loss"] for h in history], label="Train", marker="o"
)
axes[0].plot(epochs_range, [h["val_loss"] for h in history], label="Val", marker="o")
axes[0].set_title("Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    epochs_range, [h["train_f1"] for h in history], label="Train F1", marker="o"
)
axes[1].plot(epochs_range, [h["val_f1"] for h in history], label="Val F1", marker="o")
axes[1].set_title("F1 Score per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("F1")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"{MODEL_NAME} - Training History", fontsize=13)
plt.tight_layout()
plt.savefig(TRAINING_HISTORY_PATH, dpi=150)
plt.close()
print("✅ Training history saved")


# ============================================================
# LOAD BEST MODEL AND TEST
# ============================================================
print("\nLoading best model for final test evaluation...")
model.load_state_dict(
    torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
)

(
    test_loss,
    test_acc,
    test_prec,
    test_rec,
    test_f1,
    test_auc,
    test_preds,
    test_labels,
    test_probs,
) = evaluate(model, test_loader)

print(f"\n{'=' * 55}")
print(f"  FINAL TEST RESULTS — {BERT_MODEL}")
print(f"{'=' * 55}")
print(f"  Accuracy  : {test_acc:.4f}")
print(f"  Precision : {test_prec:.4f}")
print(f"  Recall    : {test_rec:.4f}")
print(f"  F1 Score  : {test_f1:.4f}")
print(f"  ROC-AUC   : {test_auc:.4f}")

print(f"\n{'=' * 55}")
print("  CLASSIFICATION REPORT")
print(f"{'=' * 55}")
print(
    classification_report(
        test_labels, test_preds, target_names=["Not Risky", "Risky"], zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================
cm = confusion_matrix(test_labels, test_preds)

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
plt.title(f"{BERT_MODEL} — Confusion Matrix", fontsize=13)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
plt.close()
print("✅ Confusion matrix saved")


# ============================================================
# SAVE PREDICTIONS FOR 06_evaluate.py
# ============================================================
np.save(PREDS_PATH, np.array(test_preds))
np.save(PROBS_PATH, np.array(test_probs))
np.save(LABELS_PATH, np.array(test_labels))

metrics = {
    "model": MODEL_NAME,
    "base_model": BERT_MODEL,
    "accuracy": round(test_acc, 4),
    "precision": round(test_prec, 4),
    "recall": round(test_rec, 4),
    "f1": round(test_f1, 4),
    "roc_auc": round(test_auc, 4),
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Predictions saved to predictions/")
print(f"{'=' * 55}")
print(f"  SUMMARY")
print(f"{'=' * 55}")
print(f"  Best val F1      : {best_val_f1:.4f}")
print(f"  Test F1          : {test_f1:.4f}")
print(f"  Test ROC-AUC     : {test_auc:.4f}")
print(f"  Model saved to   : {MODEL_CHECKPOINT_PATH}")
print(f"  Predictions at   : {PREDS_PATH}")
print(f"\nNext step → 06_evaluate.py")
