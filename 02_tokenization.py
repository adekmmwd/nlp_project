# ============================================================
# TOKENIZATION — PYTORCH PIPELINE (LSTM/GRU + BERT)
# ============================================================
# Replaces TensorFlow/Keras tokenizer with pure PyTorch
# One framework for all three models
# ============================================================

# %% Cell 1 — Imports
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import AutoTokenizer

print(f"✅ Imports complete")
print(f"   PyTorch version : {torch.__version__}")
print(f"   CUDA available  : {torch.cuda.is_available()}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using device    : {DEVICE}")


# %% Cell 2 — Load splits
train_df = pd.read_csv("splits/train.csv")
val_df = pd.read_csv("splits/val.csv")
test_df = pd.read_csv("splits/test.csv")

train_texts = train_df["clean_clause"].astype(str).tolist()
val_texts = val_df["clean_clause"].astype(str).tolist()
test_texts = test_df["clean_clause"].astype(str).tolist()

y_train = train_df["risk"].values
y_val = val_df["risk"].values
y_test = test_df["risk"].values

print(f"\nTrain : {len(train_texts)} clauses")
print(f"Val   : {len(val_texts)} clauses")
print(f"Test  : {len(test_texts)} clauses")


# ============================================================
# PART A — LSTM / GRU TOKENIZATION (PyTorch vocab builder)
# ============================================================

# %% Cell 3 — Build vocabulary from scratch (replaces Keras Tokenizer)
MAX_LEN = 140  # 95th percentile from step 1 + small buffer
VOCAB_SIZE = 20000  # keep top 20k most frequent words

# Special tokens
PAD_TOKEN = "<PAD>"  # index 0 — used to fill short sequences
OOV_TOKEN = "<OOV>"  # index 1 — used for unseen words

# Count word frequencies — ONLY on training data
word_counts = Counter()
for text in train_texts:
    word_counts.update(text.split())

print(f"\nTotal unique words in training data: {len(word_counts)}")

# Build vocab: take top VOCAB_SIZE-2 words (leaving room for PAD and OOV)
most_common = word_counts.most_common(VOCAB_SIZE - 2)

# word → index mapping
word2idx = {PAD_TOKEN: 0, OOV_TOKEN: 1}
for word, _ in most_common:
    word2idx[word] = len(word2idx)

# index → word mapping (useful for debugging)
idx2word = {v: k for k, v in word2idx.items()}

actual_vocab_size = len(word2idx)
print(f"Vocabulary size (with PAD + OOV): {actual_vocab_size}")


# %% Cell 4 — Encode text to integer sequences
def encode_text(texts, word2idx, max_len):
    """
    Converts a list of text strings into a padded numpy array.
    - Unknown words → OOV index (1)
    - Short sequences → padded with 0 on the right
    - Long sequences → truncated on the right
    """
    oov_idx = word2idx[OOV_TOKEN]
    pad_idx = word2idx[PAD_TOKEN]
    encoded = []

    for text in texts:
        tokens = text.split()[:max_len]  # truncate
        indices = [word2idx.get(t, oov_idx) for t in tokens]
        # pad to max_len
        padded = indices + [pad_idx] * (max_len - len(indices))
        encoded.append(padded)

    return np.array(encoded, dtype=np.int64)


X_train_lstm = encode_text(train_texts, word2idx, MAX_LEN)
X_val_lstm = encode_text(val_texts, word2idx, MAX_LEN)
X_test_lstm = encode_text(test_texts, word2idx, MAX_LEN)

print(f"\nLSTM/GRU tensor shapes:")
print(f"X_train : {X_train_lstm.shape}")  # (8180, 140)
print(f"X_val   : {X_val_lstm.shape}")  # (1753, 140)
print(f"X_test  : {X_test_lstm.shape}")  # (1754, 140)


# %% Cell 5 — Check OOV rate
def oov_rate(encoded, oov_index=1):
    """
    How many tokens in the encoded array are OOV?
    Should be low — if test OOV is much higher than train,
    your splits may have domain mismatch.
    """
    total = encoded.size
    oov_count = np.sum(encoded == oov_index)
    return oov_count / total * 100


print(f"\nOOV rate — Train : {oov_rate(X_train_lstm):.2f}%")
print(f"OOV rate — Val   : {oov_rate(X_val_lstm):.2f}%")
print(f"OOV rate — Test  : {oov_rate(X_test_lstm):.2f}%")
# All should be under 5% — if test is much higher than train, flag it


# %% Cell 6 — PyTorch Dataset class for LSTM/GRU
class ClauseDatasetLSTM(Dataset):
    """
    Wraps the encoded numpy arrays into a PyTorch Dataset.
    DataLoader uses this to create batches automatically.
    """

    def __init__(self, X, y):
        # convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(
            X, dtype=torch.long
        )  # long = int64, needed for embedding layer
        self.y = torch.tensor(y, dtype=torch.float32)  # float32 needed for BCELoss

    def __len__(self):
        return len(self.y)  # how many samples total

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # return one sample at a time


# Create datasets
train_dataset_lstm = ClauseDatasetLSTM(X_train_lstm, y_train)
val_dataset_lstm = ClauseDatasetLSTM(X_val_lstm, y_val)
test_dataset_lstm = ClauseDatasetLSTM(X_test_lstm, y_test)

# Create DataLoaders — handle batching and shuffling automatically
BATCH_SIZE = 32

train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=BATCH_SIZE, shuffle=True)
val_loader_lstm = DataLoader(val_dataset_lstm, batch_size=BATCH_SIZE, shuffle=False)
test_loader_lstm = DataLoader(test_dataset_lstm, batch_size=BATCH_SIZE, shuffle=False)

# Verify one batch
X_batch, y_batch = next(iter(train_loader_lstm))
print(f"\nSample batch shapes:")
print(f"X_batch : {X_batch.shape}")  # (32, 140)
print(f"y_batch : {y_batch.shape}")  # (32,)


# ============================================================
# PART B — BERT TOKENIZATION (AutoTokenizer)
# ============================================================

# %% Cell 7 — Load Legal-BERT tokenizer
BERT_MODEL = "nlpaueb/legal-bert-base-uncased"
BERT_MAX_LEN = 512  # BERT hard limit — cannot go higher

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
print(f"\n✅ Loaded BERT tokenizer: {BERT_MODEL}")
print(f"   BERT vocab size: {bert_tokenizer.vocab_size}")

# Show what BERT does to a legal clause
sample = "vendor shall indemnify and hold harmless all unlimited losses"
tokens = bert_tokenizer.tokenize(sample)
print(f"\nSample clause    : {sample}")
print(f"BERT tokens      : {tokens}")
print(f"Note how 'indemnify' and 'unlimited' may be split into subwords")


# %% Cell 8 — Encode for BERT
def encode_bert(texts, tokenizer, max_len):
    """
    Converts texts to BERT input format.
    Returns input_ids, attention_mask as numpy arrays.

    input_ids      → integer ID of each token in BERT's 30k vocabulary
    attention_mask → 1 for real tokens, 0 for padding —
                     tells BERT to ignore padding positions
    """
    encoding = tokenizer(
        texts,
        truncation=True,  # cut sequences longer than max_len
        padding="max_length",  # pad all sequences to exactly max_len
        max_length=max_len,
        return_tensors="np",  # return numpy arrays (easier to save)
    )
    return encoding["input_ids"], encoding["attention_mask"]


print("\nEncoding BERT splits (this may take a minute)...")
X_train_bert_ids, X_train_bert_mask = encode_bert(
    train_texts, bert_tokenizer, BERT_MAX_LEN
)
X_val_bert_ids, X_val_bert_mask = encode_bert(val_texts, bert_tokenizer, BERT_MAX_LEN)
X_test_bert_ids, X_test_bert_mask = encode_bert(
    test_texts, bert_tokenizer, BERT_MAX_LEN
)

print(f"\nBERT tensor shapes:")
print(f"Train input_ids   : {X_train_bert_ids.shape}")  # (8180, 512)
print(f"Train attn_mask   : {X_train_bert_mask.shape}")  # (8180, 512)
print(f"Val   input_ids   : {X_val_bert_ids.shape}")
print(f"Test  input_ids   : {X_test_bert_ids.shape}")


# %% Cell 9 — PyTorch Dataset for BERT
class ClauseDatasetBERT(Dataset):
    """
    Wraps BERT inputs into a PyTorch Dataset.
    BERT needs three things per sample:
    - input_ids      (the tokenized clause)
    - attention_mask (which positions are real vs padding)
    - label          (0 or 1)
    """

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


train_dataset_bert = ClauseDatasetBERT(X_train_bert_ids, X_train_bert_mask, y_train)
val_dataset_bert = ClauseDatasetBERT(X_val_bert_ids, X_val_bert_mask, y_val)
test_dataset_bert = ClauseDatasetBERT(X_test_bert_ids, X_test_bert_mask, y_test)

# BERT uses smaller batch sizes — it is a much larger model
BERT_BATCH_SIZE = 16

train_loader_bert = DataLoader(
    train_dataset_bert, batch_size=BERT_BATCH_SIZE, shuffle=True
)
val_loader_bert = DataLoader(
    val_dataset_bert, batch_size=BERT_BATCH_SIZE, shuffle=False
)
test_loader_bert = DataLoader(
    test_dataset_bert, batch_size=BERT_BATCH_SIZE, shuffle=False
)

# Verify one BERT batch
sample_batch = next(iter(train_loader_bert))
print(f"\nSample BERT batch shapes:")
print(f"input_ids      : {sample_batch['input_ids'].shape}")  # (16, 512)
print(f"attention_mask : {sample_batch['attention_mask'].shape}")  # (16, 512)
print(f"labels         : {sample_batch['labels'].shape}")  # (16,)


# ============================================================
# PART C — SAVE EVERYTHING
# ============================================================

# %% Cell 10 — Save all artifacts
os.makedirs("tokens", exist_ok=True)

# LSTM/GRU arrays
np.save("tokens/X_train_lstm.npy", X_train_lstm)
np.save("tokens/X_val_lstm.npy", X_val_lstm)
np.save("tokens/X_test_lstm.npy", X_test_lstm)

# BERT arrays
np.save("tokens/X_train_bert_ids.npy", X_train_bert_ids)
np.save("tokens/X_train_bert_mask.npy", X_train_bert_mask)
np.save("tokens/X_val_bert_ids.npy", X_val_bert_ids)
np.save("tokens/X_val_bert_mask.npy", X_val_bert_mask)
np.save("tokens/X_test_bert_ids.npy", X_test_bert_ids)
np.save("tokens/X_test_bert_mask.npy", X_test_bert_mask)

# Labels (shared by both)
np.save("tokens/y_train.npy", y_train)
np.save("tokens/y_val.npy", y_val)
np.save("tokens/y_test.npy", y_test)

# Vocabulary (needed by LSTM model to build embedding layer)
with open("tokens/word2idx.pkl", "wb") as f:
    pickle.dump(word2idx, f)
with open("tokens/idx2word.pkl", "wb") as f:
    pickle.dump(idx2word, f)

# Config (so model scripts don't hardcode these)
config = {
    "vocab_size": actual_vocab_size,
    "max_len": MAX_LEN,
    "bert_max_len": BERT_MAX_LEN,
    "batch_size": BATCH_SIZE,
    "bert_batch_size": BERT_BATCH_SIZE,
    "bert_model": BERT_MODEL,
}
import json

with open("tokens/config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n✅ All artifacts saved to tokens/")
print("   LSTM arrays  → X_train_lstm.npy, X_val_lstm.npy, X_test_lstm.npy")
print("   BERT arrays  → X_*_bert_ids.npy, X_*_bert_mask.npy")
print("   Labels       → y_train.npy, y_val.npy, y_test.npy")
print("   Vocabulary   → word2idx.pkl, idx2word.pkl")
print("   Config       → config.json")
