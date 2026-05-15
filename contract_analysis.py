# ============================================================
# LEGAL CONTRACT RISK ANALYSIS — FULL TRAINING PIPELINE
# ============================================================


# %% Cell 2 — Imports (all in one place, cleaner)

import os
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("✅ All imports done")


# %% Cell 3 — Load dataset
ds = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_clause_classification")
df = ds["train"].to_pandas()

print(f"Shape      : {df.shape}")
print(f"Columns    : {df.columns.tolist()}")
print(f"Null values:\n{df.isnull().sum()}")
print(df.head(3))


# %% Cell 4 — Understand labels
print(f"\nUnique labels: {df['label'].nunique()}")
print(df["label"].value_counts().to_string())


# %% Cell 5 — Map risk labels
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

counts = df["risk"].value_counts()
print(f"\nNot Risky : {counts[0]} ({counts[0] / len(df) * 100:.1f}%)")
print(f"Risky     : {counts[1]} ({counts[1] / len(df) * 100:.1f}%)")

plt.figure(figsize=(14, 4))
df["label"].value_counts().plot(kind="bar", color="steelblue")
plt.title("Clause Type Distribution")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig("label_distribution.png", dpi=150)
plt.show()
print("✅ Label distribution saved")


# %% Cell 6 — Analyse clause lengths BEFORE cleaning
df["word_count"] = df["clause"].apply(lambda x: len(str(x).split()))
print(f"\nClause length stats:")
print(df["word_count"].describe())
print(f"\n95th percentile : {df['word_count'].quantile(0.95):.0f} words")
print(f"99th percentile : {df['word_count'].quantile(0.99):.0f} words")

plt.figure(figsize=(10, 4))
df["word_count"].hist(bins=50, color="steelblue", edgecolor="white")
plt.axvline(
    df["word_count"].quantile(0.95),
    color="red",
    linestyle="--",
    label="95th percentile",
)
plt.title("Clause Length Distribution")
plt.xlabel("Word count")
plt.legend()
plt.tight_layout()
plt.savefig("clause_lengths.png", dpi=150)
plt.show()

# %% Cell 7 — Clean text
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

LEGAL_KEEP = {
    "not",
    "no",
    "shall",
    "must",
    "will",
    "may",
    "cannot",
    "never",
    "without",
    "unless",
    "except",
    "terminate",
    "liable",
    "liability",
    "indemnify",
    "waive",
    "void",
    "breach",
    "default",
    "obligation",
    "warranted",
    "only",
    "sole",
    "exclusive",
    "unlimited",
    "limited",
}


def clean_text(text):
    # 1. force string and lowercase
    text = str(text).lower()
    # 2. remove redaction markers like [REDACTED] or [***]
    text = re.sub(r"\[.*?\]", "", text)
    # 3. IMPROVEMENT: remove URLs — sometimes appear in contracts
    text = re.sub(r"http\S+|www\S+", "", text)
    # 4. IMPROVEMENT: expand common contractions
    text = text.replace("won't", "will not")
    text = text.replace("can't", "cannot")
    text = text.replace("n't", " not")
    # 5. remove punctuation and numbers — keep letters and spaces only
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # 6. collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    # 7. tokenize
    tokens = text.split()
    # 8. IMPROVEMENT: drop very short tokens (single letters are noise)
    tokens = [t for t in tokens if len(t) > 1]
    # 9. remove stopwords but keep legal words
    tokens = [t for t in tokens if t not in stop_words or t in LEGAL_KEEP]

    # 10. lemmatize — "terminating" → "terminate"
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


df["clean_clause"] = df["clause"].apply(clean_text)

before = len(df)
df = df[df["clean_clause"].str.strip().str.len() > 0]
print(f"\nDropped {before - len(df)} empty rows after cleaning")

df = df[df["clean_clause"].str.split().str.len() >= 2]
print(f"Rows remaining: {len(df)}")

# Verify cleaning worked
print("\n--- BEFORE ---")
print(df["clause"].iloc[0][:300])
print("\n--- AFTER ---")
print(df["clean_clause"].iloc[0][:300])


# %% Cell 8 — Handle class imbalance
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.array([0, 1]), y=df["risk"]
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights: {class_weight_dict}")


# %% Cell 9 — Train / Val / Test split
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["risk"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["risk"]
)

print(f"\nTrain : {len(train_df)} rows")
print(f"Val   : {len(val_df)} rows")
print(f"Test  : {len(test_df)} rows")

print(f"\nTrain risk ratio : {train_df['risk'].mean():.3f}")
print(f"Val   risk ratio : {val_df['risk'].mean():.3f}")
print(f"Test  risk ratio : {test_df['risk'].mean():.3f}")


# %% Cell 10 — Save splits
os.makedirs("splits", exist_ok=True)
train_df.to_csv("splits/train.csv", index=False)
val_df.to_csv("splits/val.csv", index=False)
test_df.to_csv("splits/test.csv", index=False)

np.save("splits/class_weights.npy", class_weights)

print("\n✅ Splits saved to splits/")
print("✅ Class weights saved")
print("\nNext step → open 02_tokenization.py")
print(f"Recommended MAX_LEN = {int(df['word_count'].quantile(0.95))}")
