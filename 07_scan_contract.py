# ============================================================
# 07_scan_contract.py
# PDF CONTRACT RISK SCANNER
# ============================================================
# Takes any PDF contract, applies the same NLP pipeline
# from 01_preprocessing.py, runs the trained BERT model,
# and outputs all risky sections with confidence scores.
#
# Usage:
#   python 07_scan_contract.py --pdf path/to/contract.pdf
#   python 07_scan_contract.py --pdf path/to/contract.pdf --threshold 0.7
# ============================================================

import os
import re
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import nltk
import numpy as np
import torch
import pdfplumber
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)


# ============================================================
# ARGUMENT PARSER
# ============================================================
parser = argparse.ArgumentParser(description="Scan a PDF contract for risky clauses")
parser.add_argument("--pdf",       type=str, required=True,  help="Path to the PDF contract")
parser.add_argument("--model",     type=str, default="models/bert_best.pt", help="Path to trained model weights")
parser.add_argument("--threshold", type=float, default=0.5,  help="Risk probability threshold (default 0.5)")
parser.add_argument("--output",    type=str, default="results/scan_report.txt", help="Where to save the report")
args = parser.parse_args()


# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {DEVICE}")

if torch.cuda.is_available():
    try: torch.tensor([1.0]).cuda()
    except RuntimeError: pass

print(f"PDF          : {args.pdf}")
print(f"Model        : {args.model}")
print(f"Threshold    : {args.threshold}")


# ============================================================
# SAME NLP SETTINGS AS 01_preprocessing.py
# MUST be identical — if cleaning differs from training
# the model sees inputs it was not trained on
# ============================================================
BERT_MODEL = "bert-base-uncased"
MAX_LEN    = 140

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# identical to 01_preprocessing.py
LEGAL_KEEP = {
    "not", "no", "shall", "must", "will", "may", "cannot",
    "never", "without", "unless", "except", "terminate",
    "liable", "liability", "indemnify", "waive", "void",
    "breach", "default", "obligation", "warranted", "only",
    "sole", "exclusive", "unlimited", "limited"
}

def clean_text(text):
    """
    Exact same function as 01_preprocessing.py.
    Must stay identical — any difference breaks the pipeline.
    """
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)           # remove [REDACTED]
    text = re.sub(r"http\S+|www\S+", "", text)    # remove URLs
    text = text.replace("won't",  "will not")
    text = text.replace("can't",  "cannot")
    text = text.replace("n't",    " not")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)      # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [t for t in tokens if t not in stop_words or t in LEGAL_KEEP]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ============================================================
# STEP 1 — EXTRACT TEXT FROM PDF
# ============================================================
print(f"\n{'='*55}")
print(f"  STEP 1 — Extracting text from PDF")
print(f"{'='*55}")

if not os.path.exists(args.pdf):
    print(f"❌ PDF not found: {args.pdf}")
    sys.exit(1)

pages_text = []
with pdfplumber.open(args.pdf) as pdf:
    total_pages = len(pdf.pages)
    print(f"Total pages : {total_pages}")

    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_text.append({
                "page"  : i + 1,
                "text"  : text.strip()
            })

print(f"Pages with text : {len(pages_text)} / {total_pages}")

if len(pages_text) == 0:
    print("❌ No text extracted — PDF may be scanned/image-based")
    print("   Try OCR first: pip install pytesseract")
    sys.exit(1)

# join all pages into one document
full_text = "\n".join([p["text"] for p in pages_text])
print(f"Total characters extracted : {len(full_text):,}")


# ============================================================
# STEP 2 — SPLIT INTO CLAUSE-SIZED CHUNKS
# ============================================================
print(f"\n{'='*55}")
print(f"  STEP 2 — Splitting into clause-sized chunks")
print(f"{'='*55}")

def split_into_chunks(text, chunk_size=3):
    """
    Splits full contract text into chunks of chunk_size sentences.
    chunk_size=3 matches the average clause length in training data.
    Each chunk gets its own risk prediction.
    """
    # split into sentences on period, newline, or section markers
    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i : i + chunk_size])
        if len(chunk.split()) >= 5:   # skip very short chunks
            chunks.append({
                "chunk_id"    : len(chunks),
                "raw_text"    : chunk,
                "approx_page" : pages_text[min(i // 10, len(pages_text)-1)]["page"]
            })

    return chunks

chunks = split_into_chunks(full_text, chunk_size=3)
print(f"Total chunks : {len(chunks)}")
print(f"Sample chunk : {chunks[0]['raw_text'][:200]}...")


# ============================================================
# STEP 3 — APPLY SAME NLP PREPROCESSING AS TRAINING
# ============================================================
print(f"\n{'='*55}")
print(f"  STEP 3 — Applying NLP preprocessing")
print(f"{'='*55}")

for chunk in chunks:
    chunk["clean_text"] = clean_text(chunk["raw_text"])

# drop chunks that became empty after cleaning
before = len(chunks)
chunks = [c for c in chunks if len(c["clean_text"].split()) >= 2]
print(f"Chunks after cleaning : {len(chunks)} (dropped {before - len(chunks)} empty)")
print(f"Sample clean : {chunks[0]['clean_text'][:200]}...")


# ============================================================
# STEP 4 — TOKENIZE WITH BERT TOKENIZER
# ============================================================
print(f"\n{'='*55}")
print(f"  STEP 4 — Tokenizing with BERT tokenizer")
print(f"{'='*55}")

print(f"Loading tokenizer: {BERT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

def tokenize_chunks(chunks, tokenizer, max_len, batch_size=64):
    """
    Tokenizes chunks in batches.
    Uses clean_text (preprocessed) — same as training.
    Returns input_ids and attention_mask tensors.
    """
    all_ids  = []
    all_mask = []
    texts    = [c["clean_text"] for c in chunks]

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        all_ids.append(encoded["input_ids"])
        all_mask.append(encoded["attention_mask"])

    return torch.cat(all_ids, dim=0), torch.cat(all_mask, dim=0)

input_ids, attention_mask = tokenize_chunks(chunks, tokenizer, MAX_LEN)
print(f"Tokenized shape : {input_ids.shape}")   # (num_chunks, 140)


# ============================================================
# STEP 5 — LOAD TRAINED MODEL
# ============================================================
print(f"\n{'='*55}")
print(f"  STEP 5 — Loading trained model")
print(f"{'='*55}")

if not os.path.exists(args.model):
    print(f"❌ Model not found: {args.model}")
    print("   Run 05_lbert.py first to train and save the model")
    sys.exit(1)

model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL,
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.load_state_dict(
    torch.load(args.model, map_location=DEVICE, weights_only=True)
)
model = model.to(DEVICE)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model loaded : {total_params:,} parameters")


# ============================================================
# STEP 6 — PREDICT EACH CHUNK
# ============================================================
print(f"\n{'='*55}")
print(f"  STEP 6 — Predicting risk for each chunk")
print(f"{'='*55}")

BATCH_SIZE = 32
all_probs  = []
all_preds  = []

with torch.no_grad():
    pbar = tqdm(range(0, len(chunks), BATCH_SIZE), desc="Scanning")

    for i in pbar:
        batch_ids  = input_ids[i : i + BATCH_SIZE].to(DEVICE)
        batch_mask = attention_mask[i : i + BATCH_SIZE].to(DEVICE)

        outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        logits  = outputs.logits

        probs = torch.softmax(logits, dim=1)[:, 1]   # probability of RISKY
        preds = (probs >= args.threshold).long()

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# attach predictions back to chunks
for i, chunk in enumerate(chunks):
    chunk["risk_prob"] = float(all_probs[i])
    chunk["predicted"] = int(all_preds[i])
    chunk["risk_label"] = "🔴 RISKY" if all_preds[i] == 1 else "🟢 SAFE"

risky_chunks = [c for c in chunks if c["predicted"] == 1]
safe_chunks  = [c for c in chunks if c["predicted"] == 0]

print(f"\nTotal chunks scanned : {len(chunks)}")
print(f"Flagged as RISKY     : {len(risky_chunks)} ({len(risky_chunks)/len(chunks)*100:.1f}%)")
print(f"Flagged as SAFE      : {len(safe_chunks)}")


# ============================================================
# STEP 7 — SORT AND DISPLAY RESULTS
# ============================================================
print(f"\n{'='*55}")
print(f"  STEP 7 — Risk Report")
print(f"{'='*55}")

# sort risky chunks by confidence — highest risk first
risky_sorted = sorted(risky_chunks, key=lambda x: x["risk_prob"], reverse=True)

print(f"\n🔴 TOP RISKY SECTIONS (sorted by confidence)\n")
for i, chunk in enumerate(risky_sorted[:10]):   # show top 10
    print(f"  Rank {i+1} | Confidence: {chunk['risk_prob']*100:.1f}% | Page ~{chunk['approx_page']}")
    print(f"  {chunk['raw_text'][:300]}")
    print(f"  {'─'*50}")


# ============================================================
# STEP 8 — SAVE TEXT REPORT
# ============================================================
os.makedirs("results", exist_ok=True)
report_path = args.output

with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"CONTRACT RISK ANALYSIS REPORT\n")
    f.write(f"{'='*60}\n")
    f.write(f"PDF          : {args.pdf}\n")
    f.write(f"Model        : {args.model}\n")
    f.write(f"Threshold    : {args.threshold}\n")
    f.write(f"Total chunks : {len(chunks)}\n")
    f.write(f"Risky chunks : {len(risky_chunks)} ({len(risky_chunks)/len(chunks)*100:.1f}%)\n")
    f.write(f"{'='*60}\n\n")

    f.write(f"🔴 RISKY SECTIONS (sorted by confidence)\n")
    f.write(f"{'='*60}\n\n")

    for i, chunk in enumerate(risky_sorted):
        f.write(f"[{i+1}] Confidence: {chunk['risk_prob']*100:.1f}% | Page ~{chunk['approx_page']}\n")
        f.write(f"{chunk['raw_text']}\n")
        f.write(f"{'-'*60}\n\n")

    f.write(f"\n🟢 SAFE SECTIONS\n")
    f.write(f"{'='*60}\n\n")

    for chunk in safe_chunks:
        f.write(f"[SAFE] Confidence: {(1-chunk['risk_prob'])*100:.1f}% | Page ~{chunk['approx_page']}\n")
        f.write(f"{chunk['raw_text'][:200]}...\n")
        f.write(f"{'-'*60}\n\n")

print(f"\n✅ Full report saved to: {report_path}")


# ============================================================
# STEP 9 — VISUALIZATIONS
# ============================================================
print("\nGenerating visualizations...")

# --- Plot 1: Risk probability distribution ---
plt.figure(figsize=(10, 4))
plt.hist(
    [c["risk_prob"] for c in safe_chunks],
    bins=30, alpha=0.6, color="steelblue", label="Safe chunks"
)
plt.hist(
    [c["risk_prob"] for c in risky_chunks],
    bins=30, alpha=0.6, color="crimson", label="Risky chunks"
)
plt.axvline(args.threshold, color="black", linestyle="--", label=f"Threshold ({args.threshold})")
plt.title("Risk Probability Distribution Across Contract Chunks")
plt.xlabel("Risk Probability")
plt.ylabel("Number of Chunks")
plt.legend()
plt.tight_layout()
plt.savefig("results/scan_distribution.png", dpi=150)
plt.show()

# --- Plot 2: Risk heatmap across the contract ---
probs_array = np.array([c["risk_prob"] for c in chunks])
width       = 20
height      = max(1, len(probs_array) // width + 1)
padded      = np.pad(
    probs_array,
    (0, width * height - len(probs_array)),
    constant_values=np.nan
)
grid = padded.reshape(height, width)

plt.figure(figsize=(14, max(3, height // 2)))
plt.imshow(grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
plt.colorbar(label="Risk Probability")
plt.title(f"Contract Risk Heatmap — {Path(args.pdf).name}")
plt.xlabel("Chunk position (across)")
plt.ylabel("Chunk position (down)")
risky_patch = mpatches.Patch(color="red",   label="High risk")
safe_patch  = mpatches.Patch(color="green", label="Low risk")
plt.legend(handles=[risky_patch, safe_patch], loc="upper right")
plt.tight_layout()
plt.savefig("results/scan_heatmap.png", dpi=150)
plt.show()

# --- Plot 3: Risk timeline across contract ---
plt.figure(figsize=(14, 3))
colors = ["crimson" if p >= args.threshold else "steelblue" for p in all_probs]
plt.bar(range(len(all_probs)), all_probs, color=colors, width=1.0)
plt.axhline(args.threshold, color="black", linestyle="--",
            label=f"Threshold ({args.threshold})")
plt.title("Risk Score Timeline Across Contract")
plt.xlabel("Chunk number (document order)")
plt.ylabel("Risk probability")
plt.legend()
plt.tight_layout()
plt.savefig("results/scan_timeline.png", dpi=150)
plt.show()

print("✅ Visualizations saved to results/")


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*55}")
print(f"  SCAN COMPLETE")
print(f"{'='*55}")
print(f"  Contract     : {Path(args.pdf).name}")
print(f"  Pages        : {total_pages}")
print(f"  Chunks       : {len(chunks)}")
print(f"  RISKY        : {len(risky_chunks)} ({len(risky_chunks)/len(chunks)*100:.1f}%)")
print(f"  SAFE         : {len(safe_chunks)}")
print(f"  Top risk     : {max(all_probs)*100:.1f}% confidence")
print(f"  Avg risk     : {np.mean(all_probs)*100:.1f}%")
print(f"\n  Outputs:")
print(f"    results/scan_report.txt       ← full text report")
print(f"    results/scan_distribution.png ← probability distribution")
print(f"    results/scan_heatmap.png      ← risk heatmap")
print(f"    results/scan_timeline.png     ← risk timeline")
