# t2_sliding_window_demo.py
"""
Paragraph-level mixed-authorship demo on the Task 2 dev set.

Steps:
- Load the Task 1 RoBERTa checkpoint (binary AI detector).
- Select a small subset of Task 2 dev documents with different collaboration labels.
- Split each document into sentences.
- Apply a sliding window over the sentences.
- For each window, call the Task 1 model to obtain P(AI).
- Save:
    * A CSV with window indices + P(AI)
    * A PNG plot: window_index vs P(AI)
"""

import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---- Config ----
# Use the Task 1 RoBERTa checkpoint that you have validated
CHECKPOINT_DIR = "t1_bert_outputs/checkpoint-17781"

# Path to the Task 2 dev dataset (your structure: task1 / task2 are sibling folders)
TASK2_DEV_PATH = "../task2/task2_dev.jsonl"

# Output directory
OUT_DIR = "t2_sliding_outputs_demo"

# Sliding window hyperparameters
WINDOW_SENT = 5          # number of sentences per window
STRIDE_SENT = 2          # number of sentences to slide each step
MAX_LEN = 512            # RoBERTa maximum input length
BATCH_SIZE = 8           # inference batch size (can be larger since this is inference)

# How many documents to sample per label
SAMPLES_PER_LABEL = 2    # You may increase to 3–4; small for demo

# Which labels to analyze (mixed-authorship types)
TARGET_LABELS = [0, 1, 2, 3, 4, 5]  # You may choose only [3,4,5] plus 0 as control


# ---- Step 1: Load the Task 1 model ----
def load_t1_model_and_tokenizer():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_t1(
    texts: List[str],
    tokenizer,
    model,
    device,
    max_length: int = MAX_LEN,
    batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """
    Given a list of texts, return P(AI) for each text.
    Output shape: numpy array of shape [N].
    """
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            logits = outputs.logits  # (B, 2)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(AI)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ---- Step 2: Simple sentence splitter ----
SENT_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+|_SEP_')


def split_into_sentences(text: str) -> List[str]:
    """
    A very rough sentence splitter (splits on . ! ? + space, or _SEP_).
    Sufficient for demo visualization; not intended for precise boundary detection.
    """
    text = text.replace("\n", " ").strip()
    if not text:
        return []

    parts = SENT_SPLIT_REGEX.split(text)
    sents = [s.strip() for s in parts if s.strip()]
    return sents


# ---- Step 3: Construct sliding windows ----
def build_windows_from_sentences(
    sentences: List[str],
    window_size: int = WINDOW_SENT,
    stride: int = STRIDE_SENT
) -> List[Tuple[int, int, str]]:
    """
    Given a list of sentences, return a list of windows:
        (start_idx, end_idx, window_text)
    end_idx is inclusive.
    """
    windows = []
    n = len(sentences)
    if n == 0:
        return windows

    start = 0
    while start < n:
        end = min(start + window_size, n)
        window_sents = sentences[start:end]
        window_text = " ".join(window_sents)
        windows.append((start, end - 1, window_text))
        if end == n:
            break
        start += stride
    return windows


# ---- Step 4: Main workflow ----
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 4.1 Load Task 1 model
    tokenizer, model, device = load_t1_model_and_tokenizer()

    # 4.2 Load Task 2 dev set
    print("Loading Task 2 dev data from:", TASK2_DEV_PATH)
    dev_df = pd.read_json(TASK2_DEV_PATH, lines=True)
    print("Dev size:", len(dev_df))

    # Check if label_text is present (optional)
    has_label_text = "label_text" in dev_df.columns

    # 4.3 Sample documents from each label
    demo_rows = []
    for lab in TARGET_LABELS:
        subset = dev_df[dev_df["label"] == lab]
        if subset.empty:
            continue
        demo_rows.append(subset.head(SAMPLES_PER_LABEL))

    if not demo_rows:
        print("No demo samples found for the specified TARGET_LABELS.")
        return

    demo_df = pd.concat(demo_rows, ignore_index=True)
    print(f"Selected {len(demo_df)} demo documents.")
    print("Label distribution in demo:")
    print(demo_df["label"].value_counts())

    # 4.4 For each document, run sliding-window + visualization
    for idx, row in demo_df.iterrows():
        doc_id = row.get("id", f"dev_{idx}")
        text = row["text"]
        label = int(row["label"])
        label_text = row["label_text"] if has_label_text else ""

        print(f"\n=== Processing doc {doc_id} (label={label}) ===")
        if label_text:
            print("Label text:", label_text)

        # Sentence splitting
        sentences = split_into_sentences(text)
        print(f"Number of sentences: {len(sentences)}")

        # Build windows
        windows = build_windows_from_sentences(sentences)
        if not windows:
            print("No windows constructed; skipping this document.")
            continue

        start_indices = [w[0] for w in windows]
        end_indices = [w[1] for w in windows]
        window_texts = [w[2] for w in windows]

        print(f"Number of windows: {len(windows)}")

        # Run Task 1 detector to obtain P(AI) per window
        p_ai = predict_t1(window_texts, tokenizer, model, device)

        # ---- 4.4.1 Save CSV ----
        out_csv = os.path.join(
            OUT_DIR,
            f"sliding_doc_{doc_id}_label{label}.csv"
        )
        df_out = pd.DataFrame({
            "doc_id": doc_id,
            "label": label,
            "label_text": label_text,
            "window_index": range(len(windows)),
            "start_sent": start_indices,
            "end_sent": end_indices,
            "p_ai": p_ai,
        })
        df_out.to_csv(out_csv, index=False)
        print("Saved window scores to:", out_csv)

        # ---- 4.4.2 Plot P(AI) curve ----
        plt.figure(figsize=(8, 4))
        x = range(len(windows))
        plt.plot(x, p_ai, marker="o")
        plt.ylim(0.0, 1.0)
        plt.xlabel("Window index")
        plt.ylabel("P(AI)")
        plt.title(f"Doc {doc_id} (label {label})")

        out_png = os.path.join(
            OUT_DIR,
            f"sliding_curve_doc_{doc_id}_label{label}.png"
        )
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print("Saved curve figure to:", out_png)

    print("\nAll demo sliding-window analyses completed.")
    print("You can now open the PNG files in", OUT_DIR, "for your slides or paper.")


if __name__ == "__main__":
    main()
