# t1_on_task2_binary.py
#
# Use the PAN Task 1 RoBERTa detector on PAN Task 2 (binary setting).
# Goal: measure how well the Task 1 document-level detector
# generalizes to Task 2 texts annotated with 6-way labels.

import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

# -------- Paths and config --------
# Path to Task 2 dev split (6-way labels)
TASK2_DEV_PATH = "../task2/task2_dev.jsonl"

# Best checkpoint from Task 1 RoBERTa training
CHECKPOINT_DIR = "t1_bert_outputs/checkpoint-17781"

MAX_LENGTH = 512
BATCH_SIZE = 8  # reduce to 4 or 2 if you run out of memory

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# -------- Label mapping: 6-way -> binary --------
def map_label_to_binary(label: int) -> int:
    """
    Map PAN Task 2 six-way labels to a binary label:
    - 0: fully human-written              -> 0 (human-only)
    - 1: human-written, then machine-polished
    - 2: machine-written, then machine-humanized
    - 3: human-initiated, then machine-continued
    - 4: deeply-mixed human/machine text
    - 5: machine-written, then human-edited
      -> 1 (contains AI involvement)

    Return:
        0 for human-only, 1 for any text with AI involvement.
    """
    if label == 0:
        return 0
    else:
        return 1


def load_task2_dev_binary(path: str):
    """
    Load the Task 2 dev file and convert labels to binary.

    Returns:
        texts:  list of raw strings
        labels: np.array of shape (N,) with 0/1 labels
        ids:    list of IDs or None (if the file has no 'id' field)
    """
    texts = []
    labels = []
    ids = []

    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            # Task 2 JSONL does not necessarily have an 'id' field;
            # we keep it if present, otherwise store None.
            ids.append(obj.get("id", None))

            label_6 = obj["label"]
            label_bin = map_label_to_binary(label_6)
            labels.append(label_bin)

    return texts, np.array(labels, dtype=np.int64), ids


def main():
    print("Using device:", DEVICE)

    # 1. Load tokenizer and model from the Task 1 checkpoint
    print("Loading tokenizer & model from", CHECKPOINT_DIR, "...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    model.to(DEVICE)
    model.eval()

    # 2. Load Task 2 dev data and map to binary labels
    print("Loading Task 2 dev data (binary mapping)...")
    texts, labels, ids = load_task2_dev_binary(TASK2_DEV_PATH)
    print(f"Number of dev samples: {len(texts)}")

    # 3. Run inference in batches and collect P(AI)
    all_probs = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(texts), BATCH_SIZE),
            desc="Running Task1 RoBERTa on Task2 dev",
        ):
            batch_texts = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            outputs = model(**enc)
            logits = outputs.logits  # shape: (batch, 2)

            # Probability that the class is 1 (AI side)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    preds = (all_probs >= 0.5).astype(int)

    # 4. Compute evaluation metrics
    auc = roc_auc_score(labels, all_probs)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)

    metrics = {
        "AUC": float(auc),
        "F1": float(f1),
        "ACC": float(acc),
        "FPR": float(fpr),
        "FNR": float(fnr),
    }

    print("Task1 RoBERTa on Task2 (binary) metrics:")
    print(metrics)

    # 5. Save per-example scores for later analysis/plots
    os.makedirs("t1_on_task2_outputs", exist_ok=True)
    out_csv = os.path.join(
        "t1_on_task2_outputs", "task2_dev_roberta_binary.csv"
    )

    import csv

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "id", "binary_label", "p_ai"])
        for idx, (doc_id, y, p) in enumerate(zip(ids, labels, all_probs)):
            writer.writerow([idx, doc_id, int(y), float(p)])

    print("Saved per-example scores to", out_csv)

    # 6. (Optional) also save metrics to a txt file
    metrics_path = os.path.join(
        "t1_on_task2_outputs", "task2_binary_metrics.txt"
    )
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print("Saved metrics to", metrics_path)


if __name__ == "__main__":
    main()
