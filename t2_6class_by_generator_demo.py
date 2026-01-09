# t2_6class_by_generator_demo.py
"""
Analyse robustness of the Task 2 6-class RoBERTa DEMO model
across different generator models (chatgpt, llm1-llm2, etc.).

It uses:
  - logits/labels from: t2_roberta_outputs_demo/t2_dev_logits_demo.npy
                        t2_roberta_outputs_demo/t2_dev_labels_demo.npy
  - metadata from:      ../task2/task2_dev.jsonl
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

OUTPUT_DIR = "t2_roberta_outputs_demo"
TASK2_DEV_PATH = "../task2/task2_dev.jsonl"


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    # 1) Load demo logits & labels
    logits_path = os.path.join(OUTPUT_DIR, "t2_dev_logits_demo.npy")
    labels_path = os.path.join(OUTPUT_DIR, "t2_dev_labels_demo.npy")

    logits = np.load(logits_path)   # (N, 6)
    labels = np.load(labels_path)   # (N,)
    probs = softmax(logits)
    preds = probs.argmax(axis=1)

    print("Loaded logits:", logits.shape, "labels:", labels.shape)

    # 2) Load Task2 dev metadata (to get `model` field)
    dev_rows = []
    with open(TASK2_DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            dev_rows.append(json.loads(line))

    assert len(dev_rows) == len(labels), \
        f"Dev jsonl has {len(dev_rows)} rows, but logits have {len(labels)}."

    models = [row.get("model", "unknown") for row in dev_rows]

    df = pd.DataFrame({
        "model": models,
        "label": labels,
        "pred": preds,
    })

    # 3) Overall performance (sanity check)
    overall_acc = accuracy_score(labels, preds)
    overall_macro_f1 = f1_score(labels, preds, average="macro")
    overall_micro_f1 = f1_score(labels, preds, average="micro")
    print("\n=== Overall (all generators) ===")
    print(f"Accuracy:  {overall_acc:.4f}")
    print(f"Macro F1:  {overall_macro_f1:.4f}")
    print(f"Micro F1:  {overall_micro_f1:.4f}")

    # 4) Per-generator performance
    rows = []
    print("\n=== Per-generator performance (6-class DEMO) ===")
    for m, g in df.groupby("model"):
        y_true = g["label"].to_numpy()
        y_pred = g["pred"].to_numpy()
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        micro_f1 = f1_score(y_true, y_pred, average="micro")
        print(f"[{m:15s}] n={len(g):6d}  acc={acc:.4f}  macroF1={macro_f1:.4f}  microF1={micro_f1:.4f}")
        rows.append({
            "model": m,
            "n": len(g),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
        })

    out_df = pd.DataFrame(rows).sort_values("n", ascending=False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, "t2_6class_by_generator_demo.csv")
    out_df.to_csv(out_csv, index=False)
    print("\nSaved per-generator results to:", out_csv)


if __name__ == "__main__":
    main()
