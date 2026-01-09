# t2_derive_binary_from_6class.py
"""
Use the 6-class Task 2 RoBERTa model outputs to derive:
1) human-dominated vs AI-dominated
2) mixed vs non-mixed

Inputs:
    t2_roberta_outputs/t2_dev_logits.npy  (N, 6)
    t2_roberta_outputs/t2_dev_labels.npy  (N,)
Outputs:
    t2_roberta_outputs/task2_dom_binary_metrics.txt
    t2_roberta_outputs/task2_mix_binary_metrics.txt
"""

import os
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import torch

OUTPUT_DIR = "t2_roberta_outputs"


def evaluate_binary(probs, labels, positive_label=1):
    """
    Generic binary evaluation.
    probs: numpy array of shape (N,) with P(y=1)
    labels: numpy array of shape (N,) in {0,1}
    """
    auc = roc_auc_score(labels, probs)
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)

    return {
        "AUC": auc,
        "F1": f1,
        "ACC": acc,
        "FPR": fpr,
        "FNR": fnr,
    }


def main():
    logits_path = os.path.join(OUTPUT_DIR, "t2_dev_logits.npy")
    labels_path = os.path.join(OUTPUT_DIR, "t2_dev_labels.npy")

    logits = np.load(logits_path)   # (N, 6)
    labels6 = np.load(labels_path)  # (N,)

    # softmax probabilities over 6 classes
    probs6 = torch.softmax(torch.tensor(logits), dim=-1).numpy()  # (N, 6)

    # --- 1) Human-dominated vs AI-dominated ---
    # mapping: 0,1,2 -> 0 (human-dominated); 3,4,5 -> 1 (AI-dominated)
    dom_labels = np.isin(labels6, [3, 4, 5]).astype(int)

    # P(AI-dominated) = sum softmax[:, 3:6]
    p_ai_dom = probs6[:, 3] + probs6[:, 4] + probs6[:, 5]

    dom_metrics = evaluate_binary(p_ai_dom, dom_labels)
    print("Human-dominated vs AI-dominated metrics:")
    print(dom_metrics)

    with open(os.path.join(OUTPUT_DIR, "task2_dom_binary_metrics.txt"), "w") as f:
        f.write(str(dom_metrics) + "\n")

    # --- 2) Mixed vs non-mixed ---
    # non-mixed = label 0; mixed = 1..5
    mix_labels = (labels6 != 0).astype(int)

    # P(mixed) = 1 - P(label=0)
    p_mixed = 1.0 - probs6[:, 0]

    mix_metrics = evaluate_binary(p_mixed, mix_labels)
    print("\nMixed vs non-mixed metrics:")
    print(mix_metrics)

    with open(os.path.join(OUTPUT_DIR, "task2_mix_binary_metrics.txt"), "w") as f:
        f.write(str(mix_metrics) + "\n")


if __name__ == "__main__":
    main()
