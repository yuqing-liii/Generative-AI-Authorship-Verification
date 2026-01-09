# t2_derive_binary_from_6class_demo.py
"""
Derive binary tasks from Task 2 6-class RoBERTa demo outputs.

Inputs (produced by t2_results_6class_demo.py):
    - t2_roberta_outputs_demo/t2_dev_logits_demo.npy  (N, 6)
    - t2_roberta_outputs_demo/t2_dev_labels_demo.npy  (N,)

We derive two binary tasks:

1) Human-dominated vs AI-dominated
   - human-dominated labels: {0, 1, 2} -> binary label 0
   - AI-dominated labels:    {3, 4, 5} -> binary label 1
   - P(AI-dominated) = sum_{k=3..5} softmax_k

2) Mixed vs non-mixed
   - non-mixed: {0}                      -> binary label 0
   - mixed:     {1, 2, 3, 4, 5}          -> binary label 1
   - P(mixed) = 1 - softmax_0

Outputs:
    - t2_roberta_outputs_demo/t2_binary_from_6class_demo.txt
        (metrics for both binary tasks)
    - t2_roberta_outputs_demo/task2_dev_human_vs_ai_demo.csv
    - t2_roberta_outputs_demo/task2_dev_mixed_vs_nonmixed_demo.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


OUTPUT_DIR = "t2_roberta_outputs_demo"
LOGITS_PATH = os.path.join(OUTPUT_DIR, "t2_dev_logits_demo.npy")
LABELS_PATH = os.path.join(OUTPUT_DIR, "t2_dev_labels_demo.npy")


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """
    Stable softmax over last dimension.
    logits: (N, C)
    return: (N, C)
    """
    x = logits - logits.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    probs = exp_x / exp_x.sum(axis=1, keepdims=True)
    return probs


def compute_binary_metrics(y_true, p_pos, threshold: float = 0.5, task_name: str = ""):
    """
    Compute common binary classification metrics.
    y_true: (N,) binary {0,1}
    p_pos:  (N,) predicted probability for class 1
    """
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos)

    y_pred = (p_pos >= threshold).astype(int)

    # AUC
    try:
        auc = roc_auc_score(y_true, p_pos)
    except ValueError:
        auc = float("nan")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # confusion matrix to get FPR/FNR
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "task": task_name,
        "AUC": auc,
        "F1": f1,
        "ACC": acc,
        "FPR": fpr,
        "FNR": fnr,
    }
    return metrics, y_pred


def main():
    if not os.path.exists(LOGITS_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"Cannot find demo logits/labels at:\n"
            f"  {LOGITS_PATH}\n"
            f"  {LABELS_PATH}\n"
            f"Please run t2_results_6class_demo.py first."
        )

    print("Loading demo logits and labels...")
    logits = np.load(LOGITS_PATH)  # shape (N, 6)
    labels_6 = np.load(LABELS_PATH)  # shape (N,)

    print("Logits shape:", logits.shape)
    print("Labels shape:", labels_6.shape)

    # 1. softmax over 6 classes
    probs = softmax_np(logits)  # (N, 6)

    # ------------------------------------------------------------------
    # Task A: human-dominated vs AI-dominated
    #
    # human-dominated: {0,1,2} -> 0
    # AI-dominated:    {3,4,5} -> 1
    # P(AI) = sum probs[:,3:6]
    # ------------------------------------------------------------------
    print("\nDeriving Task A: human-dominated vs AI-dominated ...")
    y_true_A = np.isin(labels_6, [3, 4, 5]).astype(int)          # 1 if AI-dominated
    p_pos_A = probs[:, 3:6].sum(axis=1)                          # P(AI-dominated)

    metrics_A, y_pred_A = compute_binary_metrics(
        y_true_A, p_pos_A, threshold=0.5, task_name="human_vs_ai_dominated"
    )
    print("Task A metrics:", metrics_A)

    # Save per-example CSV for Task A (demo)
    df_A = pd.DataFrame({
        "index": np.arange(len(labels_6)),
        "label_6class": labels_6,
        "label_binary_human_vs_ai": y_true_A,
        "p_ai_dominated": p_pos_A,
        "pred_binary_human_vs_ai": y_pred_A,
    })
    csv_A_path = os.path.join(OUTPUT_DIR, "task2_dev_human_vs_ai_demo.csv")
    df_A.to_csv(csv_A_path, index=False)
    print("Saved Task A per-example scores to", csv_A_path)

    # ------------------------------------------------------------------
    # Task B: mixed vs non-mixed
    #
    # non-mixed: {0} -> 0
    # mixed:    {1,2,3,4,5} -> 1
    # P(mixed) = 1 - P(label=0) = 1 - probs[:,0]
    # ------------------------------------------------------------------
    print("\nDeriving Task B: mixed vs non-mixed ...")
    y_true_B = (labels_6 != 0).astype(int)                       # 1 if mixed
    p_pos_B = 1.0 - probs[:, 0]                                  # P(mixed)

    metrics_B, y_pred_B = compute_binary_metrics(
        y_true_B, p_pos_B, threshold=0.5, task_name="mixed_vs_nonmixed"
    )
    print("Task B metrics:", metrics_B)

    # Save per-example CSV for Task B (demo)
    df_B = pd.DataFrame({
        "index": np.arange(len(labels_6)),
        "label_6class": labels_6,
        "label_binary_mixed": y_true_B,
        "p_mixed": p_pos_B,
        "pred_binary_mixed": y_pred_B,
    })
    csv_B_path = os.path.join(OUTPUT_DIR, "task2_dev_mixed_vs_nonmixed_demo.csv")
    df_B.to_csv(csv_B_path, index=False)
    print("Saved Task B per-example scores to", csv_B_path)

    # ------------------------------------------------------------------
    # Save both metric dicts to one txt file
    # ------------------------------------------------------------------
    metrics_txt_path = os.path.join(OUTPUT_DIR, "t2_binary_from_6class_demo.txt")
    with open(metrics_txt_path, "w") as f:
        f.write("Task A: human-dominated vs AI-dominated\n")
        for k, v in metrics_A.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTask B: mixed vs non-mixed\n")
        for k, v in metrics_B.items():
            f.write(f"{k}: {v}\n")
    print("Saved binary metrics (demo) to", metrics_txt_path)


if __name__ == "__main__":
    main()
