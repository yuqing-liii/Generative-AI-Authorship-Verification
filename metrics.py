# metrics.py
from typing import Dict, Sequence

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


def evaluate_binary(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    General binary classification evaluation:
    Given 0/1 labels and predicted scores P(1), return AUC/F1/ACC/FPR/FNR.
    """
    # Scores -> predicted 0/1 labels
    y_pred = [1 if s >= threshold else 0 for s in y_score]

    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)

    return {
        "AUC": auc,
        "F1": f1,
        "ACC": acc,
        "FPR": fpr,
        "FNR": fnr,
    }
