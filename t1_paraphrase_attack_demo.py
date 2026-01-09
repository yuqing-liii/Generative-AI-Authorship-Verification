# t1_paraphrase_attack_demo.py
"""
Light paraphrase / noisy edit attack demo.

We:
  - sample a subset of Task 2 dev documents,
  - run Task1 RoBERTa detector on the original texts,
  - apply a simple automatic 'paraphrase' (synonym replacement + word drop + small swaps),
  - run the detector again,
  - compare AUC / F1 / ACC before vs. after.

This gives a Day6-style robustness result.
"""

import os
import json
import random
import re
from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from t1_predict_roberta import predict_t1  # uses your fixed Task1 RoBERTa

TASK2_DEV_PATH = "../task2/task2_dev.jsonl"
OUTPUT_DIR = "robustness_outputs"
N_SAMPLES = 1000

# -------------------- small helper functions --------------------

SYNONYMS = {
    "good": "great",
    "bad": "poor",
    "people": "individuals",
    "students": "learners",
    "teachers": "instructors",
    "important": "crucial",
    "very": "extremely",
    "really": "truly",
    "think": "believe",
    "help": "assist",
    "show": "demonstrate",
    "use": "utilize",
    "problem": "issue",
    "big": "major",
    "small": "minor",
}


def light_paraphrase(text: str, drop_p: float = 0.05) -> str:
    """
    Very cheap 'paraphrase':
      - replace some words with hand-written synonyms,
      - randomly drop a few words,
      - swap two words once.
    """
    words = text.split()
    new_words: List[str] = []

    for w in words:
        # split off trailing punctuation so we don't lose dots/commas
        m = re.match(r"^(\w+)(\W*)$", w)
        if m:
            core, punct = m.group(1), m.group(2)
        else:
            core, punct = w, ""

        low = core.lower()

        # maybe drop
        if random.random() < drop_p:
            continue

        # maybe synonym replace
        if low in SYNONYMS and random.random() < 0.5:
            repl = SYNONYMS[low]
            if core[0].isupper():
                repl = repl.capitalize()
            new_words.append(repl + punct)
        else:
            new_words.append(core + punct)

    # small word swap to disturb local order
    if len(new_words) > 10:
        i, j = sorted(random.sample(range(len(new_words)), 2))
        new_words[i], new_words[j] = new_words[j], new_words[i]

    return " ".join(new_words)


def evaluate_binary(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)

    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)

    fp = np.sum((labels == 0) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fn = np.sum((labels == 1) & (preds == 0))
    tp = np.sum((labels == 1) & (preds == 1))
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
    random.seed(42)
    np.random.seed(42)

    # 1) Load Task2 dev (just once)
    rows = []
    with open(TASK2_DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    texts = [r["text"] for r in rows]
    labels_6 = [r["label"] for r in rows]

    # binary label: AI (1) if label in {1,2,3,4,5}, else 0
    ai_set = {1, 2, 3, 4, 5}
    labels_bin = np.array([1 if y in ai_set else 0 for y in labels_6], dtype=int)

    # 2) sample a subset
    n = len(texts)
    idx_all = list(range(n))
    random.shuffle(idx_all)
    idx_sample = idx_all[:N_SAMPLES]

    sample_texts = [texts[i] for i in idx_sample]
    sample_labels = labels_bin[idx_sample]

    print(f"Sampled {len(sample_texts)} documents from Task2 dev.")

    # 3) Original predictions
    print("Running Task1 RoBERTa on ORIGINAL texts...")
    probs_orig = np.array(predict_t1(sample_texts))

    metrics_orig = evaluate_binary(sample_labels, probs_orig)
    print("\nMetrics on ORIGINAL texts:")
    for k, v in metrics_orig.items():
        print(f"  {k}: {v:.4f}")

    # 4) Paraphrased texts + predictions
    print("\nGenerating light paraphrases...")
    paraphrased_texts = [light_paraphrase(t) for t in sample_texts]

    print("Running Task1 RoBERTa on PARAPHRASED texts...")
    probs_para = np.array(predict_t1(paraphrased_texts))

    metrics_para = evaluate_binary(sample_labels, probs_para)
    print("\nMetrics on PARAPHRASED texts:")
    for k, v in metrics_para.items():
        print(f"  {k}: {v:.4f}")

    # 5) Save metrics + a few examples
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # metrics
    with open(os.path.join(OUTPUT_DIR, "t1_paraphrase_attack_metrics.txt"), "w") as f:
        f.write("ORIGINAL:\n")
        for k, v in metrics_orig.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\nPARAPHRASED:\n")
        for k, v in metrics_para.items():
            f.write(f"{k}: {v:.6f}\n")

    # a small CSV of examples
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "t1_paraphrase_attack_examples.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label_bin", "p_ai_orig", "p_ai_paraphrased", "text_orig", "text_paraphrased"])
        for idx, lbl, p_o, p_p, t_o, t_p in zip(
            idx_sample, sample_labels, probs_orig, probs_para, sample_texts, paraphrased_texts
        ):
            writer.writerow([idx, int(lbl), float(p_o), float(p_p), t_o, t_p])

    print("\nSaved metrics and example pairs to folder:", OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
