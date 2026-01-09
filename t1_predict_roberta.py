# t1_predict_roberta.py
import os
from typing import List

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

MODEL_DIR = "t1_bert_outputs/checkpoint-17781"  # Your selected best checkpoint

print(f"Using device: {DEVICE}")

# ---- 1. Globally load tokenizer & model ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()


def predict_t1(texts: List[str], batch_size: int = 16) -> List[float]:
    """
    Given a batch of texts, return for each text the probability of being AI, P(AI).
    All robustness experiments for Day6/Day7 will use this function.
    """
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            logits = model(**enc).logits  # (batch, 2)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(class=1, i.e., AI)
            all_probs.extend(probs.cpu().tolist())

    return all_probs


def sanity_check_on_val(n_samples: int = 200):
    """
    Quickly run a sanity check on the first n_samples rows in val.jsonl.
    """
    from metrics import evaluate_binary  # We'll write this file in the next step

    df_val = pd.read_json("val.jsonl", lines=True)
    df_sub = df_val.head(n_samples)

    texts = df_sub["text"].tolist()
    labels = df_sub["label"].tolist()

    p_ai = predict_t1(texts, batch_size=8)
    metrics = evaluate_binary(labels, p_ai)

    print(f"Sanity check metrics (first {n_samples}): {metrics}")


if __name__ == "__main__":
    sanity_check_on_val(200)
