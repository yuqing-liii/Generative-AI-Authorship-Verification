# t1_predict_roberta.py

import os
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "roberta-base"
CHECKPOINT_DIR = "t1_bert_outputs/checkpoint-17781"
MAX_LENGTH = 512

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

print("Using device:", device)

print("Loading tokenizer & model for Task1...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
model.to(device)
model.eval()

@torch.no_grad()
def predict_t1(text_list: List[str]) -> List[float]:
    """
    Input: list of texts
    Output: list of P(AI) probabilities (float)
    """
    all_probs = []

    # Process in batches to avoid using too much memory at once
    batch_size = 8
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        logits = outputs.logits  # (batch, 2)
        probs = torch.softmax(logits, dim=-1)[:, 1]  # index 1 is the AI class

        all_probs.extend(probs.cpu().numpy().tolist())

    return all_probs

if __name__ == "__main__":
    # Simple self-test: use the first few samples from val.jsonl to check if outputs look reasonable
    import pandas as pd
    from metrics import evaluate_binary

    df_val = pd.read_json("val.jsonl", lines=True)
    texts = df_val["text"].tolist()
    labels = df_val["label"].tolist()

    print("Running quick sanity check on first 200 val samples...")
    probs = predict_t1(texts[:200])
    metrics = evaluate_binary(labels[:200], probs)
    print("Sanity check metrics (first 200):", metrics)
