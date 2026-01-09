"""
Task 1 – Document-level Transformer baseline (BERT/RoBERTa)

- Model: roberta-base (2-class)
- Input: text (truncated to 512 tokens)
- Train on: train.jsonl
- Validate on: val.jsonl
- Output:
    * Validation metrics: AUC / F1 / Accuracy
    * Per-document scores: t1_bert_val_pred.csv
"""

import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# -----------------------
# 1. Basic config
# -----------------------
MODEL_NAME = "roberta-base"
MAX_LENGTH = 512
OUTPUT_DIR = "t1_bert_outputs"

def main():
    # -----------------------
    # 2. Load data with pandas
    # -----------------------
    print("Loading PAN Task1 data (train/val)...")

    df_train = pd.read_json("train.jsonl", lines=True)
    df_val = pd.read_json("val.jsonl", lines=True)

    df_train = df_train[["id", "text", "label"]]
    df_val = df_val[["id", "text", "label"]]

    print("Train size:", len(df_train))
    print("Val size:", len(df_val))
    print("Train label distribution:\n", df_train["label"].value_counts())
    print("Val label distribution:\n", df_val["label"].value_counts())

    # df_train = df_train.sample(n=5000, random_state=42).reset_index(drop=True)

    # -----------------------
    # 3. Convert to HF Dataset
    # -----------------------
    train_ds = Dataset.from_pandas(df_train)
    val_ds = Dataset.from_pandas(df_val)

    # HuggingFace
    for name, ds in [("train", train_ds), ("val", val_ds)]:
        extra_cols = [c for c in ds.column_names
                      if c not in ["id", "text", "label"]]
        if extra_cols:
            ds = ds.remove_columns(extra_cols)
        if name == "train":
            train_ds = ds
        else:
            val_ds = ds

    # -----------------------
    # 4. Tokenizer & preprocess
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("Tokenizing...")
    train_ds = train_ds.map(preprocess_function, batched=True)
    val_ds = val_ds.map(preprocess_function, batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    val_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # -----------------------
    # 5. Load model
    # -----------------------
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # -----------------------
    # 6. TrainingArguments
    # -----------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3, 
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    # -----------------------
    # 7. Metrics function
    # -----------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # logits shape: (batch_size, 2)
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)

        return {
            "auc": float(auc),
            "f1": float(f1),
            "accuracy": float(acc),
        }

    # -----------------------
    # 8. Trainer
    # -----------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -----------------------
    # 9. Train & evaluate
    # -----------------------
    print("Start training...")
    trainer.train()

    print("Evaluating on val set...")
    eval_metrics = trainer.evaluate()
    print("Validation metrics:", eval_metrics)

    # -----------------------
    # 10. Save per-document scores
    # -----------------------
    print("Predicting on val set and saving scores...")
    pred_output = trainer.predict(val_ds)
    logits = pred_output.predictions
    labels = pred_output.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    df_scores = pd.DataFrame({
        "id": df_val["id"].values,
        "label": labels,
        "prob_ai": probs,
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "t1_bert_val_pred.csv")
    df_scores.to_csv(out_path, index=False)
    print(f"Saved val scores to {out_path}")

    print("Done.")

if __name__ == "__main__":
    main()
