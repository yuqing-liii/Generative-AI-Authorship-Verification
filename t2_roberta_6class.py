# t2_roberta_6class.py
"""
Train a 6-way RoBERTa classifier on PAN Task 2.

- Input: ../task2/task2_train.jsonl, task2_dev.jsonl
- Model: roberta-base with 6 output labels
- Tricks:
    * Stratified downsampling on train set (max_per_label = 20000)
    * Class weights for imbalanced labels
- Outputs:
    * t2_roberta_outputs/checkpoint-... (HF checkpoint)
    * t2_roberta_outputs/t2_dev_logits.npy  (N, 6)
    * t2_roberta_outputs/t2_dev_labels.npy  (N,)
    * t2_roberta_outputs/t2_results_6class.txt (macro/micro F1, accuracy)
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ---- Config ----
DATA_DIR = "../task2"         # you are running from task1 folder
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "t2_roberta_outputs"

MAX_LENGTH = 512
BATCH_SIZE = 4                # small because of MPS memory
NUM_EPOCHS = 3
LR = 2e-5
MAX_PER_LABEL = 20000         # stratified downsampling per label
RANDOM_SEED = 42
NUM_LABELS = 6
# ---------------


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_task2_data():
    train_path = os.path.join(DATA_DIR, "task2_train.jsonl")
    dev_path = os.path.join(DATA_DIR, "task2_dev.jsonl")

    train_df = pd.read_json(train_path, lines=True)
    dev_df = pd.read_json(dev_path, lines=True)

    print("Train size (raw):", len(train_df))
    print("Dev size:", len(dev_df))

    # label column is already 0..5
    print("Train label distribution:", Counter(train_df["label"].tolist()))
    print("Dev   label distribution:", Counter(dev_df["label"].tolist()))

    return train_df, dev_df


def stratified_downsample(train_df):
    """Limit each label to at most MAX_PER_LABEL examples."""
    if MAX_PER_LABEL is None:
        return train_df

    dfs = []
    for lab, group in train_df.groupby("label"):
        n = len(group)
        if n > MAX_PER_LABEL:
            group = group.sample(MAX_PER_LABEL, random_state=RANDOM_SEED)
        dfs.append(group)
    new_train = pd.concat(dfs).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    print("After downsampling, train size:", len(new_train))
    print("New train label distribution:", Counter(new_train["label"].tolist()))
    return new_train


def compute_class_weights(labels):
    """
    Compute inverse-frequency class weights.
    """
    counter = Counter(labels)
    total = sum(counter.values())
    # weight[c] = total / (num_classes * count_c)
    weights = []
    for c in range(NUM_LABELS):
        count_c = counter[c]
        w = total / (NUM_LABELS * count_c)
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    print("Class weights:", weights)
    return weights


class Task2Roberta6Class(nn.Module):
    """
    Thin wrapper around AutoModelForSequenceClassification
    to plug in our own weighted CrossEntropyLoss.
    """

    def __init__(self, model_name: str, num_labels: int, class_weights=None):
        super().__init__()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,        # we will compute loss ourselves
        )
        logits = outputs.logits
        loss = None
        if labels is not None:
            if self.class_weights is not None:
                ce = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                ce = nn.CrossEntropyLoss()
            loss = ce(logits, labels)
        return {"loss": loss, "logits": logits}


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)

    # --- Load data ---
    train_df, dev_df = load_task2_data()
    train_df = stratified_downsample(train_df)

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].astype(int).tolist()
    dev_texts = dev_df["text"].tolist()
    dev_labels = dev_df["label"].astype(int).tolist()

    # --- Class weights ---
    class_weights = compute_class_weights(train_labels)

    # --- Tokenizer & datasets ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    from datasets import Dataset

    train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    dev_ds = Dataset.from_dict({"text": dev_texts, "labels": dev_labels})

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dev_ds = dev_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    dev_ds.set_format(type="torch", columns=cols)

    # --- Model ---
    model = Task2Roberta6Class(
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        class_weights=class_weights,
    )

    # --- Training args ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=2,
        logging_steps=200,
        report_to="none",
    )




    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        micro_f1 = f1_score(labels, preds, average="micro")
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Start training 6-class RoBERTa on Task 2...")
    trainer.train()

    print("Evaluating on dev (6-class)...")
    metrics = trainer.evaluate()
    print("Dev metrics:", metrics)

    # Extra: full classification report
    preds_output = trainer.predict(dev_ds)
    logits = preds_output.predictions
    preds = logits.argmax(axis=-1)

    print("\nClassification report on dev:")
    print(classification_report(dev_labels, preds, digits=4))

    # Save metrics and logits/labels
    with open(os.path.join(OUTPUT_DIR, "t2_results_6class.txt"), "w") as f:
        f.write(str(metrics) + "\n\n")
        f.write(classification_report(dev_labels, preds, digits=4))

    np.save(os.path.join(OUTPUT_DIR, "t2_dev_logits.npy"), logits)
    np.save(os.path.join(OUTPUT_DIR, "t2_dev_labels.npy"), np.array(dev_labels))

    print("Saved dev logits/labels to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
