# t1_hybrid_contrastive.py

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from style_features import compute_style_features  # Not used directly for now, kept for future extensions


# ========= Config (feel free to tweak) =========
MODEL_NAME = "roberta-base"
MAX_LENGTH = 512          # Texts are long; truncate to 512 tokens to reduce memory

BATCH_SIZE = 2            # MPS memory is small; start with 2 and increase if it fits
NUM_EPOCHS = 2            # For now train 2 epochs, enough to compare models
LR = 2e-5

USE_CONTRASTIVE = True       # Whether to enable contrastive learning
LAMBDA_CONTRASTIVE = 0.2     # L = CE + λ * L_con
TEMPERATURE = 0.1

OUTPUT_DIR = "t1_hybrid_outputs"
STYLE_FEAT_DIR = "t1_style_features"
# ==============================================


class HybridContrastiveModel(nn.Module):
    """
    Inputs:
      - CLS vector from RoBERTa
      - Handcrafted style feature vector

    Outputs:
      - Binary classification logits
      - During training: CE loss + λ * supervised contrastive loss
    """

    def __init__(
        self,
        model_name: str,
        style_dim: int,
        num_labels: int = 2,
        use_contrastive: bool = True,
        lambda_contrastive: float = 0.2,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.style_dim = style_dim
        self.use_contrastive = use_contrastive
        self.lambda_contrastive = lambda_contrastive
        self.temperature = temperature

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + style_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        style_feats=None,
        labels=None,
    ):
        # 1. Transformer encoding
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS vector
        cls = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)

        # 2. Concatenate style features
        if style_feats is None:
            raise ValueError("style_feats is required for Hybrid model")
        style_feats = style_feats.to(cls.dtype)
        h = torch.cat([cls, style_feats], dim=-1)  # (batch, hidden + style_dim)

        # 3. Classification head
        logits = self.mlp(h)

        loss = None
        if labels is not None:
            loss_ce = self.ce_loss(logits, labels)
            loss = loss_ce

            if self.use_contrastive:
                loss_con = self.supervised_contrastive_loss(h, labels)
                loss = loss_ce + self.lambda_contrastive * loss_con

        return {"loss": loss, "logits": logits}

    def supervised_contrastive_loss(self, features, labels):
        """
        Supervised contrastive loss (InfoNCE style) within a mini-batch.

        features: (batch, dim)
        labels:   (batch,)
        """
        device = features.device
        labels = labels.view(-1, 1)  # (batch, 1)

        # L2 normalization
        features = F.normalize(features, p=2, dim=1)

        # Similarity matrix
        logits = torch.matmul(features, features.T) / self.temperature  # (b, b)

        batch_size = features.size(0)
        logits_mask = torch.ones_like(logits) - torch.eye(batch_size, device=device)
        logits = logits * logits_mask

        # Positive mask: same label
        label_mask = torch.eq(labels, labels.T).float().to(device)
        positive_mask = label_mask * logits_mask  # remove self-comparisons

        # Numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - max_logits.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_log_prob = (positive_mask * log_prob).sum(1) / positive_mask.sum(1).clamp(min=1.0)

        loss = -pos_log_prob.mean()
        return loss


def load_style_features():
    train_path = os.path.join(STYLE_FEAT_DIR, "train_style.npy")
    val_path = os.path.join(STYLE_FEAT_DIR, "val_style.npy")
    train_feats = np.load(train_path)
    val_feats = np.load(val_path)
    return train_feats, val_feats


def build_datasets(tokenizer):
    # 1. Read Task1 jsonl files
    train_df = pd.read_json("train.jsonl", lines=True)
    val_df = pd.read_json("val.jsonl", lines=True)

    train_texts = train_df["text"].tolist()
    val_texts = val_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_labels = val_df["label"].tolist()

    # 2. Read style features
    train_style, val_style = load_style_features()
    print("Train style feats:", train_style.shape)
    print("Val  style feats:", val_style.shape)

    style_dim = train_style.shape[1]

    # 3. Build HuggingFace Datasets
    train_dict = {
        "text": train_texts,
        "label": train_labels,
        "style_feats": train_style.tolist(),
    }
    val_dict = {
        "text": val_texts,
        "label": val_labels,
        "style_feats": val_style.tolist(),
    }

    train_ds = Dataset.from_dict(train_dict)
    val_ds = Dataset.from_dict(val_dict)

    # 4. Tokenization
    def tokenize_fn(batch):
        enc = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        return enc

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)



    # 5. Set PyTorch format
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    
    columns = ["input_ids", "attention_mask", "labels", "style_feats"]
    
    train_ds.set_format(type="torch", columns=columns)
    val_ds.set_format(type="torch", columns=columns)

    return train_ds, val_ds, style_dim


def collate_fn_with_style(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collator: also batch the style_feats into a tensor.
    """
    style_feats = torch.stack(
        [example["style_feats"] for example in batch]
    )

    batch_wo_style = []
    for example in batch:
        ex = dict(example)
        ex.pop("style_feats")
        batch_wo_style.append(ex)

    batch_enc = default_data_collator(batch_wo_style)
    batch_enc["style_feats"] = style_feats
    return batch_enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "f1": f1, "auc": auc}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Building datasets with style features...")
    train_ds, val_ds, style_dim = build_datasets(tokenizer)
    print("Train size:", len(train_ds), "Val size:", len(val_ds))
    print("Style feature dim:", style_dim)

    print("Loading hybrid model...")
    model = HybridContrastiveModel(
        model_name=MODEL_NAME,
        style_dim=style_dim,
        num_labels=2,
        use_contrastive=USE_CONTRASTIVE,
        lambda_contrastive=LAMBDA_CONTRASTIVE,
        temperature=TEMPERATURE,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        eval_strategy="epoch",  # For recent transformers versions the argument name is evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        logging_steps=100,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collate_fn_with_style,
        compute_metrics=compute_metrics,
    )

    print("Start training hybrid model...")
    trainer.train()

    print("Evaluating on val set...")
    metrics = trainer.evaluate()
    print("Val metrics:", metrics)

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "val_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print("Saved val metrics to", metrics_path)

    # Save validation probabilities
    print("Predicting on val set...")
    preds = trainer.predict(val_ds)
    logits = preds.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    val_df = pd.read_json("val.jsonl", lines=True)
    val_ids = val_df["id"].tolist()
    val_labels = val_df["label"].tolist()

    import csv
    out_csv = os.path.join(OUTPUT_DIR, "t1_hybrid_val_pred.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "p_ai"])
        for doc_id, lab, p in zip(val_ids, val_labels, probs):
            writer.writerow([doc_id, lab, float(p)])

    print("Saved val scores to", out_csv)
    print("Done.")


if __name__ == "__main__":
    main()
