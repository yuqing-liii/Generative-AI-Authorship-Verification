# t1_compute_style_features.py
"""
Compute stylometric features for PAN Task 1 train/val
and save them as .npy files.
"""

import os
import numpy as np
import pandas as pd

from style_features import compute_style_features


def main():
    print("Loading PAN Task1 data (train/val)...")
    train_df = pd.read_json("train.jsonl", lines=True)
    val_df = pd.read_json("val.jsonl", lines=True)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    train_texts = train_df["text"].tolist()
    val_texts = val_df["text"].tolist()

    print("Computing style features for train...")
    train_feats = compute_style_features(train_texts)
    print("Train style features shape:", train_feats.shape)

    print("Computing style features for val...")
    val_feats = compute_style_features(val_texts)
    print("Val style features shape:", val_feats.shape)

    os.makedirs("t1_style_features", exist_ok=True)
    np.save("t1_style_features/train_style.npy", train_feats)
    np.save("t1_style_features/val_style.npy", val_feats)
    print("Saved style features to t1_style_features/train_style.npy and val_style.npy")


if __name__ == "__main__":
    main()

