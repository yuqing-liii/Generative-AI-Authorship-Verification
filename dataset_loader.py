import pandas as pd
from typing import List, Tuple

TEXT_COL = "text"
LABEL_COL = "label"

def load_split(path: str) -> Tuple[List[str], List[int]]:
    """
    Read texts and labels from a given jsonl file.
    Returns:
        texts: List[str]
        labels: List[int]  # 0/1
    """
    df = pd.read_json(path, lines=True)

    # Print basic info for inspection
    print(f"\nLoaded {path}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Extract text and label columns
    texts = df[TEXT_COL].astype(str).tolist()
    labels = df[LABEL_COL].astype(int).tolist()

    return texts, labels

if __name__ == "__main__":
    train_texts, train_labels = load_split("train.jsonl")
    val_texts, val_labels = load_split("val.jsonl")

    print("\nNumber of train samples:", len(train_texts))
    print("Number of val samples:", len(val_texts))

    # Print the first sample for inspection
    print("\nFirst train text (first 300 chars):")
    print(train_texts[0][:300], "...")
    print("First train label:", train_labels[0])
