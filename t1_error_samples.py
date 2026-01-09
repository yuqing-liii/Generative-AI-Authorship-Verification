# t1_error_samples.py
"""
Export typical false positives / false negatives for the Task 1 RoBERTa model
on the PAN Task 1 validation set.

Inputs:
  - val.jsonl
  - t1_bert_outputs/t1_bert_val_pred.csv (columns: id, label, prob_ai)

Output:
  - t1_bert_outputs/t1_error_samples.csv
"""

import pandas as pd

VAL_JSON = "val.jsonl"
PRED_CSV = "t1_bert_outputs/t1_bert_val_pred.csv"
OUTPUT_CSV = "t1_bert_outputs/t1_error_samples.csv"

# Name of the probability column in the prediction CSV
PROB_COL = "prob_ai"


def main() -> None:
    # Load ground-truth validation data
    val_df = pd.read_json(VAL_JSON, lines=True)

    # Load model predictions
    pred_df = pd.read_csv(PRED_CSV)

    # Basic checks on prediction columns
    required_cols = {"id", "label", PROB_COL}
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing columns in prediction CSV: {missing}")

    # Merge on id to be robust against any ordering differences
    merged_df = val_df.merge(
        pred_df[["id", "label", PROB_COL]].rename(
            columns={"label": "label_check", PROB_COL: "p_ai"}
        ),
        on="id",
        how="inner",
    )

    if len(merged_df) != len(val_df):
        raise ValueError(
            f"After merging on 'id' we have {len(merged_df)} rows, "
            f"but val.jsonl has {len(val_df)} rows. "
            "Please check that the IDs match between files."
        )

    # Use ground-truth label from val.jsonl
    df = merged_df
    df["y_true"] = df["label"]

    # Predicted label from probability with threshold 0.5
    df["y_pred"] = (df["p_ai"] >= 0.5).astype(int)

    # False positives: true = 0 (human), pred = 1 (AI), sorted by p_ai descending
    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].copy()
    fp = fp.sort_values("p_ai", ascending=False).head(20)

    # False negatives: true = 1 (AI), pred = 0 (human), sorted by p_ai ascending
    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].copy()
    fn = fn.sort_values("p_ai", ascending=True).head(20)

    out_df = pd.concat(
        [
            fp.assign(error_type="FP_human_as_AI"),
            fn.assign(error_type="FN_AI_as_human"),
        ],
        ignore_index=True,
    )

    # Save only the most important fields; add others if you like
    out_df[["error_type", "y_true", "p_ai", "text"]].to_csv(
        OUTPUT_CSV, index=False
    )
    print("Saved error samples to:", OUTPUT_CSV)
    print("FP:", len(fp), "FN:", len(fn))


if __name__ == "__main__":
    main()
