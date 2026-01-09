import pandas as pd

def main():
    # read jsonl
    df_train = pd.read_json("train.jsonl", lines=True)
    df_val = pd.read_json("val.jsonl", lines=True)

    print("Train shape:", df_train.shape)
    print("Val shape:", df_val.shape)

    print("\nTrain columns:", df_train.columns.tolist())

    print("\nTrain head():")
    print(df_train.head())

    if "label" in df_train.columns:
        print("\nTrain label value counts:")
        print(df_train["label"].value_counts())

    if "label" in df_val.columns:
        print("\nVal label value counts:")
        print(df_val["label"].value_counts())

if __name__ == "__main__":
    main()
