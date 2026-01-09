import pandas as pd

def main():
    df_train = pd.read_json("train.jsonl", lines=True)
    df_val = pd.read_json("val.jsonl", lines=True)

    print("=== Train basic info ===")
    print(df_train.info())

    print("\nLabel distribution (train):")
    print(df_train["label"].value_counts())

    print("\nGenre distribution (train):")
    print(df_train["genre"].value_counts())

    print("\nModel distribution (train, top 15):")
    print(df_train["model"].value_counts().head(15))

    # text lenth
    df_train["len"] = df_train["text"].str.len()
    print("\nText length statistics (train):")
    print(df_train["len"].describe())

    print("\nAverage length per label (train):")
    print(df_train.groupby("label")["len"].mean())

if __name__ == "__main__":
    main()
