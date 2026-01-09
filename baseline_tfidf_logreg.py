import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

TEXT_COL = "text"
LABEL_COL = "label"

def load_split(path):
    df = pd.read_json(path, lines=True)
    texts = df[TEXT_COL].astype(str).tolist()
    labels = df[LABEL_COL].astype(int).tolist()
    return texts, labels

def main():
    print("Loading data...")
    train_texts, train_labels = load_split("train.jsonl")
    val_texts, val_labels = load_split("val.jsonl")

    print("Number of train samples:", len(train_texts))
    print("Number of val samples:", len(val_texts))

    print("\nFitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=50000, 
        ngram_range=(1, 2), 
        lowercase=True
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )
    clf.fit(X_train, train_labels)

    print("\nEvaluating on validation set...")
    y_pred = clf.predict(X_val)
    acc = accuracy_score(val_labels, y_pred)
    print(f"Validation accuracy: {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(val_labels, y_pred))

if __name__ == "__main__":
    main()
