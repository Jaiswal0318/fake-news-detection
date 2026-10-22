
import os
import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Project paths
DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")

FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")


def clean_text(text):
    """Normalize whitespace and remove URLs."""
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    print("Loading datasets...")

    if not os.path.exists(FAKE_PATH) or not os.path.exists(TRUE_PATH):
        raise FileNotFoundError(
            "Place Fake.csv and True.csv inside the data folder."
        )

    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    # Assign labels: 1 = Fake, 0 = Real
    fake_df["label"] = 1
    true_df["label"] = 0

    # Combine title and article text
    for df in (fake_df, true_df):
        if "text" not in df.columns:
            raise ValueError("Each CSV must contain a 'text' column.")

        if "title" not in df.columns:
            df["title"] = ""

        df["content"] = (
            df["title"].fillna("").astype(str)
            + " "
            + df["text"].fillna("").astype(str)
        ).map(clean_text)

    data = pd.concat(
        [
            fake_df[["content", "label"]],
            true_df[["content", "label"]],
        ],
        ignore_index=True,
    )

    data = data[data["content"].str.len() >= 40]
    data = data.drop_duplicates(subset=["content"])
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total usable articles: {len(data):,}")
    print(f"Fake articles: {(data['label'] == 1).sum():,}")
    print(f"Real articles: {(data['label'] == 0).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        data["content"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"],
    )

    print("\nTraining TF-IDF + Logistic Regression model...")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n--- Evaluation on held-out test data ---")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Real News", "Fake News"],
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    main()
