"""
train_model.py

One-time training script.

- Downloads the 'Updated Resume Dataset' from Kaggle using kagglehub.
- Trains a TF-IDF + Logistic Regression classifier to predict resume Category from text.
- Saves the fitted vectorizer, classifier, and label encoder into the models/ folder.
"""

import os
import re
import string

import kagglehub  # pip install kagglehub
import joblib     # pip install joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# ---------- Utility: basic text cleaning ----------

def clean_text(text: str) -> str:
    """
    Very simple text-preprocessing function.

    - Lowercases text
    - Removes HTML tags
    - Removes URLs
    - Removes numbers
    - Removes punctuation
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Lowercase
    text = text.lower()

    # Remove digits
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    # Make sure output directory exists
    os.makedirs("models", exist_ok=True)

    # ---------- 1. Download Kaggle dataset using kagglehub ----------
    print("Downloading dataset from Kaggle via kagglehub...")
    dataset_path = kagglehub.dataset_download("jillanisofttech/updated-resume-dataset")
    print("Dataset downloaded to:", dataset_path)

    # The CSV file is named 'UpdatedResumeDataSet.csv'
    csv_path = os.path.join(dataset_path, "UpdatedResumeDataSet.csv")

    # Load the dataset into a DataFrame
    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path)

    # Dataset has columns "Category" (label) and "Resume" (text).
    print("DataFrame head:")
    print(df.head())

    # Drop rows with missing values in either column
    df = df.dropna(subset=["Category", "Resume"])

    # ---------- 2. Clean text ----------
    print("Cleaning resume text...")
    df["clean_resume"] = df["Resume"].apply(clean_text)

    # ---------- 3. Encode target labels ----------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Category"])

    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print("Label mapping:", label_mapping)

    # ---------- 4. Create train/test split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_resume"], y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- 5. Fit TF-IDF vectorizer ----------
    print("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # ---------- 6. Train classifier ----------
    print("Training Logistic Regression classifier...")

    # <<< CHANGED HERE >>>
    # Remove multi_class kwarg (and keep config simple for compatibility).
    # Your version of scikit-learn complains that 'multi_class' is an unexpected argument.[web:35]
    clf = LogisticRegression(
        max_iter=1000,
        # If this still errors in your environment, try the simplest:
        # clf = LogisticRegression()
    )
    # <<< END CHANGE >>>

    clf.fit(X_train_tfidf, y_train)

    # ---------- 7. Evaluate on hold-out test set ----------
    print("Evaluating model on test split...")
    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # ---------- 8. Save artifacts ----------
    print("Saving model artifacts to models/ ...")
    joblib.dump(tfidf, os.path.join("models", "tfidf_vectorizer.pkl"))
    joblib.dump(clf, os.path.join("models", "classifier.pkl"))
    joblib.dump(label_encoder, os.path.join("models", "label_encoder.pkl"))
    print("Done. Artifacts saved in ./models")


if __name__ == "__main__":
    main()