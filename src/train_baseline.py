"""
Train the full-data baseline model.

This model uses the entire training dataset and is expected to
be more stable than the few-shot models.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from train_few_shot import build_model


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULT_DIR = Path("outputs/results")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load and clean a processed CSV file.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    df = df.dropna(
        subset=["clean_text", "label"]
    )

    df["clean_text"] = (
        df["clean_text"]
        .astype(str)
        .str.strip()
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df = df[
        df["clean_text"] != ""
    ]

    df = df[
        df["label"].isin(
            ["fake", "real"]
        )
    ]

    return df


def main() -> None:
    train_df = load_dataset(
        DATA_DIR / "train_clean.csv"
    )

    test_df = load_dataset(
        DATA_DIR / "test_clean.csv"
    )

    print("Training samples:", len(train_df))
    print("Test samples:", len(test_df))

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts())

    model = build_model(
        c_value=2.0
    )

    print("\nTraining baseline model...")

    model.fit(
        train_df["clean_text"],
        train_df["label"],
    )

    predictions = model.predict(
        test_df["clean_text"]
    )

    accuracy = accuracy_score(
        test_df["label"],
        predictions,
    )

    (
        precision,
        recall,
        f1_score,
        _,
    ) = precision_recall_fscore_support(
        test_df["label"],
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        test_df["label"],
        predictions,
        labels=["fake", "real"],
        zero_division=0,
    )

    matrix = confusion_matrix(
        test_df["label"],
        predictions,
        labels=["fake", "real"],
    )

    print("\nBaseline Accuracy:", round(accuracy, 4))
    print("Baseline Precision:", round(precision, 4))
    print("Baseline Recall:", round(recall, 4))
    print("Baseline F1-score:", round(f1_score, 4))

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix [fake, real]:")
    print(matrix)

    model_path = (
        MODEL_DIR /
        "baseline_tfidf_logistic.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    result_text = (
        f"Baseline Accuracy: {accuracy:.4f}\n"
        f"Baseline Precision: {precision:.4f}\n"
        f"Baseline Recall: {recall:.4f}\n"
        f"Baseline F1-score: {f1_score:.4f}\n\n"
        f"Classification Report:\n"
        f"{report}\n"
        f"Confusion Matrix [fake, real]:\n"
        f"{matrix}\n"
    )

    result_path = (
        RESULT_DIR /
        "baseline_results.txt"
    )

    result_path.write_text(
        result_text,
        encoding="utf-8",
    )

    print("\nBaseline model saved:", model_path)
    print("Results saved:", result_path)


if __name__ == "__main__":
    main()