"""
Train improved few-shot fake-news classifiers.

The model combines:

1. Word-level TF-IDF features
2. Character-level TF-IDF features
3. Logistic Regression
4. Validation-based hyperparameter selection

Models created:

    models/few_4_shot_model.pkl
    models/few_8_shot_model.pkl
    models/few_16_shot_model.pkl
    models/few_32_shot_model.pkl
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import (
    FeatureUnion,
    Pipeline,
)


DATA_DIR = Path("data/processed")
FEW_SHOT_DIR = DATA_DIR / "few_shot"

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


def build_model(
    c_value: float = 2.0,
) -> Pipeline:
    """
    Build the word + character TF-IDF model.
    """

    feature_extractor = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=20000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    min_df=1,
                    max_df=0.98,
                ),
            ),
            (
                "character_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=20000,
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
        ]
    )

    classifier = LogisticRegression(
        C=c_value,
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    model = Pipeline(
        [
            (
                "features",
                feature_extractor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return model


def load_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load and validate a processed dataset.
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
    validation_df = load_dataset(
        DATA_DIR / "validation_clean.csv"
    )

    test_df = load_dataset(
        DATA_DIR / "test_clean.csv"
    )

    shot_values = [
        4,
        8,
        16,
        32,
    ]

    # Regularization values tested using validation data
    candidate_c_values = [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    ]

    all_results = []

    for k in shot_values:
        print("\n" + "=" * 70)
        print(f"TRAINING {k}-SHOT MODEL")
        print("=" * 70)

        few_shot_path = (
            FEW_SHOT_DIR /
            f"few_{k}_shot.csv"
        )

        train_df = load_dataset(
            few_shot_path
        )

        print("Training samples:", len(train_df))
        print("Training labels:")
        print(train_df["label"].value_counts())

        best_model = None
        best_c_value = None
        best_validation_f1 = -1.0

        # Select the best Logistic Regression C value
        for c_value in candidate_c_values:
            candidate_model = build_model(
                c_value=c_value
            )

            candidate_model.fit(
                train_df["clean_text"],
                train_df["label"],
            )

            validation_predictions = (
                candidate_model.predict(
                    validation_df["clean_text"]
                )
            )

            validation_metrics = (
                precision_recall_fscore_support(
                    validation_df["label"],
                    validation_predictions,
                    average="macro",
                    zero_division=0,
                )
            )

            validation_macro_f1 = (
                validation_metrics[2]
            )

            print(
                f"C={c_value:<4} "
                f"Validation macro F1="
                f"{validation_macro_f1:.4f}"
            )

            if (
                validation_macro_f1
                > best_validation_f1
            ):
                best_model = candidate_model
                best_c_value = c_value
                best_validation_f1 = (
                    validation_macro_f1
                )

        if best_model is None:
            raise RuntimeError(
                "No model was successfully trained."
            )

        print(
            "\nSelected C value:",
            best_c_value,
        )

        print(
            "Best validation macro F1:",
            round(best_validation_f1, 4),
        )

        # Evaluate the selected model on test data
        test_predictions = best_model.predict(
            test_df["clean_text"]
        )

        test_accuracy = accuracy_score(
            test_df["label"],
            test_predictions,
        )

        (
            test_precision,
            test_recall,
            test_f1,
            _,
        ) = precision_recall_fscore_support(
            test_df["label"],
            test_predictions,
            average="weighted",
            zero_division=0,
        )

        report = classification_report(
            test_df["label"],
            test_predictions,
            labels=["fake", "real"],
            zero_division=0,
        )

        matrix = confusion_matrix(
            test_df["label"],
            test_predictions,
            labels=["fake", "real"],
        )

        print("\nTest Accuracy:", round(test_accuracy, 4))
        print("Test Weighted F1:", round(test_f1, 4))

        print("\nClassification Report:")
        print(report)

        print("Confusion Matrix [fake, real]:")
        print(matrix)

        model_path = (
            MODEL_DIR /
            f"few_{k}_shot_model.pkl"
        )

        joblib.dump(
            best_model,
            model_path,
        )

        metadata = {
            "model_name": f"few_{k}_shot_model",
            "classes": [
                str(label)
                for label in best_model.classes_
            ],
            "training_samples": len(train_df),
            "examples_per_class": k,
            "best_C": best_c_value,
            "validation_macro_f1": float(
                best_validation_f1
            ),
            "test_accuracy": float(
                test_accuracy
            ),
            "test_weighted_f1": float(
                test_f1
            ),
            "recommended_uncertain_threshold": 0.60,
        }

        metadata_path = (
            MODEL_DIR /
            f"few_{k}_shot_model.json"
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        all_results.append(
            {
                "shot": k,
                "training_samples": len(train_df),
                "best_C": best_c_value,
                "validation_macro_f1": (
                    best_validation_f1
                ),
                "accuracy": test_accuracy,
                "precision": test_precision,
                "recall": test_recall,
                "f1_score": test_f1,
            }
        )

        print("\nModel saved:", model_path)
        print("Metadata saved:", metadata_path)

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        RESULT_DIR /
        "few_shot_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("ALL FEW-SHOT RESULTS")
    print("=" * 70)

    print(results_df.to_string(index=False))

    print("\nResults saved:", results_path)


if __name__ == "__main__":
    main()