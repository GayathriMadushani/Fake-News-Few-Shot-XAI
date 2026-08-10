from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline


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

    return Pipeline(
        [
            ("features", feature_extractor),
            ("classifier", classifier),
        ]
    )


def load_dataset(
    file_path: Path,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {file_path}"
        )

    df = pd.read_csv(file_path)

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
        df["label"].isin(["fake", "real"])
    ]

    return df


def main() -> None:
    test_df = load_dataset(
        DATA_DIR / "test_clean.csv"
    )

    shot_values = [
        4,
        8,
        16,
        32,
    ]

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
            FEW_SHOT_DIR
            / f"few_{k}_shot.csv"
        )

        train_df = load_dataset(
            few_shot_path
        )

        print(
            "Training samples:",
            len(train_df),
        )
        print("Training labels:")
        print(
            train_df["label"].value_counts()
        )

        minimum_class_samples = int(
            train_df["label"]
            .value_counts()
            .min()
        )

        cv_splits = min(
            4,
            minimum_class_samples,
        )

        cv_repeats = 5

        cross_validation = (
            RepeatedStratifiedKFold(
                n_splits=cv_splits,
                n_repeats=cv_repeats,
                random_state=42,
            )
        )

        best_c_value = None
        best_cv_f1_mean = -1.0
        best_cv_f1_std = 0.0

        # Select C using only the few-shot
        # training samples.
        for c_value in candidate_c_values:
            candidate_model = build_model(
                c_value=c_value
            )

            cv_scores = cross_val_score(
                candidate_model,
                train_df["clean_text"],
                train_df["label"],
                scoring="f1_macro",
                cv=cross_validation,
                n_jobs=-1,
            )

            cv_f1_mean = float(
                np.mean(cv_scores)
            )

            cv_f1_std = float(
                np.std(cv_scores)
            )

            print(
                f"C={c_value:<4} "
                f"CV macro F1="
                f"{cv_f1_mean:.4f} "
                f"(+/- {cv_f1_std:.4f})"
            )

            if cv_f1_mean > best_cv_f1_mean:
                best_c_value = c_value
                best_cv_f1_mean = cv_f1_mean
                best_cv_f1_std = cv_f1_std

        if best_c_value is None:
            raise RuntimeError(
                "No C value was selected."
            )

        print(
            "\nSelected C value:",
            best_c_value,
        )

        print(
            "Best CV macro F1:",
            round(best_cv_f1_mean, 4),
        )

        print(
            "CV standard deviation:",
            round(best_cv_f1_std, 4),
        )

        # Train the selected model using all
        # available few-shot training samples.
        best_model = build_model(
            c_value=best_c_value
        )

        best_model.fit(
            train_df["clean_text"],
            train_df["label"],
        )

        # Use the test set only for final evaluation.
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

        print(
            "\nTest Accuracy:",
            round(test_accuracy, 4),
        )

        print(
            "Test Weighted F1:",
            round(test_f1, 4),
        )

        print("\nClassification Report:")
        print(report)

        print(
            "Confusion Matrix [fake, real]:"
        )
        print(matrix)

        model_path = (
            MODEL_DIR
            / f"few_{k}_shot_model.pkl"
        )

        joblib.dump(
            best_model,
            model_path,
        )

        metadata = {
            "model_name": (
                f"few_{k}_shot_model"
            ),
            "classes": [
                str(label)
                for label in best_model.classes_
            ],
            "training_samples": len(
                train_df
            ),
            "examples_per_class": k,
            "best_C": best_c_value,
            "cv_method": (
                "RepeatedStratifiedKFold"
            ),
            "cv_splits": cv_splits,
            "cv_repeats": cv_repeats,
            "cv_macro_f1_mean": (
                best_cv_f1_mean
            ),
            "cv_macro_f1_std": (
                best_cv_f1_std
            ),
            "test_accuracy": float(
                test_accuracy
            ),
            "test_weighted_f1": float(
                test_f1
            ),
        }

        metadata_path = (
            MODEL_DIR
            / f"few_{k}_shot_model.json"
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
                "training_samples": len(
                    train_df
                ),
                "best_C": best_c_value,
                "cv_macro_f1_mean": (
                    best_cv_f1_mean
                ),
                "cv_macro_f1_std": (
                    best_cv_f1_std
                ),
                "accuracy": test_accuracy,
                "precision": test_precision,
                "recall": test_recall,
                "f1_score": test_f1,
            }
        )

        print(
            "\nModel saved:",
            model_path,
        )

        print(
            "Metadata saved:",
            metadata_path,
        )

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        RESULT_DIR
        / "few_shot_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("ALL FEW-SHOT RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nResults saved:",
        results_path,
    )


if __name__ == "__main__":
    main()