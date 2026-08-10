from __future__ import annotations

import gc
import json
import os
import random
import time
from pathlib import Path

# Prevent accidental Hugging Face downloads.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MODEL_NAME = (
    "sentence-transformers/all-distilroberta-v1"
)

DATA_DIR = Path("data/processed")
FEW_SHOT_DIR = DATA_DIR / "few_shot"

VALIDATION_PATH = (
    DATA_DIR / "validation_clean.csv"
)

TEST_PATH = (
    DATA_DIR / "test_clean.csv"
)

MODEL_DIR = Path("models")
RESULT_DIR = Path("outputs/results")
CHECKPOINT_DIR = Path(
    "outputs/setfit_checkpoints"
)

SHOT_VALUES = [4, 8, 16, 32]

LABELS = ["fake", "real"]

LABEL_TO_ID = {
    "fake": 0,
    "real": 1,
}

MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 4
NUM_EPOCHS = 1
NUM_ITERATIONS = 5
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

RESULTS_PATH = (
    RESULT_DIR
    / "setfit_distilroberta_results.csv"
)


# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_dataset_file(
    file_path: Path,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {
        "clean_text",
        "label",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns in {file_path}: "
            f"{sorted(missing_columns)}"
        )

    df = df.dropna(
        subset=["clean_text", "label"]
    ).copy()

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
    ].copy()

    df = df[
        df["label"].isin(LABELS)
    ].copy()

    return df.reset_index(drop=True)


def create_training_dataset(
    df: pd.DataFrame,
) -> Dataset:
    training_df = df[
        ["clean_text", "label"]
    ].copy()

    training_df["label"] = (
        training_df["label"]
        .map(LABEL_TO_ID)
        .astype(int)
    )

    training_df = training_df.rename(
        columns={"clean_text": "text"}
    )

    return Dataset.from_pandas(
        training_df[["text", "label"]],
        preserve_index=False,
    )


def validate_few_shot_data(
    df: pd.DataFrame,
    shot: int,
) -> None:
    label_counts = (
        df["label"]
        .value_counts()
        .to_dict()
    )

    for label in LABELS:
        actual_count = int(
            label_counts.get(label, 0)
        )

        if actual_count != shot:
            raise ValueError(
                f"{shot}-shot dataset should "
                f"contain {shot} '{label}' "
                f"examples, but found "
                f"{actual_count}."
            )


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_model(
    model: SetFitModel,
    df: pd.DataFrame,
) -> tuple[dict, dict, list]:
    true_labels = (
        df["label"]
        .astype(str)
        .tolist()
    )

    texts = (
        df["clean_text"]
        .astype(str)
        .tolist()
    )

    predictions = model.predict(texts)

    predicted_labels = [
        str(prediction)
        for prediction in predictions
    ]

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=LABELS,
    )

    metrics = {
        "accuracy": float(accuracy),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(macro_f1),
    }

    return (
        metrics,
        report,
        matrix.tolist(),
    )


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

def main() -> None:
    set_random_seed(RANDOM_SEED)

    print("Loading validation dataset...")

    validation_df = load_dataset_file(
        VALIDATION_PATH
    )

    print("Loading test dataset...")

    test_df = load_dataset_file(
        TEST_PATH
    )

    print(
        "Validation samples:",
        len(validation_df),
    )

    print(
        "Test samples:",
        len(test_df),
    )

    all_results = []

    for shot in SHOT_VALUES:
        print("\n" + "=" * 70)
        print(
            f"TRAINING SETFIT "
            f"DISTILROBERTA {shot}-SHOT MODEL"
        )
        print("=" * 70)

        set_random_seed(RANDOM_SEED)

        training_path = (
            FEW_SHOT_DIR
            / f"few_{shot}_shot.csv"
        )

        train_df = load_dataset_file(
            training_path
        )

        validate_few_shot_data(
            train_df,
            shot,
        )

        print(
            "Training samples:",
            len(train_df),
        )

        print("Training labels:")
        print(
            train_df["label"].value_counts()
        )

        train_dataset = (
            create_training_dataset(train_df)
        )

        print(
            "Loading cached DistilRoBERTa..."
        )

        model = SetFitModel.from_pretrained(
            MODEL_NAME,
            labels=LABELS,
            device="cpu",
            local_files_only=True,
        )

        model.model_body.max_seq_length = (
            MAX_SEQUENCE_LENGTH
        )

        shot_checkpoint_dir = (
            CHECKPOINT_DIR
            / f"{shot}_shot"
        )

        training_args = TrainingArguments(
            output_dir=str(
                shot_checkpoint_dir
            ),
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            num_iterations=NUM_ITERATIONS,
            max_length=MAX_SEQUENCE_LENGTH,
            body_learning_rate=(
                LEARNING_RATE
            ),
            warmup_proportion=0.1,
            use_amp=False,
            seed=RANDOM_SEED,
            report_to="none",
            logging_strategy="steps",
            logging_steps=5,
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        print("Starting CPU training...")

        start_time = time.time()

        trainer.train()

        training_seconds = (
            time.time() - start_time
        )

        print(
            "\nTraining time in minutes:",
            round(training_seconds / 60, 2),
        )

        print(
            "\nEvaluating validation set..."
        )

        (
            validation_metrics,
            validation_report,
            validation_matrix,
        ) = evaluate_model(
            model,
            validation_df,
        )

        print("Evaluating test set...")

        (
            test_metrics,
            test_report,
            test_matrix,
        ) = evaluate_model(
            model,
            test_df,
        )

        print(
            "\nValidation accuracy:",
            round(
                validation_metrics[
                    "accuracy"
                ],
                4,
            ),
        )

        print(
            "Validation macro F1:",
            round(
                validation_metrics[
                    "macro_f1"
                ],
                4,
            ),
        )

        print(
            "Test accuracy:",
            round(
                test_metrics["accuracy"],
                4,
            ),
        )

        print(
            "Test macro F1:",
            round(
                test_metrics["macro_f1"],
                4,
            ),
        )

        print(
            "Test weighted F1:",
            round(
                test_metrics[
                    "weighted_f1"
                ],
                4,
            ),
        )

        print(
            "\nTest confusion matrix "
            "[fake, real]:"
        )

        print(
            np.array(test_matrix)
        )

        model_path = (
            MODEL_DIR
            / (
                "setfit_distilroberta_"
                f"{shot}_shot"
            )
        )

        print(
            "\nSaving model to:",
            model_path,
        )

        model.save_pretrained(
            str(model_path)
        )

        details = {
            "model_name": MODEL_NAME,
            "shot": shot,
            "training_samples": len(
                train_df
            ),
            "labels": LABELS,
            "random_seed": RANDOM_SEED,
            "maximum_sequence_length": (
                MAX_SEQUENCE_LENGTH
            ),
            "batch_size": BATCH_SIZE,
            "number_of_epochs": (
                NUM_EPOCHS
            ),
            "number_of_iterations": (
                NUM_ITERATIONS
            ),
            "body_learning_rate": (
                LEARNING_RATE
            ),
            "training_seconds": float(
                training_seconds
            ),
            "validation_metrics": (
                validation_metrics
            ),
            "validation_report": (
                validation_report
            ),
            "validation_confusion_matrix": (
                validation_matrix
            ),
            "test_metrics": test_metrics,
            "test_report": test_report,
            "test_confusion_matrix": (
                test_matrix
            ),
            "confusion_matrix_labels": (
                LABELS
            ),
        }

        details_path = (
            RESULT_DIR
            / (
                "setfit_distilroberta_"
                f"{shot}_shot_details.json"
            )
        )

        details_path.write_text(
            json.dumps(
                details,
                indent=2,
            ),
            encoding="utf-8",
        )

        all_results.append(
            {
                "shot": shot,
                "training_samples": len(
                    train_df
                ),
                "training_minutes": (
                    training_seconds / 60
                ),
                "validation_accuracy": (
                    validation_metrics[
                        "accuracy"
                    ]
                ),
                "validation_macro_f1": (
                    validation_metrics[
                        "macro_f1"
                    ]
                ),
                "validation_weighted_f1": (
                    validation_metrics[
                        "weighted_f1"
                    ]
                ),
                "test_accuracy": (
                    test_metrics["accuracy"]
                ),
                "test_precision": (
                    test_metrics[
                        "weighted_precision"
                    ]
                ),
                "test_recall": (
                    test_metrics[
                        "weighted_recall"
                    ]
                ),
                "test_weighted_f1": (
                    test_metrics[
                        "weighted_f1"
                    ]
                ),
                "test_macro_f1": (
                    test_metrics["macro_f1"]
                ),
            }
        )

        # Save after every completed model.
        results_df = pd.DataFrame(
            all_results
        )

        results_df.to_csv(
            RESULTS_PATH,
            index=False,
        )

        print(
            "Model saved:",
            model_path,
        )

        print(
            "Detailed results saved:",
            details_path,
        )

        print(
            "Combined results updated:",
            RESULTS_PATH,
        )

        del trainer
        del model
        gc.collect()

    final_results_df = pd.DataFrame(
        all_results
    )

    print("\n" + "=" * 70)
    print("ALL SETFIT DISTILROBERTA RESULTS")
    print("=" * 70)

    print(
        final_results_df.to_string(
            index=False
        )
    )

    print(
        "\nFinal results saved:",
        RESULTS_PATH,
    )

    print(
        "\nAll SetFit training completed "
        "successfully."
    )


if __name__ == "__main__":
    main()