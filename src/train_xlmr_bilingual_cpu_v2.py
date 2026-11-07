from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

# Keep Hugging Face and PyTorch caches on the E: drive.
os.environ.setdefault("HF_HOME", r"E:\AI_Cache\huggingface")
os.environ.setdefault("HF_HUB_CACHE", r"E:\AI_Cache\huggingface\hub")
os.environ.setdefault("TORCH_HOME", r"E:\AI_Cache\torch")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


MODEL_NAME = "FacebookAI/xlm-roberta-base"

TRAIN_PATH = Path(
    "data/processed/multilingual/bilingual_64_shot_train.csv"
)
VALIDATION_PATH = Path(
    "data/processed/multilingual/bilingual_validation.csv"
)
TEST_PATH = Path(
    "data/processed/multilingual/bilingual_test.csv"
)

MODEL_OUTPUT_DIR = Path("models/xlmr_english_sinhala_cpu_v2")
RESULT_OUTPUT_PATH = Path(
    "outputs/results/xlmr_english_sinhala_cpu_v2_results.json"
)

LABELS = ["fake", "real"]
LABEL_TO_ID = {"fake": 0, "real": 1}
ID_TO_LABEL = {0: "fake", 1: "real"}

MAX_LENGTH = 128
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 5
ENCODER_LEARNING_RATE = 2e-5
CLASSIFIER_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
FROZEN_ENCODER_LAYERS = 6
RANDOM_SEED = 42


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataframe(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset was not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    required_columns = {"clean_text", "label", "language"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{file_path} is missing columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe.dropna(
        subset=["clean_text", "label", "language"]
    ).copy()
    dataframe["clean_text"] = (
        dataframe["clean_text"].astype(str).str.strip()
    )
    dataframe["label"] = (
        dataframe["label"].astype(str).str.lower().str.strip()
    )
    dataframe["language"] = (
        dataframe["language"].astype(str).str.lower().str.strip()
    )
    dataframe = dataframe[
        (dataframe["clean_text"] != "")
        & dataframe["label"].isin(LABELS)
        & dataframe["language"].isin(["en", "si"])
    ].copy()

    return dataframe.reset_index(drop=True)


class EncodedTextDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
    ) -> None:
        self.languages = dataframe["language"].tolist()
        self.labels = torch.tensor(
            dataframe["label"].map(LABEL_TO_ID).tolist(),
            dtype=torch.long,
        )
        self.encodings = tokenizer(
            dataframe["clean_text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
            "language": self.languages[index],
        }


def freeze_lower_layers(model: torch.nn.Module) -> None:
    for parameter in model.roberta.embeddings.parameters():
        parameter.requires_grad = False

    for layer in model.roberta.encoder.layer[
        :FROZEN_ENCODER_LAYERS
    ]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def calculate_metrics(
    true_ids: list[int],
    predicted_ids: list[int],
) -> dict:
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_ids,
            predicted_ids,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            true_ids,
            predicted_ids,
            labels=[0, 1],
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_ids, predicted_ids)
        ),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": confusion_matrix(
            true_ids,
            predicted_ids,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            true_ids,
            predicted_ids,
            labels=[0, 1],
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    losses: list[float] = []
    true_ids: list[int] = []
    predicted_ids: list[int] = []
    languages: list[str] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            losses.append(float(outputs.loss.item()))
            predictions = outputs.logits.argmax(dim=1)

            true_ids.extend(labels.cpu().tolist())
            predicted_ids.extend(predictions.cpu().tolist())
            languages.extend(list(batch["language"]))

    results = {
        "loss": float(np.mean(losses)),
        "overall": calculate_metrics(true_ids, predicted_ids),
        "by_language": {},
    }

    for language in ["en", "si"]:
        indices = [
            index
            for index, value in enumerate(languages)
            if value == language
        ]
        language_true = [true_ids[index] for index in indices]
        language_predictions = [
            predicted_ids[index] for index in indices
        ]
        results["by_language"][language] = calculate_metrics(
            language_true,
            language_predictions,
        )

    results["mean_language_macro_f1"] = float(
        np.mean(
            [
                results["by_language"]["en"]["macro_f1"],
                results["by_language"]["si"]["macro_f1"],
            ]
        )
    )
    return results


def print_evaluation(title: str, results: dict) -> None:
    print("\n" + title)
    print("-" * 60)
    print("Loss:", round(results["loss"], 4))
    print(
        "Overall accuracy:",
        round(results["overall"]["accuracy"], 4),
    )
    print(
        "Overall macro F1:",
        round(results["overall"]["macro_f1"], 4),
    )
    print(
        "English macro F1:",
        round(results["by_language"]["en"]["macro_f1"], 4),
    )
    print(
        "Sinhala macro F1:",
        round(results["by_language"]["si"]["macro_f1"], 4),
    )
    print(
        "Mean language macro F1:",
        round(results["mean_language_macro_f1"], 4),
    )
    print(
        "Overall confusion matrix [fake, real]:",
        results["overall"]["confusion_matrix"],
    )


def main() -> None:
    set_random_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Model:", MODEL_NAME)
    print("Loading datasets...")

    train_dataframe = load_dataframe(TRAIN_PATH)
    validation_dataframe = load_dataframe(VALIDATION_PATH)
    test_dataframe = load_dataframe(TEST_PATH)

    print("Training samples:", len(train_dataframe))
    print("Validation samples:", len(validation_dataframe))
    print("Test samples:", len(test_dataframe))

    print("Loading cached XLM-R tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    print("Loading cached XLM-R model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        local_files_only=True,
    )

    freeze_lower_layers(model)
    model.to(device)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("Total parameters:", f"{total_parameters:,}")
    print("Trainable parameters:", f"{trainable_parameters:,}")
    print("Frozen encoder layers:", FROZEN_ENCODER_LAYERS)

    print("Tokenizing datasets...")
    train_dataset = EncodedTextDataset(train_dataframe, tokenizer)
    validation_dataset = EncodedTextDataset(
        validation_dataframe,
        tokenizer,
    )
    test_dataset = EncodedTextDataset(test_dataframe, tokenizer)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    train_evaluation_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    encoder_parameters = []
    classifier_parameters = []

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter_name.startswith("classifier."):
            classifier_parameters.append(parameter)
        else:
            encoder_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": ENCODER_LEARNING_RATE,
            },
            {
                "params": classifier_parameters,
                "lr": CLASSIFIER_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / GRADIENT_ACCUMULATION_STEPS
    )
    total_training_steps = updates_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_validation_score = -1.0
    epoch_history: list[dict] = []
    training_start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print("\n" + "=" * 60)
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print("=" * 60)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            raw_loss = outputs.loss
            epoch_loss += float(raw_loss.item())
            loss = raw_loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            should_update = (
                step % GRADIENT_ACCUMULATION_STEPS == 0
                or step == len(train_loader)
            )

            if should_update:
                clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0 or step == len(train_loader):
                print(
                    f"Step {step}/{len(train_loader)} "
                    f"- average loss: {epoch_loss / step:.4f}"
                )

        training_results = evaluate_model(
            model,
            train_evaluation_loader,
            device,
        )
        print_evaluation(
            f"TRAINING RESULTS - EPOCH {epoch}",
            training_results,
        )

        validation_results = evaluate_model(
            model,
            validation_loader,
            device,
        )
        print_evaluation(
            f"VALIDATION RESULTS - EPOCH {epoch}",
            validation_results,
        )

        epoch_history.append(
            {
                "epoch": epoch,
                "training_loss": float(epoch_loss / len(train_loader)),
                "training": training_results,
                "validation": validation_results,
            }
        )

        current_score = validation_results[
            "mean_language_macro_f1"
        ]

        if current_score > best_validation_score:
            best_validation_score = current_score
            print("Saving new best bilingual model...")
            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    training_seconds = time.time() - training_start_time

    print("\nLoading best saved model for final test...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_OUTPUT_DIR,
        local_files_only=True,
    )
    best_model.to(device)

    test_results = evaluate_model(
        best_model,
        test_loader,
        device,
    )
    print_evaluation("FINAL TEST RESULTS", test_results)

    final_results = {
        "model_name": MODEL_NAME,
        "device": str(device),
        "training_samples": len(train_dataframe),
        "validation_samples": len(validation_dataframe),
        "test_samples": len(test_dataframe),
        "maximum_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "epochs": NUM_EPOCHS,
        "encoder_learning_rate": ENCODER_LEARNING_RATE,
        "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
        "frozen_encoder_layers": FROZEN_ENCODER_LAYERS,
        "training_minutes": float(training_seconds / 60),
        "best_validation_mean_language_macro_f1": float(
            best_validation_score
        ),
        "epoch_history": epoch_history,
        "test": test_results,
    }

    with RESULT_OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(
            final_results,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nTraining completed successfully.")
    print("Model saved to:", MODEL_OUTPUT_DIR)
    print("Results saved to:", RESULT_OUTPUT_PATH)
    print("Training minutes:", round(training_seconds / 60, 2))


if __name__ == "__main__":
    main()
