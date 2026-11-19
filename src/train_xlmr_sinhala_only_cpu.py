from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

# Store model caches on E: (set before importing transformers).
os.environ["HF_HOME"] = r"E:\AI_Cache\huggingface"
os.environ["HF_HUB_CACHE"] = r"E:\AI_Cache\huggingface\hub"
os.environ["TORCH_HOME"] = r"E:\AI_Cache\torch"
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
DATA_DIR = Path("data/processed/multilingual")
TRAIN_PATH = DATA_DIR / "bilingual_64_shot_train.csv"
VALIDATION_PATH = DATA_DIR / "bilingual_validation.csv"
TEST_PATH = DATA_DIR / "bilingual_test.csv"

# Separate output locations: existing bilingual checkpoints remain untouched.
MODEL_OUTPUT_DIR = Path("models/xlmr_sinhala_only_64shot")
RESULT_OUTPUT_PATH = Path("outputs/results/xlmr_sinhala_only_64shot_results.json")

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


def load_sinhala_dataframe(file_path: Path) -> pd.DataFrame:
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    df = pd.read_csv(file_path)
    required = {"clean_text", "label", "language"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_path} missing columns: {sorted(missing)}")
    df = df.dropna(subset=["clean_text", "label", "language"]).copy()
    for col in ("clean_text", "label", "language"):
        df[col] = df[col].astype(str).str.strip()
    df["label"] = df["label"].str.lower()
    df["language"] = df["language"].str.lower()
    df = df.loc[
        (df["language"] == "si")
        & (df["label"].isin(LABELS))
        & (df["clean_text"] != "")
    ].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No Sinhala fake/real examples in {file_path}")
    return df


class EncodedTextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer) -> None:
        self.labels = torch.tensor(
            dataframe["label"].map(LABEL_TO_ID).tolist(), dtype=torch.long
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
        }


def freeze_lower_layers(model: torch.nn.Module) -> None:
    for parameter in model.roberta.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.roberta.encoder.layer[:FROZEN_ENCODER_LAYERS]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def calculate_metrics(true_ids: list[int], predicted_ids: list[int]) -> dict:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_ids, predicted_ids, labels=[0, 1], average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_ids, predicted_ids, labels=[0, 1], average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "balanced_accuracy": float(balanced_accuracy_score(true_ids, predicted_ids)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": confusion_matrix(true_ids, predicted_ids, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            true_ids,
            predicted_ids,
            labels=[0, 1],
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    losses: list[float] = []
    true_ids: list[int] = []
    predicted_ids: list[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(float(outputs.loss.item()))
            true_ids.extend(labels.cpu().tolist())
            predicted_ids.extend(outputs.logits.argmax(dim=1).cpu().tolist())
    metrics = calculate_metrics(true_ids, predicted_ids)
    return {
        "loss": float(np.mean(losses)),
        "overall": metrics,
        "by_language": {"si": metrics},
        "mean_language_macro_f1": metrics["macro_f1"],
    }


def print_evaluation(title: str, results: dict) -> None:
    metrics = results["by_language"]["si"]
    fake = metrics["classification_report"]["fake"]
    print("\n" + title)
    print("-" * 60)
    print("Loss:", round(results["loss"], 4))
    print("Sinhala accuracy:", round(metrics["accuracy"], 4))
    print("Sinhala macro F1:", round(metrics["macro_f1"], 4))
    print("Fake precision:", round(fake["precision"], 4))
    print("Fake recall:", round(fake["recall"], 4))
    print("Confusion matrix [fake, real]:", metrics["confusion_matrix"])


def main() -> None:
    set_random_seed(RANDOM_SEED)
    device = torch.device("cpu")
    print("Device:", device)
    print("Model:", MODEL_NAME)
    print("Loading Sinhala-only datasets...")
    train_df = load_sinhala_dataframe(TRAIN_PATH)
    validation_df = load_sinhala_dataframe(VALIDATION_PATH)
    test_df = load_sinhala_dataframe(TEST_PATH)
    for name, df in (("Training", train_df), ("Validation", validation_df), ("Test", test_df)):
        print(f"{name} samples: {len(df)}; labels: {df['label'].value_counts().to_dict()}")

    print("Loading cached XLM-R tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        local_files_only=True,
    )
    freeze_lower_layers(model)
    model.to(device)
    print("Frozen encoder layers:", FROZEN_ENCODER_LAYERS)
    print("Trainable parameters:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_dataset = EncodedTextDataset(train_df, tokenizer)
    val_dataset = EncodedTextDataset(validation_df, tokenizer)
    test_dataset = EncodedTextDataset(test_df, tokenizer)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=generator)
    train_eval_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    encoder_parameters = []
    classifier_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("classifier."):
            classifier_parameters.append(parameter)
        else:
            encoder_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": ENCODER_LEARNING_RATE},
            {"params": classifier_parameters, "lr": CLASSIFIER_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    updates_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    total_training_steps = updates_per_epoch * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_training_steps * WARMUP_RATIO),
        num_training_steps=total_training_steps,
    )

    # Avoid accidentally overwriting an existing Sinhala-only experiment.
    if MODEL_OUTPUT_DIR.exists() or RESULT_OUTPUT_PATH.exists():
        raise FileExistsError(
            "Sinhala-only output already exists. Back it up or select a new output name: "
            f"{MODEL_OUTPUT_DIR} / {RESULT_OUTPUT_PATH}"
        )
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    RESULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_validation_score = -1.0
    best_epoch = None
    history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'=' * 20} EPOCH {epoch}/{NUM_EPOCHS} {'=' * 20}")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            total_loss += float(outputs.loss.item())
            (outputs.loss / GRADIENT_ACCUMULATION_STEPS).backward()
            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader):
                clip_grad_norm_((p for p in model.parameters() if p.requires_grad), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 10 == 0 or step == len(train_loader):
                print(f"Step {step}/{len(train_loader)} - average loss: {total_loss / step:.4f}")

        train_results = evaluate_model(model, train_eval_loader, device)
        val_results = evaluate_model(model, val_loader, device)
        print_evaluation(f"TRAIN - EPOCH {epoch}", train_results)
        print_evaluation(f"VALIDATION - EPOCH {epoch}", val_results)
        history.append({
            "epoch": epoch,
            "training_loss": total_loss / len(train_loader),
            "training": train_results,
            "validation": val_results,
        })
        score = val_results["by_language"]["si"]["macro_f1"]
        if score > best_validation_score:
            best_validation_score = score
            best_epoch = epoch
            print("Saving best Sinhala-only model...")
            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print("\nLoading best checkpoint for final test...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_OUTPUT_DIR, local_files_only=True
    ).to(device)
    test_results = evaluate_model(best_model, test_loader, device)
    print_evaluation("FINAL SINHALA TEST RESULTS", test_results)

    results = {
        "model_name": MODEL_NAME,
        "experiment": "Sinhala-only XLM-R 64-shot per class",
        "device": str(device),
        "seed": RANDOM_SEED,
        "training_samples": len(train_df),
        "validation_samples": len(validation_df),
        "test_samples": len(test_df),
        "maximum_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "epochs": NUM_EPOCHS,
        "encoder_learning_rate": ENCODER_LEARNING_RATE,
        "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
        "frozen_encoder_layers": FROZEN_ENCODER_LAYERS,
        "best_epoch": best_epoch,
        "best_validation_mean_language_macro_f1": best_validation_score,
        "training_minutes": (time.time() - start_time) / 60,
        "epoch_history": history,
        "test": test_results,
    }
    with RESULT_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nTraining completed successfully.")
    print("Model saved to:", MODEL_OUTPUT_DIR)
    print("Results saved to:", RESULT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
