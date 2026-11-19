
import os
import random
import json
from pathlib import Path

os.environ["HF_HOME"] = r"E:\AI_Cache\huggingface"
os.environ["HF_HUB_CACHE"] = r"E:\AI_Cache\huggingface\hub"
os.environ["TORCH_HOME"] = r"E:\AI_Cache\torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

MODEL_PATH = r"E:\AI_Cache\models\SinBERT-small"

DATA_DIR = Path("data/processed/multilingual")

TRAIN_PATH = Path("data/processed/sinhala/few_shot/sinhala_32_shot.csv")
VAL_PATH = DATA_DIR / "bilingual_validation.csv"
TEST_PATH = DATA_DIR / "bilingual_test.csv"

OUTPUT_DIR = Path("models/sinbert_sinhala_32shot")
RESULT_PATH = Path(
    "outputs/results/sinbert_sinhala_32shot_results.json"
)

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 2e-5

LABEL_TO_ID = {"fake": 0, "real": 1}
ID_TO_LABEL = {0: "fake", 1: "real"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sinhala_data(path):
    df = pd.read_csv(path)

    df = df.dropna(
        subset=["clean_text", "label", "language"]
    ).copy()

    df["language"] = (
        df["language"].astype(str).str.lower().str.strip()
    )
    df["label"] = (
        df["label"].astype(str).str.lower().str.strip()
    )

    df = df[
        (df["language"] == "si")
        & df["label"].isin(LABEL_TO_ID)
    ].copy()

    df["clean_text"] = (
        df["clean_text"].astype(str).str.strip()
    )
    df = df[df["clean_text"] != ""]

    return df.reset_index(drop=True)


class NewsDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.encodings = tokenizer(
            df["clean_text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(
            df["label"].map(LABEL_TO_ID).tolist(),
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
        }


def evaluate(model, loader, device):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**inputs)
            predicted = outputs.logits.argmax(dim=1)

            true_labels.extend(
                inputs["labels"].cpu().tolist()
            )
            predictions.extend(
                predicted.cpu().tolist()
            )

    return {
        "accuracy": accuracy_score(
            true_labels, predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
            true_labels, predictions
        ),
        "macro_f1": f1_score(
            true_labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "classification_report": classification_report(
            true_labels,
            predictions,
            labels=[0, 1],
            target_names=["fake", "real"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            true_labels,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }


def main():
    set_seed(SEED)

    device = torch.device("cpu")
    print("Device:", device)

    train_df = load_sinhala_data(TRAIN_PATH)
    val_df = load_sinhala_data(VAL_PATH)
    test_df = load_sinhala_data(TEST_PATH)

    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Testing:", len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        local_files_only=True,
    )

    # Freeze embeddings and first six encoder layers
    for parameter in model.roberta.embeddings.parameters():
        parameter.requires_grad = False

    for layer in model.roberta.encoder.layer[:6]:
        for parameter in layer.parameters():
            parameter.requires_grad = False

    model.to(device)

    train_loader = DataLoader(
        NewsDataset(train_df, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        NewsDataset(val_df, tokenizer),
        batch_size=BATCH_SIZE,
    )
    test_loader = DataLoader(
        NewsDataset(test_df, tokenizer),
        batch_size=BATCH_SIZE,
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    best_f1 = -1.0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()
            outputs = model(**inputs)
            outputs.loss.backward()
            optimizer.step()

            total_loss += outputs.loss.item()

        val_results = evaluate(
            model, val_loader, device
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"Loss: {total_loss / len(train_loader):.4f} "
            f"Validation Macro-F1: "
            f"{val_results['macro_f1']:.4f}"
        )

        if val_results["macro_f1"] > best_f1:
            best_f1 = val_results["macro_f1"]
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nLoading best model...")

    best_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            OUTPUT_DIR,
            local_files_only=True,
        )
    )
    best_model.to(device)

    test_results = evaluate(
        best_model, test_loader, device
    )

    results = {
        "model": "NLPC-UOM/SinBERT-small",
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "best_validation_macro_f1": best_f1,
        "test": test_results,
    }

    with open(
        RESULT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nFINAL TEST RESULTS")
    print(json.dumps(
        test_results,
        indent=2,
        ensure_ascii=False,
    ))

    print("\nModel saved to:", OUTPUT_DIR)
    print("Results saved to:", RESULT_PATH)


if __name__ == "__main__":
    main()
