"""
Step 4: Multi-seed SetFit experiment (original vs masked text).

Run from the project root. Needs in src/:
    audit_shortcuts.py, audit_shortcut_groups.py, run_fewshot_seeds.py,
    create_few_shot.py

    python src/run_setfit_seeds.py

It trains SetFit (DistilRoBERTa) for each shot size and seed on
  - original text and
  - masked text (shortcut groups removed),
and tests on the matching test version. Sampling uses "random_len30"
(random articles with at least 30 words), the best strategy in step 3.

This is SLOW on CPU. Results are saved after every run, so you can stop
with Ctrl+C and run the same command again to resume.
Hyperparameters match your train_setfit_few_shot.py so results are
comparable with your earlier single-seed SetFit table.
"""

from __future__ import annotations

import gc
import os
import random
import time
from pathlib import Path

# Prevent accidental Hugging Face downloads (same as your other scripts).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

from audit_shortcut_groups import GROUPS, apply_mask, compile_group
from audit_shortcuts import load
from run_fewshot_seeds import draw_sample

MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
LABELS = ["fake", "real"]
LABEL_TO_ID = {"fake": 0, "real": 1}

SHOTS = [4]
SEEDS = [0]
STRATEGY = "random_len30"
VERSIONS = ["original", "masked"]

MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 4
NUM_EPOCHS = 1
NUM_ITERATIONS = 5
LEARNING_RATE = 2e-5

OUT_DIR = Path("outputs/results")
CHECKPOINT_DIR = Path("outputs/setfit_seed_runs")
RAW_PATH = OUT_DIR / "setfit_seeds_raw.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(texts: pd.Series, labels: pd.Series) -> Dataset:
    frame = pd.DataFrame(
        {
            "text": texts.astype(str).tolist(),
            "label": labels.map(LABEL_TO_ID).astype(int).tolist(),
        }
    )
    return Dataset.from_pandas(frame, preserve_index=False)


def train_and_test(
    train_texts: pd.Series,
    train_labels: pd.Series,
    test_texts: pd.Series,
    test_labels: pd.Series,
    seed: int,
) -> tuple[float, float, float]:
    set_random_seed(seed)

    model = SetFitModel.from_pretrained(
        MODEL_NAME,
        labels=LABELS,
        device="cpu",
        local_files_only=True,
    )
    model.model_body.max_seq_length = MAX_SEQUENCE_LENGTH

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        num_iterations=NUM_ITERATIONS,
        max_length=MAX_SEQUENCE_LENGTH,
        body_learning_rate=LEARNING_RATE,
        warmup_proportion=0.1,
        use_amp=False,
        seed=seed,
        report_to="none",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=make_dataset(train_texts, train_labels),
    )

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60

    predictions = [
        str(p) for p in model.predict(test_texts.astype(str).tolist())
    ]
    truth = test_labels.astype(str).tolist()

    accuracy = accuracy_score(truth, predictions)
    macro_f1 = f1_score(truth, predictions, average="macro")

    del trainer
    del model
    gc.collect()

    return accuracy, macro_f1, minutes


def print_summary(raw: pd.DataFrame) -> None:
    summary = (
        raw.groupby(["text_version", "shot"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["accuracy"] = summary.apply(
        lambda r: f"{r['mean']:.3f} +- {r['std']:.3f} (n={int(r['count'])})",
        axis=1,
    )
    table = summary.pivot(
        index="text_version", columns="shot", values="accuracy"
    )
    print("\nSetFit test accuracy, mean +- std over seeds")
    print("=" * 70)
    print(table.to_string())


def main() -> None:
    train_df = load("train_clean.csv")
    test_df = load("test_clean.csv")
    train_df["n_words"] = train_df["clean_text"].str.split().str.len()

    all_words = [w for words, _ in GROUPS.values() for w in words]
    all_patterns = [p for _, patterns in GROUPS.values() for p in patterns]
    pattern = compile_group(all_words, all_patterns)

    test_texts = {
        "original": test_df["clean_text"],
        "masked": apply_mask(test_df["clean_text"], pattern),
    }

    rows: list[dict] = []
    done: set[tuple[int, int, str]] = set()
    if RAW_PATH.exists():
        previous = pd.read_csv(RAW_PATH)
        rows = previous.to_dict("records")
        done = {
            (int(r.shot), int(r.seed), str(r.text_version))
            for r in previous.itertuples()
        }
        print(f"Resuming: {len(done)} runs already completed.")

    total = len(SHOTS) * len(SEEDS) * len(VERSIONS)

    for shot in SHOTS:
        for seed in SEEDS:
            sample = draw_sample(train_df, shot, seed, STRATEGY)
            train_texts = {
                "original": sample["clean_text"],
                "masked": apply_mask(sample["clean_text"], pattern),
            }

            for version in VERSIONS:
                if (shot, seed, version) in done:
                    continue

                print(
                    f"\n[{len(rows) + 1}/{total}] shot={shot} "
                    f"seed={seed} text={version}"
                )

                accuracy, macro_f1, minutes = train_and_test(
                    train_texts[version],
                    sample["label"],
                    test_texts[version],
                    test_df["label"],
                    seed,
                )

                print(
                    f"accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} "
                    f"time={minutes:.1f} min"
                )

                rows.append(
                    {
                        "shot": shot,
                        "seed": seed,
                        "strategy": STRATEGY,
                        "text_version": version,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "minutes": minutes,
                    }
                )
                pd.DataFrame(rows).to_csv(RAW_PATH, index=False)

    print_summary(pd.DataFrame(rows))
    print(f"\nSaved: {RAW_PATH}")


if __name__ == "__main__":
    main()
