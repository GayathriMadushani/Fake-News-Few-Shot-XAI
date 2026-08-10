from pathlib import Path

import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments


MODEL_NAME = "sentence-transformers/all-distilroberta-v1"

TRAIN_PATH = Path(
    "data/processed/few_shot/few_4_shot.csv"
)

VALIDATION_PATH = Path(
    "data/processed/validation_clean.csv"
)

OUTPUT_DIR = Path(
    "outputs/setfit_test_checkpoints"
)

LABELS = ["fake", "real"]

LABEL_TO_ID = {
    "fake": 0,
    "real": 1,
}


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

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
        df["label"].isin(LABEL_TO_ID)
    ].copy()

    df["label"] = (
        df["label"]
        .map(LABEL_TO_ID)
        .astype(int)
    )

    return df


def to_dataset(df: pd.DataFrame) -> Dataset:
    dataset_df = df.rename(
        columns={"clean_text": "text"}
    )

    return Dataset.from_pandas(
        dataset_df[["text", "label"]],
        preserve_index=False,
    )


def main() -> None:
    print("Loading datasets...")

    train_df = load_data(TRAIN_PATH)
    validation_df = load_data(
        VALIDATION_PATH
    )

    # Use only 20 validation examples from
    # each class for this short compatibility test.
    validation_sample = (
        validation_df
        .groupby(
            "label",
            group_keys=False,
        )
        .sample(
            n=20,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    train_dataset = to_dataset(train_df)
    validation_dataset = to_dataset(
        validation_sample
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(validation_dataset),
    )

    print("Loading cached DistilRoBERTa...")

    model = SetFitModel.from_pretrained(
        MODEL_NAME,
        labels=LABELS,
        device="cpu",
        local_files_only=True,
    )

    model.model_body.max_seq_length = 256

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        batch_size=4,
        max_steps=2,
        num_iterations=2,
        max_length=256,
        body_learning_rate=2e-5,
        warmup_proportion=0.0,
        use_amp=False,
        seed=42,
        report_to="none",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting short CPU training test...")

    trainer.train()

    print("Evaluating compatibility test...")

    metrics = trainer.evaluate(
        validation_dataset
    )

    print("\nSetFit test completed successfully")
    print("Validation metrics:", metrics)

    example_predictions = model.predict(
        validation_sample[
            "clean_text"
        ].head(4).tolist()
    )

    print(
        "Example predictions:",
        example_predictions,
    )


if __name__ == "__main__":
    main()