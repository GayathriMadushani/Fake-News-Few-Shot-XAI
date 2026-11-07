from pathlib import Path
import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

INPUT_PATH = Path(
    "data/raw/sinhala/Corpus.xlsx"
)

OUTPUT_DIR = Path(
    "data/processed/sinhala"
)

ALL_OUTPUT_PATH = (
    OUTPUT_DIR / "sinhala_all_clean.csv"
)

TRAIN_OUTPUT_PATH = (
    OUTPUT_DIR / "sinhala_train_clean.csv"
)

VALIDATION_OUTPUT_PATH = (
    OUTPUT_DIR / "sinhala_validation_clean.csv"
)

TEST_OUTPUT_PATH = (
    OUTPUT_DIR / "sinhala_test_clean.csv"
)

RANDOM_SEED = 42


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text: object) -> str:
    """Clean text while preserving Sinhala characters."""

    if pd.isna(text):
        return ""

    cleaned = unicodedata.normalize(
        "NFC",
        str(text),
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    )

    return cleaned.strip()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {INPUT_PATH}"
        )

    print("Reading Sinhala Excel dataset...")

    dataframe = pd.read_excel(
        INPUT_PATH,
        engine="openpyxl",
    )

    print("Original rows:", len(dataframe))
    print("Original columns:", dataframe.columns.tolist())

    required_columns = {
        "domain",
        "datestamp",
        "type",
        "content",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe[
        [
            "domain",
            "datestamp",
            "type",
            "content",
        ]
    ].copy()

    # Convert labels such as False into FALSE.
    dataframe["original_type"] = (
        dataframe["type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    print("\nOriginal label counts:")
    print(
        dataframe["original_type"]
        .value_counts(dropna=False)
    )

    # Binary misinformation mapping.
    label_map = {
        "CREDIBLE": "real",
        "FALSE": "fake",
        "PARTIAL": "fake",
    }

    dataframe["label"] = (
        dataframe["original_type"]
        .map(label_map)
    )

    # UNCERTAIN and unknown labels are excluded.
    dataframe = dataframe.dropna(
        subset=["label"]
    ).copy()

    dataframe["clean_text"] = (
        dataframe["content"]
        .apply(clean_text)
    )

    dataframe = dataframe[
        dataframe["clean_text"] != ""
    ].copy()

    # Remove conflicting duplicate texts.
    label_counts_per_text = (
        dataframe
        .groupby("clean_text")["label"]
        .nunique()
    )

    conflicting_texts = set(
        label_counts_per_text[
            label_counts_per_text > 1
        ].index
    )

    print(
        "\nConflicting duplicate texts removed:",
        len(conflicting_texts),
    )

    dataframe = dataframe[
        ~dataframe["clean_text"].isin(
            conflicting_texts
        )
    ].copy()

    before_duplicates = len(dataframe)

    dataframe = dataframe.drop_duplicates(
        subset=["clean_text"],
        keep="first",
    ).copy()

    print(
        "Exact duplicate rows removed:",
        before_duplicates - len(dataframe),
    )

    dataframe["language"] = "si"

    final_dataframe = dataframe[
        [
            "clean_text",
            "label",
            "language",
            "original_type",
            "domain",
            "datestamp",
        ]
    ].reset_index(drop=True)

    print("\nClean binary label counts:")
    print(
        final_dataframe["label"]
        .value_counts()
    )

    print("\nOriginal types retained:")
    print(
        final_dataframe["original_type"]
        .value_counts()
    )

    # 20% test; remaining data becomes 64% train and 16% validation.
    train_validation_df, test_df = (
        train_test_split(
            final_dataframe,
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=final_dataframe["label"],
        )
    )

    train_df, validation_df = (
        train_test_split(
            train_validation_df,
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=train_validation_df["label"],
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_dataframe.to_csv(
        ALL_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    train_df.to_csv(
        TRAIN_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    validation_df.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    test_df.to_csv(
        TEST_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDataset split completed.")
    print("-" * 60)

    for dataset_name, split_df in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Testing", test_df),
    ]:
        print(f"\n{dataset_name}:")
        print("Total:", len(split_df))
        print(split_df["label"].value_counts())

    print("\nFiles saved:")
    print(ALL_OUTPUT_PATH)
    print(TRAIN_OUTPUT_PATH)
    print(VALIDATION_OUTPUT_PATH)
    print(TEST_OUTPUT_PATH)


if __name__ == "__main__":
    main()