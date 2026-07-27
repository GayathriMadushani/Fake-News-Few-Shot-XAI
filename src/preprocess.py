

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


TEXT_COLUMN_NAMES = {
    "title",
    "text",
    "content",
    "article",
    "news",
    "statement",
    "body",
}

LABEL_COLUMN_NAMES = {
    "label",
    "class",
    "target",
    "type",
    "category",
}


# IMPORTANT:
# Confirm that your original dataset really uses:
#     0 = real
#     1 = fake
#
# Some datasets use the opposite format.
LABEL_MAPPING = {
    "0": "real",
    "1": "fake",
    "real": "real",
    "fake": "fake",
    "true": "real",
    "false": "fake",
    "reliable": "real",
    "unreliable": "fake",
}


def clean_text(text: object) -> str:
    """
    Perform light text cleaning.

    This preserves useful punctuation and terms such as:
        COVID-19
        U.S.
        25%
        $500
    """

    text = str(text)

    # Replace non-breaking spaces
    text = text.replace("\u00a0", " ")

    # Lowercase
    text = text.lower()

    # Replace links with a token
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text,
    )

    # Replace email addresses with a token
    text = re.sub(
        r"\S+@\S+",
        " EMAIL ",
        text,
    )

    # Keep letters, numbers and useful news punctuation
    text = re.sub(
        r"[^a-z0-9.,!?%$'\-\s]",
        " ",
        text,
    )

    # Remove repeated spaces
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def detect_text_columns(df: pd.DataFrame) -> list[str]:
    """
    Automatically detect possible news text columns.

    For example:
        title
        text
        article
        content
    """

    detected_columns = []

    for column in df.columns:
        normalized_name = column.strip().lower()

        if normalized_name in TEXT_COLUMN_NAMES:
            detected_columns.append(column)

    if detected_columns:
        return detected_columns

    # Fallback: choose the first object/string column
    object_columns = df.select_dtypes(
        include="object"
    ).columns.tolist()

    if not object_columns:
        raise ValueError(
            "No text column was found in the dataset."
        )

    return object_columns[:1]


def detect_label_column(df: pd.DataFrame) -> str:
    """
    Automatically detect the label column.
    """

    for column in df.columns:
        normalized_name = column.strip().lower()

        if normalized_name in LABEL_COLUMN_NAMES:
            return column

    raise ValueError(
        "Label column was not found.\n"
        f"Available columns: {list(df.columns)}"
    )


def prepare_dataset(file_name: str) -> pd.DataFrame:
    """
    Load and preprocess one raw dataset.
    """

    file_path = RAW_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Remove unwanted spaces from column names
    df.columns = df.columns.str.strip()

    text_columns = detect_text_columns(df)
    label_column = detect_label_column(df)

    print("\nProcessing:", file_name)
    print("Text columns:", text_columns)
    print("Label column:", label_column)

    # Combine all available text columns
    combined_text = (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    # Clean and map labels
    original_labels = (
        df[label_column]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    mapped_labels = original_labels.map(LABEL_MAPPING)

    processed_df = pd.DataFrame(
        {
            "clean_text": combined_text.map(clean_text),
            "label": mapped_labels,
        }
    )

    # Remove unknown labels
    unknown_labels = original_labels[
        mapped_labels.isna()
    ].unique()

    if len(unknown_labels) > 0:
        print(
            "Warning: Unrecognized labels were removed:",
            unknown_labels,
        )

    processed_df = processed_df.dropna(
        subset=["label"]
    )

    # Remove very short or empty articles
    processed_df = processed_df[
        processed_df["clean_text"].str.len() >= 20
    ]

    # Keep only expected classes
    processed_df = processed_df[
        processed_df["label"].isin(["real", "fake"])
    ]

    # Remove duplicated articles
    processed_df = processed_df.drop_duplicates(
        subset=["clean_text"]
    )

    processed_df = processed_df.reset_index(
        drop=True
    )

    print("Processed shape:", processed_df.shape)
    print("Label distribution:")
    print(processed_df["label"].value_counts())

    return processed_df


def main() -> None:
    

    dataset_files = {
        "train_set.csv": "train_clean.csv",
        "validation_set.csv": "validation_clean.csv",
        "test_set.csv": "test_clean.csv",
    }

    for raw_file, processed_file in dataset_files.items():
        processed_df = prepare_dataset(raw_file)

        output_path = PROCESSED_DIR / processed_file

        processed_df.to_csv(
            output_path,
            index=False,
        )

        print("Saved:", output_path)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()