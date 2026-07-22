"""
Check processed datasets for:

- Class balance
- Missing values
- Duplicate articles
- Very short articles
- Train/validation/test data leakage
"""

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed")


def load_and_check(file_name: str) -> pd.DataFrame:
    """
    Load and display information about one processed dataset.
    """

    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            "Run preprocess.py first."
        )

    df = pd.read_csv(file_path)

    print("\n" + "=" * 60)
    print(file_name)
    print("=" * 60)

    print("Dataset shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False))

    print("\nMissing values:")
    print(df.isna().sum())

    duplicate_count = df[
        "clean_text"
    ].duplicated().sum()

    print("\nDuplicate text rows:", duplicate_count)

    word_lengths = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    print("Minimum words:", word_lengths.min())
    print("Median words:", word_lengths.median())
    print("Average words:", round(word_lengths.mean(), 2))
    print("Maximum words:", word_lengths.max())

    print(
        "Articles with fewer than 8 words:",
        int((word_lengths < 8).sum()),
    )

    return df


def main() -> None:
    train_df = load_and_check(
        "train_clean.csv"
    )

    validation_df = load_and_check(
        "validation_clean.csv"
    )

    test_df = load_and_check(
        "test_clean.csv"
    )

    # Use sets to detect exact duplicate articles across splits
    train_text = set(
        train_df["clean_text"].dropna()
    )

    validation_text = set(
        validation_df["clean_text"].dropna()
    )

    test_text = set(
        test_df["clean_text"].dropna()
    )

    print("\n" + "=" * 60)
    print("DATA LEAKAGE CHECK")
    print("=" * 60)

    print(
        "Exact overlap between train and validation:",
        len(train_text & validation_text),
    )

    print(
        "Exact overlap between train and test:",
        len(train_text & test_text),
    )

    print(
        "Exact overlap between validation and test:",
        len(validation_text & test_text),
    )

    print("\nDataset checking completed.")


if __name__ == "__main__":
    main()