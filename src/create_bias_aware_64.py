"""
Create a separate 64-shot dataset that balances the confirmed
'live' lexical shortcut.

This is a targeted bias experiment. It does not claim to remove
every possible source, topic, or writing-style bias.
"""

from pathlib import Path

import pandas as pd


FULL_TRAIN_PATH = Path(
    "data/processed/train_clean.csv"
)

BASE_64_PATH = Path(
    "data/processed/few_shot/few_64_shot.csv"
)

OUTPUT_PATH = Path(
    "data/processed/few_shot/"
    "few_64_shot_bias_aware.csv"
)

RANDOM_SEED = 42
TEXT_COLUMN = "clean_text"
LABEL_COLUMN = "label"
EXPECTED_LABELS = {"fake", "real"}

# Previously inspected real-news rows in train_clean.csv.
# Each uses "live" in a broadcast or reporting context.
REAL_BROADCAST_ROW_INDICES = [
    347,   # live coverage
    487,   # broadcast live on television
    731,   # join me live
    829,   # television channels broadcast live
    832,   # debate aired live
    1796,  # news broadcast live
    2632,  # watch live / celebration live
]


def contains_live(texts: pd.Series) -> pd.Series:
    """Identify the standalone word 'live'."""

    return (
        texts
        .fillna("")
        .astype(str)
        .str.contains(
            r"\blive\b",
            case=False,
            regex=True,
        )
    )


def validate_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Check that required columns exist."""

    required_columns = {
        TEXT_COLUMN,
        LABEL_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def display_live_counts(
    dataframe: pd.DataFrame,
    title: str,
) -> dict[str, int]:
    """Display label counts for rows containing 'live'."""

    mask = contains_live(
        dataframe[TEXT_COLUMN]
    )

    counts = (
        dataframe.loc[mask, LABEL_COLUMN]
        .value_counts()
        .to_dict()
    )

    print(f"\n{title}")
    print("-" * 60)
    print("Rows containing 'live':", int(mask.sum()))
    print("Label counts:", counts)

    return {
        str(label): int(count)
        for label, count in counts.items()
    }


def main() -> None:
    if not FULL_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Full training dataset not found: "
            f"{FULL_TRAIN_PATH}"
        )

    if not BASE_64_PATH.exists():
        raise FileNotFoundError(
            f"Original 64-shot dataset not found: "
            f"{BASE_64_PATH}"
        )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "The script will not overwrite it."
        )

    full_train = pd.read_csv(
        FULL_TRAIN_PATH
    )

    base_64 = pd.read_csv(
        BASE_64_PATH
    )

    validate_columns(
        full_train,
        "Full training dataset",
    )

    validate_columns(
        base_64,
        "Original 64-shot dataset",
    )

    full_train = full_train[
        [TEXT_COLUMN, LABEL_COLUMN]
    ].copy()

    base_64 = base_64[
        [TEXT_COLUMN, LABEL_COLUMN]
    ].copy()

    base_counts = (
        base_64[LABEL_COLUMN]
        .value_counts()
        .to_dict()
    )

    expected_base_counts = {
        "fake": 64,
        "real": 64,
    }

    if base_counts != expected_base_counts:
        raise ValueError(
            "Original 64-shot class counts are "
            f"unexpected: {base_counts}"
        )

    if base_64[TEXT_COLUMN].duplicated().any():
        raise ValueError(
            "Original 64-shot dataset contains "
            "duplicate text."
        )

    base_live_counts = display_live_counts(
        base_64,
        "Original 64-shot dataset",
    )

    if base_live_counts != {"fake": 7}:
        raise ValueError(
            "The original live-word distribution has "
            "changed since it was inspected: "
            f"{base_live_counts}"
        )

    missing_indices = [
        row_index
        for row_index
        in REAL_BROADCAST_ROW_INDICES
        if row_index not in full_train.index
    ]

    if missing_indices:
        raise ValueError(
            "Required source row indices were not found: "
            f"{missing_indices}"
        )

    selected_real = full_train.loc[
        REAL_BROADCAST_ROW_INDICES,
        [TEXT_COLUMN, LABEL_COLUMN],
    ].copy()

    if not (
        selected_real[LABEL_COLUMN] == "real"
    ).all():
        raise ValueError(
            "One or more selected rows are not "
            "labelled real."
        )

    if not contains_live(
        selected_real[TEXT_COLUMN]
    ).all():
        raise ValueError(
            "One or more selected rows no longer "
            "contain the word 'live'."
        )

    selected_texts = set(
        selected_real[TEXT_COLUMN]
    )

    existing_texts = set(
        base_64[TEXT_COLUMN]
    )

    overlap = (
        selected_texts
        & existing_texts
    )

    if overlap:
        raise ValueError(
            "Selected real rows already occur in the "
            "original 64-shot dataset."
        )

    # Remove the same number of real, non-live records
    # so the final dataset remains balanced.
    removal_pool = base_64[
        (base_64[LABEL_COLUMN] == "real")
        & ~contains_live(base_64[TEXT_COLUMN])
    ]

    number_to_replace = len(
        selected_real
    )

    if len(removal_pool) < number_to_replace:
        raise ValueError(
            "Not enough eligible real rows are "
            "available for replacement."
        )

    removed_real = removal_pool.sample(
        n=number_to_replace,
        random_state=RANDOM_SEED,
        replace=False,
    )

    bias_aware_64 = base_64.drop(
        index=removed_real.index
    )

    bias_aware_64 = pd.concat(
        [
            bias_aware_64,
            selected_real,
        ],
        ignore_index=True,
    )

    bias_aware_64 = (
        bias_aware_64
        .sample(
            frac=1,
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )

    final_counts = (
        bias_aware_64[LABEL_COLUMN]
        .value_counts()
        .to_dict()
    )

    if final_counts != expected_base_counts:
        raise ValueError(
            "Final class counts are incorrect: "
            f"{final_counts}"
        )

    if len(bias_aware_64) != 128:
        raise ValueError(
            "Final dataset should contain exactly "
            f"128 rows, but contains "
            f"{len(bias_aware_64)}."
        )

    if bias_aware_64[
        TEXT_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Final dataset contains duplicate text."
        )

    if set(
        bias_aware_64[LABEL_COLUMN].unique()
    ) != EXPECTED_LABELS:
        raise ValueError(
            "Final dataset contains unexpected labels."
        )

    final_live_counts = display_live_counts(
        bias_aware_64,
        "Bias-aware 64-shot dataset",
    )

    expected_live_counts = {
        "fake": 7,
        "real": 7,
    }

    if final_live_counts != expected_live_counts:
        raise ValueError(
            "Final live-word distribution is not "
            f"balanced: {final_live_counts}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bias_aware_64.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\nBias-aware dataset created successfully.")
    print("Saved:", OUTPUT_PATH)
    print("Total rows:", len(bias_aware_64))
    print("Class counts:", final_counts)
    print(
        "Added real source rows:",
        REAL_BROADCAST_ROW_INDICES,
    )
    print(
        "Removed original row positions:",
        removed_real.index.tolist(),
    )
    print(
        "\nImportant: This dataset balances the "
        "confirmed 'live' shortcut only."
    )


if __name__ == "__main__":
    main()