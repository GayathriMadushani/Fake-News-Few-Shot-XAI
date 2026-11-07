from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# INPUT PATHS
# ---------------------------------------------------------

ENGLISH_TRAIN_PATH = Path(
    "data/processed/few_shot/"
    "few_64_shot_bias_aware.csv"
)

ENGLISH_VALIDATION_PATH = Path(
    "data/processed/validation_clean.csv"
)

ENGLISH_TEST_PATH = Path(
    "data/processed/test_clean.csv"
)

SINHALA_TRAIN_PATH = Path(
    "data/processed/sinhala/few_shot/"
    "sinhala_64_shot.csv"
)

SINHALA_VALIDATION_PATH = Path(
    "data/processed/sinhala/"
    "sinhala_validation_clean.csv"
)

SINHALA_TEST_PATH = Path(
    "data/processed/sinhala/"
    "sinhala_test_clean.csv"
)


# ---------------------------------------------------------
# OUTPUT PATHS
# ---------------------------------------------------------

OUTPUT_DIR = Path(
    "data/processed/multilingual"
)

TRAIN_OUTPUT_PATH = (
    OUTPUT_DIR
    / "bilingual_64_shot_train.csv"
)

VALIDATION_OUTPUT_PATH = (
    OUTPUT_DIR
    / "bilingual_validation.csv"
)

TEST_OUTPUT_PATH = (
    OUTPUT_DIR
    / "bilingual_test.csv"
)

RANDOM_SEED = 42


# ---------------------------------------------------------
# DATA STANDARDIZATION
# ---------------------------------------------------------

def load_and_standardize(
    file_path: Path,
    language: str,
    dataset_source: str,
) -> pd.DataFrame:
    """Load one dataset and create common columns."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    required_columns = {
        "clean_text",
        "label",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{file_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe["clean_text"] = (
        dataframe["clean_text"]
        .fillna("")
        .astype(str)
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    dataframe["label"] = (
        dataframe["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    dataframe = dataframe[
        dataframe["clean_text"] != ""
    ].copy()

    dataframe = dataframe[
        dataframe["label"].isin(
            ["fake", "real"]
        )
    ].copy()

    dataframe["language"] = language
    dataframe["dataset_source"] = (
        dataset_source
    )

    if "original_type" not in dataframe.columns:
        dataframe["original_type"] = (
            "ENGLISH_"
            + dataframe["label"].str.upper()
        )

    if "domain" not in dataframe.columns:
        dataframe["domain"] = ""

    if "datestamp" not in dataframe.columns:
        dataframe["datestamp"] = ""

    return dataframe[
        [
            "clean_text",
            "label",
            "language",
            "dataset_source",
            "original_type",
            "domain",
            "datestamp",
        ]
    ].reset_index(drop=True)


def combine_datasets(
    english_path: Path,
    sinhala_path: Path,
) -> pd.DataFrame:
    """Combine English and Sinhala datasets."""

    english_dataframe = load_and_standardize(
        file_path=english_path,
        language="en",
        dataset_source="english_existing",
    )

    sinhala_dataframe = load_and_standardize(
        file_path=sinhala_path,
        language="si",
        dataset_source="lirneasia_sinhala",
    )

    combined_dataframe = pd.concat(
        [
            english_dataframe,
            sinhala_dataframe,
        ],
        ignore_index=True,
    )

    combined_dataframe = (
        combined_dataframe
        .sample(
            frac=1,
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )

    return combined_dataframe


def display_counts(
    dataset_name: str,
    dataframe: pd.DataFrame,
) -> None:
    """Display language and label counts."""

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    print("Total:", len(dataframe))

    print("\nLanguage counts:")
    print(
        dataframe["language"]
        .value_counts()
    )

    print("\nLabel counts:")
    print(
        dataframe["label"]
        .value_counts()
    )

    print("\nLanguage and label counts:")
    print(
        pd.crosstab(
            dataframe["language"],
            dataframe["label"],
        )
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:
    bilingual_train = combine_datasets(
        english_path=ENGLISH_TRAIN_PATH,
        sinhala_path=SINHALA_TRAIN_PATH,
    )

    bilingual_validation = combine_datasets(
        english_path=ENGLISH_VALIDATION_PATH,
        sinhala_path=SINHALA_VALIDATION_PATH,
    )

    bilingual_test = combine_datasets(
        english_path=ENGLISH_TEST_PATH,
        sinhala_path=SINHALA_TEST_PATH,
    )

    # Check for exact text leakage between splits.
    train_texts = set(
        bilingual_train["clean_text"]
    )

    validation_texts = set(
        bilingual_validation["clean_text"]
    )

    test_texts = set(
        bilingual_test["clean_text"]
    )

    train_validation_overlap = (
        train_texts & validation_texts
    )

    train_test_overlap = (
        train_texts & test_texts
    )

    validation_test_overlap = (
        validation_texts & test_texts
    )

    if train_validation_overlap:
        raise ValueError(
            "Duplicate text exists in training "
            "and validation datasets."
        )

    if train_test_overlap:
        raise ValueError(
            "Duplicate text exists in training "
            "and test datasets."
        )

    if validation_test_overlap:
        raise ValueError(
            "Duplicate text exists in validation "
            "and test datasets."
        )

    expected_training_counts = {
        ("en", "fake"): 64,
        ("en", "real"): 64,
        ("si", "fake"): 64,
        ("si", "real"): 64,
    }

    actual_training_counts = {
        (language, label): int(count)
        for (
            language,
            label,
        ), count in (
            bilingual_train
            .groupby(
                ["language", "label"]
            )
            .size()
            .items()
        )
    }

    if (
        actual_training_counts
        != expected_training_counts
    ):
        raise ValueError(
            "Unexpected bilingual training counts: "
            f"{actual_training_counts}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bilingual_train.to_csv(
        TRAIN_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    bilingual_validation.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    bilingual_test.to_csv(
        TEST_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    display_counts(
        "BILINGUAL TRAINING DATASET",
        bilingual_train,
    )

    display_counts(
        "BILINGUAL VALIDATION DATASET",
        bilingual_validation,
    )

    display_counts(
        "BILINGUAL TEST DATASET",
        bilingual_test,
    )

    print("\nFiles saved successfully:")
    print(TRAIN_OUTPUT_PATH)
    print(VALIDATION_OUTPUT_PATH)
    print(TEST_OUTPUT_PATH)


if __name__ == "__main__":
    main()