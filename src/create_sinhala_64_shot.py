from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

INPUT_PATH = Path(
    "data/processed/sinhala/"
    "sinhala_train_clean.csv"
)

OUTPUT_DIR = Path(
    "data/processed/sinhala/few_shot"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "sinhala_64_shot.csv"
)

SHOT_SIZE = 64
RANDOM_SEED = 42


# ---------------------------------------------------------
# DIVERSE SAMPLING
# ---------------------------------------------------------

def diverse_sample(
    dataframe: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select diverse Sinhala articles using character-level
    TF-IDF and MiniBatchKMeans.
    """

    dataframe = (
        dataframe
        .dropna(subset=["clean_text"])
        .drop_duplicates(subset=["clean_text"])
        .reset_index(drop=True)
    )

    if len(dataframe) < sample_size:
        raise ValueError(
            f"Requested {sample_size} samples, "
            f"but only {len(dataframe)} are available."
        )

    if len(dataframe) == sample_size:
        return dataframe.copy()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=15000,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(
        dataframe["clean_text"]
    )

    cluster_model = MiniBatchKMeans(
        n_clusters=sample_size,
        random_state=seed,
        n_init=10,
        batch_size=256,
    )

    cluster_model.fit(tfidf_matrix)

    selected_indices, _ = (
        pairwise_distances_argmin_min(
            cluster_model.cluster_centers_,
            tfidf_matrix,
            metric="euclidean",
        )
    )

    selected_indices = np.unique(
        selected_indices
    )

    selected_dataframe = dataframe.iloc[
        selected_indices
    ].copy()

    # Two cluster centres can sometimes select the same article.
    if len(selected_dataframe) < sample_size:
        remaining_dataframe = dataframe.drop(
            index=selected_dataframe.index
        )

        additional_count = (
            sample_size - len(selected_dataframe)
        )

        additional_dataframe = (
            remaining_dataframe.sample(
                n=additional_count,
                random_state=seed,
                replace=False,
            )
        )

        selected_dataframe = pd.concat(
            [
                selected_dataframe,
                additional_dataframe,
            ],
            ignore_index=True,
        )

    return (
        selected_dataframe
        .head(sample_size)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset was not found: {INPUT_PATH}"
        )

    training_dataframe = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        "clean_text",
        "label",
        "language",
        "original_type",
    }

    missing_columns = (
        required_columns
        - set(training_dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    print("Sinhala training dataset:")
    print(
        training_dataframe["label"]
        .value_counts()
    )

    print("\nOriginal types:")
    print(
        training_dataframe["original_type"]
        .value_counts()
    )

    # -----------------------------------------------------
    # SELECT 64 REAL ARTICLES
    # -----------------------------------------------------

    real_pool = training_dataframe[
        training_dataframe["label"] == "real"
    ].copy()

    selected_real = diverse_sample(
        dataframe=real_pool,
        sample_size=SHOT_SIZE,
        seed=RANDOM_SEED,
    )

    # -----------------------------------------------------
    # SELECT 64 FAKE/MISLEADING ARTICLES
    # -----------------------------------------------------

    fake_pool = training_dataframe[
        training_dataframe["label"] == "fake"
    ].copy()

    false_pool = fake_pool[
        fake_pool["original_type"] == "FALSE"
    ].copy()

    partial_pool = fake_pool[
        fake_pool["original_type"] == "PARTIAL"
    ].copy()

    # Preserve the FALSE/PARTIAL proportion.
    false_target = round(
        SHOT_SIZE
        * len(false_pool)
        / len(fake_pool)
    )

    false_target = max(
        1,
        min(false_target, len(false_pool)),
    )

    partial_target = (
        SHOT_SIZE - false_target
    )

    if len(partial_pool) < partial_target:
        raise ValueError(
            "Not enough PARTIAL articles are available."
        )

    print("\nFake/misleading selection targets:")
    print("FALSE:", false_target)
    print("PARTIAL:", partial_target)

    selected_false = diverse_sample(
        dataframe=false_pool,
        sample_size=false_target,
        seed=RANDOM_SEED,
    )

    selected_partial = diverse_sample(
        dataframe=partial_pool,
        sample_size=partial_target,
        seed=RANDOM_SEED,
    )

    selected_fake = pd.concat(
        [
            selected_false,
            selected_partial,
        ],
        ignore_index=True,
    )

    # -----------------------------------------------------
    # COMBINE AND SHUFFLE
    # -----------------------------------------------------

    few_shot_dataframe = pd.concat(
        [
            selected_real,
            selected_fake,
        ],
        ignore_index=True,
    )

    few_shot_dataframe = (
        few_shot_dataframe
        .sample(
            frac=1,
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )

    if len(few_shot_dataframe) != 128:
        raise ValueError(
            "The final dataset must contain 128 articles."
        )

    if few_shot_dataframe[
        "clean_text"
    ].duplicated().any():
        raise ValueError(
            "The final dataset contains duplicate text."
        )

    final_counts = (
        few_shot_dataframe["label"]
        .value_counts()
        .to_dict()
    )

    expected_counts = {
        "real": 64,
        "fake": 64,
    }

    if final_counts != expected_counts:
        raise ValueError(
            f"Incorrect final counts: {final_counts}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    few_shot_dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSinhala 64-shot dataset created.")
    print("Saved:", OUTPUT_PATH)
    print("Total:", len(few_shot_dataframe))

    print("\nFinal label counts:")
    print(
        few_shot_dataframe["label"]
        .value_counts()
    )

    print("\nFinal original-type counts:")
    print(
        few_shot_dataframe["original_type"]
        .value_counts()
    )


if __name__ == "__main__":
    main()