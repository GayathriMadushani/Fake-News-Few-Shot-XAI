

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min


DATA_DIR = Path("data/processed")
OUTPUT_DIR = DATA_DIR / "few_shot"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def diverse_sample(
    class_df: pd.DataFrame,
    k: int,
    seed: int = 42,
) -> pd.DataFrame:
   

    class_df = (
        class_df
        .dropna(subset=["clean_text"])
        .drop_duplicates(subset=["clean_text"])
        .reset_index(drop=True)
    )

    if len(class_df) == 0:
        return class_df.copy()

    if len(class_df) <= k:
        print(
            f"Warning: Requested {k} examples but only "
            f"{len(class_df)} are available."
        )

        return class_df.copy()

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(
        class_df["clean_text"]
    )

    number_of_clusters = min(
        k,
        len(class_df),
    )

    cluster_model = MiniBatchKMeans(
        n_clusters=number_of_clusters,
        random_state=seed,
        n_init=10,
        batch_size=256,
    )

    cluster_model.fit(tfidf_matrix)

    # Select the article closest to every cluster center
    representative_indices, _ = pairwise_distances_argmin_min(
        cluster_model.cluster_centers_,
        tfidf_matrix,
        metric="euclidean",
    )

    representative_indices = np.unique(
        representative_indices
    )

    selected_df = class_df.iloc[
        representative_indices
    ].copy()

    # In rare cases, two centers may select the same row.
    # Fill any missing positions using random remaining rows.
    if len(selected_df) < k:
        remaining_df = class_df.drop(
            index=selected_df.index
        )

        additional_count = min(
            k - len(selected_df),
            len(remaining_df),
        )

        if additional_count > 0:
            additional_df = remaining_df.sample(
                n=additional_count,
                random_state=seed,
            )

            selected_df = pd.concat(
                [selected_df, additional_df],
                ignore_index=True,
            )

    return selected_df.head(k)


def create_few_shot_data(
    df: pd.DataFrame,
    k: int,
    seed: int = 42,
) -> pd.DataFrame:
    

    selected_parts = []

    for label in ["fake", "real"]:
        class_df = df[
            df["label"] == label
        ].copy()

        selected_class_df = diverse_sample(
            class_df=class_df,
            k=k,
            seed=seed,
        )

        selected_parts.append(
            selected_class_df
        )

    few_shot_df = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # Shuffle fake and real articles together
    few_shot_df = few_shot_df.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    return few_shot_df[
        ["clean_text", "label"]
    ]


def main() -> None:
    train_path = DATA_DIR / "train_clean.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            "train_clean.csv was not found. "
            "Run preprocess.py first."
        )

    train_df = pd.read_csv(
        train_path
    )

    train_df = train_df.dropna(
        subset=["clean_text", "label"]
    )

    train_df = train_df[
        train_df["label"].isin(
            ["real", "fake"]
        )
    ]

    print("Full training dataset:")
    print(train_df["label"].value_counts())

    shot_values = [
        4,
        8,
        16,
        32,
         64,
    ]

    for k in shot_values:
        few_shot_df = create_few_shot_data(
            df=train_df,
            k=k,
            seed=42,
        )

        output_path = (
            OUTPUT_DIR /
            f"few_{k}_shot.csv"
        )

        few_shot_df.to_csv(
            output_path,
            index=False,
        )

        print("\n" + "-" * 50)
        print(f"{k}-shot dataset created")
        print("Saved:", output_path)
        print("Total samples:", len(few_shot_df))
        print(few_shot_df["label"].value_counts())

    print("\nAll few-shot datasets created successfully.")


if __name__ == "__main__":
    main()