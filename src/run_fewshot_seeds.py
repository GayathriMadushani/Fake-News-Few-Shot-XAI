"""
Step 3: Multi-seed few-shot experiment (TF-IDF + Logistic Regression).

Run from the project root. Needs these files in src/:
    audit_shortcuts.py, audit_shortcut_groups.py, create_few_shot.py

    python src/run_fewshot_seeds.py

For every shot size it repeats the experiment with many random seeds and
compares:
  - sampling strategy: random / random_len30 / diverse (your KMeans method)
  - text version: original / masked (all shortcut groups removed)
and reports mean +- std of accuracy and macro F1 on the test set.

"shot" means examples PER CLASS (shot=8 -> 16 training articles).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from audit_shortcut_groups import GROUPS, apply_mask, compile_group
from audit_shortcuts import evaluate, load
from create_few_shot import create_few_shot_data

OUT_DIR = Path("outputs/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHOTS = [4, 8, 16, 32, 64]
SEEDS = list(range(10))
STRATEGIES = ["random", "random_len30", "diverse"]
MIN_WORDS = 30


def few_shot_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        LogisticRegression(C=2.0, max_iter=1000),
    )


def draw_sample(
    train_df: pd.DataFrame,
    shot: int,
    seed: int,
    strategy: str,
) -> pd.DataFrame:
    if strategy == "diverse":
        return create_few_shot_data(train_df, k=shot, seed=seed)

    pool = train_df
    if strategy == "random_len30":
        pool = train_df[train_df["n_words"] >= MIN_WORDS]

    parts = [
        pool[pool["label"] == label].sample(n=shot, random_state=seed)
        for label in ["fake", "real"]
    ]
    return pd.concat(parts, ignore_index=True)[["clean_text", "label"]]


def main() -> None:
    train_df = load("train_clean.csv")
    test_df = load("test_clean.csv")
    train_df["n_words"] = train_df["clean_text"].str.split().str.len()

    all_words = [w for words, _ in GROUPS.values() for w in words]
    all_patterns = [p for _, patterns in GROUPS.values() for p in patterns]
    pattern = compile_group(all_words, all_patterns)

    y_test = test_df["label"]
    test_texts = {
        "original": test_df["clean_text"],
        "masked": apply_mask(test_df["clean_text"], pattern),
    }

    rows = []
    for shot in SHOTS:
        print(f"Running {shot}-shot ({len(SEEDS)} seeds)...")
        for seed in SEEDS:
            for strategy in STRATEGIES:
                sample = draw_sample(train_df, shot, seed, strategy)
                texts = {
                    "original": sample["clean_text"],
                    "masked": apply_mask(sample["clean_text"], pattern),
                }
                for version in ["original", "masked"]:
                    model = few_shot_model()
                    model.fit(texts[version], sample["label"])
                    acc, f1 = evaluate(model, test_texts[version], y_test)
                    rows.append(
                        {
                            "shot": shot,
                            "seed": seed,
                            "strategy": strategy,
                            "text_version": version,
                            "accuracy": acc,
                            "macro_f1": f1,
                        }
                    )

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_DIR / "fewshot_seeds_raw.csv", index=False)

    summary = (
        raw.groupby(["strategy", "text_version", "shot"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["accuracy"] = summary.apply(
        lambda r: f"{r['mean']:.3f} +- {r['std']:.3f}", axis=1
    )
    table = summary.pivot(
        index=["strategy", "text_version"],
        columns="shot",
        values="accuracy",
    )
    table.to_csv(OUT_DIR / "fewshot_seeds_summary.csv")

    print("\nTest accuracy, mean +- std over seeds")
    print("=" * 70)
    print(table.to_string())
    print(f"\nSaved: {OUT_DIR / 'fewshot_seeds_raw.csv'}")
    print(f"Saved: {OUT_DIR / 'fewshot_seeds_summary.csv'}")


if __name__ == "__main__":
    main()
