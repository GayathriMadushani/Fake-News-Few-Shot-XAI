"""
Step 1: Shortcut audit.

Run from the project root:
    python src/audit_shortcuts.py

It answers four questions:
  1. Which words push the baseline toward "fake" or "real"?
  2. Which frequent words almost only appear in one class?
  3. How much does accuracy drop when those words are masked?
  4. How much accuracy does text length alone give?

Edit MASK_TERMS after you read the output of parts 1 and 2,
then run the script again.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

DATA_DIR = Path("data/processed")
OUT_DIR = Path("outputs/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Starting list of suspected shortcuts (taken from your few-shot CSV).
# Replace or extend it after you inspect the output.
MASK_TERMS = [
    "reuters",
    "video",
    "via",
    "breitbart",
    "sputnik",
    "featured",
    "image",
    "getty",
]
# Words that carry a random suffix, e.g. pictwittercomukwszjezo3
MASK_PREFIXES = ["pictwittercom", "httpsfedup", "https", "http", "www"]

TOP_N = 25
MIN_DOC_FREQ = 20
SKEW_THRESHOLD = 0.90


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name)
    df = df.dropna(subset=["clean_text", "label"]).copy()
    df["clean_text"] = df["clean_text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    return df[df["label"].isin(["fake", "real"])].reset_index(drop=True)


def build_mask_regex() -> re.Pattern:
    words = "|".join(re.escape(w) for w in MASK_TERMS)
    prefixes = "|".join(re.escape(p) for p in MASK_PREFIXES)
    return re.compile(rf"\b(?:{words})\b|\b(?:{prefixes})\w*")


def mask_text(texts: pd.Series, pattern: re.Pattern) -> pd.Series:
    masked = texts.map(lambda t: pattern.sub(" ", t))
    return masked.map(lambda t: re.sub(r"\s+", " ", t).strip())


def new_model():
    return make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=50000,
        ),
        LogisticRegression(C=2.0, max_iter=1000),
    )


def evaluate(model, texts, labels) -> tuple[float, float]:
    pred = model.predict(texts)
    return (
        accuracy_score(labels, pred),
        f1_score(labels, pred, average="macro"),
    )


def top_coefficients(model) -> None:
    vectorizer = model.steps[0][1]
    classifier = model.steps[1][1]
    names = np.array(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]

    # classes_ is ['fake', 'real'], so positive weight = real
    order = np.argsort(coefficients)

    print("\n" + "=" * 70)
    print("1. TOP WORDS PUSHING TOWARD 'FAKE'")
    print("=" * 70)
    for i in order[:TOP_N]:
        print(f"{names[i]:35s} {coefficients[i]:+.3f}")

    print("\n" + "=" * 70)
    print("   TOP WORDS PUSHING TOWARD 'REAL'")
    print("=" * 70)
    for i in order[::-1][:TOP_N]:
        print(f"{names[i]:35s} {coefficients[i]:+.3f}")


def class_skew(train_df: pd.DataFrame) -> None:
    counter = CountVectorizer(
        binary=True,
        min_df=MIN_DOC_FREQ,
        token_pattern=r"(?u)\b[a-z][a-z0-9]+\b",
    )
    matrix = counter.fit_transform(train_df["clean_text"])
    names = np.array(counter.get_feature_names_out())

    is_fake = (train_df["label"] == "fake").to_numpy()
    fake_docs = np.asarray(matrix[is_fake].sum(axis=0)).ravel()
    real_docs = np.asarray(matrix[~is_fake].sum(axis=0)).ravel()
    total_docs = fake_docs + real_docs
    fake_share = fake_docs / total_docs

    skew = pd.DataFrame(
        {
            "word": names,
            "docs_total": total_docs,
            "docs_fake": fake_docs,
            "docs_real": real_docs,
            "fake_share": fake_share,
        }
    )

    skewed = skew[
        (skew["fake_share"] >= SKEW_THRESHOLD)
        | (skew["fake_share"] <= 1 - SKEW_THRESHOLD)
    ].sort_values("docs_total", ascending=False)

    skewed.to_csv(OUT_DIR / "token_skew.csv", index=False)

    print("\n" + "=" * 70)
    print(
        f"2. FREQUENT WORDS (>= {MIN_DOC_FREQ} docs) THAT ARE "
        f">= {SKEW_THRESHOLD:.0%} ONE CLASS"
    )
    print("=" * 70)
    print(skewed.head(40).to_string(index=False))
    print(f"\nFull list saved: {OUT_DIR / 'token_skew.csv'}")


def length_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    def features(df: pd.DataFrame) -> np.ndarray:
        words = df["clean_text"].str.split().str.len().to_numpy()
        chars = df["clean_text"].str.len().to_numpy()
        return np.column_stack([np.log1p(words), np.log1p(chars)])

    model = LogisticRegression(max_iter=1000)
    model.fit(features(train_df), train_df["label"])
    pred = model.predict(features(test_df))
    return {
        "setting": "D. Length only (word and char count)",
        "accuracy": accuracy_score(test_df["label"], pred),
        "macro_f1": f1_score(test_df["label"], pred, average="macro"),
    }


def main() -> None:
    train_df = load("train_clean.csv")
    test_df = load("test_clean.csv")
    print("Train:", len(train_df), "| Test:", len(test_df))

    pattern = build_mask_regex()

    # A. Original data
    model_a = new_model()
    model_a.fit(train_df["clean_text"], train_df["label"])
    acc_a, f1_a = evaluate(model_a, test_df["clean_text"], test_df["label"])

    top_coefficients(model_a)
    class_skew(train_df)

    # B. Same model, masked test text only
    masked_test = mask_text(test_df["clean_text"], pattern)
    acc_b, f1_b = evaluate(model_a, masked_test, test_df["label"])

    # C. Retrain on masked train, test on masked test
    masked_train = mask_text(train_df["clean_text"], pattern)
    model_c = new_model()
    model_c.fit(masked_train, train_df["label"])
    acc_c, f1_c = evaluate(model_c, masked_test, test_df["label"])

    rows = [
        {"setting": "A. Original train / original test",
         "accuracy": acc_a, "macro_f1": f1_a},
        {"setting": "B. Original model / masked test",
         "accuracy": acc_b, "macro_f1": f1_b},
        {"setting": "C. Retrained on masked / masked test",
         "accuracy": acc_c, "macro_f1": f1_c},
        length_baseline(train_df, test_df),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "shortcut_audit_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("3 and 4. MASKING AND LENGTH-ONLY RESULTS")
    print("=" * 70)
    print(f"Masked terms: {MASK_TERMS}")
    print(f"Masked prefixes: {MASK_PREFIXES}\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {OUT_DIR / 'shortcut_audit_summary.csv'}")


if __name__ == "__main__":
    main()
