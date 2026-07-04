import re
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_text_columns(df):
    possible_cols = []

    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["title", "text", "content", "article", "news", "statement"]:
            possible_cols.append(col)

    if len(possible_cols) == 0:
        object_cols = df.select_dtypes(include="object").columns.tolist()
        possible_cols = object_cols[:1]

    return possible_cols


def detect_label_column(df):
    possible_label_names = ["label", "class", "target", "type", "category"]

    for col in df.columns:
        if col.lower() in possible_label_names:
            return col

    raise ValueError("Label column not found. Please check dataset columns manually.")


def prepare_dataset(file_name):
    df = pd.read_csv(RAW_DIR / file_name)

    text_cols = detect_text_columns(df)
    label_col = detect_label_column(df)

    print(f"Using text columns: {text_cols}")
    print(f"Using label column: {label_col}")

    df["final_text"] = df[text_cols].astype(str).agg(" ".join, axis=1)
    df["clean_text"] = df["final_text"].apply(clean_text)

    df["label"] = df[label_col].astype(str).str.lower().str.strip()

    label_mapping = {
        "0": "real",
        "1": "fake",
        "real": "real",
        "fake": "fake",
        "true": "real",
        "false": "fake"
    }

    df["label"] = df["label"].map(label_mapping).fillna(df["label"])

    return df[["clean_text", "label"]]


train_df = prepare_dataset("train_set.csv")
valid_df = prepare_dataset("validation_set.csv")
test_df = prepare_dataset("test_set.csv")

train_df.to_csv(PROCESSED_DIR / "train_clean.csv", index=False)
valid_df.to_csv(PROCESSED_DIR / "validation_clean.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test_clean.csv", index=False)

print("Preprocessing completed.")
print(train_df.head())
print(train_df["label"].value_counts())