import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed/few_shot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train_clean.csv")

print("Original columns:")
print(train_df.columns)

# Make sure correct column names exist
train_df.columns = train_df.columns.str.strip()

# Remove missing values
train_df = train_df.dropna(subset=["clean_text", "label"])
train_df["clean_text"] = train_df["clean_text"].astype(str)
train_df["label"] = train_df["label"].astype(str)

print("\nLabel distribution before few-shot:")
print(train_df["label"].value_counts())


def create_few_shot_data(df, k, seed=42):
    few_parts = []

    for label_name in df["label"].unique():
        class_df = df[df["label"] == label_name]

        sampled_df = class_df.sample(
            n=min(k, len(class_df)),
            random_state=seed
        )

        few_parts.append(sampled_df)

    few_df = pd.concat(few_parts, ignore_index=True)

    # Shuffle final few-shot dataset
    few_df = few_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return few_df[["clean_text", "label"]]


shot_values = [4, 8, 16, 32]

for k in shot_values:
    few_df = create_few_shot_data(train_df, k)

    few_df.to_csv(OUTPUT_DIR / f"few_{k}_shot.csv", index=False)

    print(f"\n{k}-shot dataset created")
    print("Total samples:", len(few_df))
    print(few_df["label"].value_counts())

print("\nAll few-shot datasets created successfully.")