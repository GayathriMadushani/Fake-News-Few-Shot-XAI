import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

train_df = pd.read_csv(DATA_DIR / "train_set.csv")
valid_df = pd.read_csv(DATA_DIR / "validation_set.csv")
test_df = pd.read_csv(DATA_DIR / "test_set.csv")

print("Train shape:", train_df.shape)
print("Validation shape:", valid_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain columns:")
print(train_df.columns)

print("\nFirst 5 rows:")
print(train_df.head())

print("\nColumn details:")
for col in train_df.columns:
    print("Column:", col)
    print("Unique values:", train_df[col].nunique())
    print(train_df[col].dropna().unique()[:10])
    print("-" * 50)