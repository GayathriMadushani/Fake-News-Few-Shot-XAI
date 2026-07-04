import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DATA_DIR = Path("data/processed")
FEW_DIR = Path("data/processed/few_shot")
MODEL_DIR = Path("models")
RESULT_DIR = Path("outputs/results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

test_df = pd.read_csv(DATA_DIR / "test_clean.csv")

X_test = test_df["clean_text"]
y_test = test_df["label"]

shot_values = [4, 8, 16, 32]

results = []

for k in shot_values:
    few_df = pd.read_csv(FEW_DIR / f"few_{k}_shot.csv")

    X_train = few_df["clean_text"]
    y_train = few_df["label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "shot": k,
        "training_samples": len(few_df),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    joblib.dump(model, MODEL_DIR / f"few_{k}_shot_model.pkl")

    print(f"\n{k}-shot model completed")
    print("Accuracy:", accuracy)
    print("F1-score:", f1)

results_df = pd.DataFrame(results)
results_df.to_csv(RESULT_DIR / "few_shot_results.csv", index=False)

print("\nAll few-shot experiments completed.")
print(results_df)