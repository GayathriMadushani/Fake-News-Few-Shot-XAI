import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULT_DIR = Path("outputs/results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned datasets
train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
test_df = pd.read_csv(DATA_DIR / "test_clean.csv")

# Fix missing values
train_df = train_df.dropna(subset=["label"])
test_df = test_df.dropna(subset=["label"])

train_df["clean_text"] = train_df["clean_text"].fillna("").astype(str)
test_df["clean_text"] = test_df["clean_text"].fillna("").astype(str)

# Remove empty text rows
train_df = train_df[train_df["clean_text"].str.strip() != ""]
test_df = test_df[test_df["clean_text"].str.strip() != ""]

print("Train data after cleaning:", train_df.shape)
print("Test data after cleaning:", test_df.shape)

X_train = train_df["clean_text"]
y_train = train_df["label"]

X_test = test_df["clean_text"]
y_test = test_df["label"]

# Baseline model: TF-IDF + Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Train model
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
matrix = confusion_matrix(y_test, predictions)

print("\nBaseline Model Accuracy:", accuracy)
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(matrix)

# Save model
joblib.dump(model, MODEL_DIR / "baseline_tfidf_logistic.pkl")

# Save results
with open(RESULT_DIR / "baseline_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Baseline Model Accuracy: {accuracy}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(matrix))

print("\nBaseline model saved successfully.")
print("Model saved to: models/baseline_tfidf_logistic.pkl")
print("Results saved to: outputs/results/baseline_results.txt")