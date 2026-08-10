from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


MODEL_PATH = Path("models/few_32_shot_model.pkl")
TEST_PATH = Path("data/processed/test_clean.csv")
THRESHOLD = 0.58


model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_PATH)

texts = test_df["clean_text"].astype(str)
true_labels = test_df["label"]

probabilities = model.predict_proba(texts)
predicted_indexes = np.argmax(probabilities, axis=1)

predicted_labels = model.classes_[predicted_indexes]
confidence_scores = np.max(probabilities, axis=1)

accepted_mask = confidence_scores >= THRESHOLD

accepted_count = int(accepted_mask.sum())
uncertain_count = int((~accepted_mask).sum())
total_count = len(test_df)

overall_accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)

if accepted_count > 0:
    accepted_accuracy = accuracy_score(
        true_labels[accepted_mask],
        predicted_labels[accepted_mask],
    )
else:
    accepted_accuracy = 0.0

coverage = accepted_count / total_count

print("Model:", MODEL_PATH)
print("Threshold:", THRESHOLD)
print("Total test samples:", total_count)
print("Overall accuracy:", f"{overall_accuracy:.4f}")
print("Accepted predictions:", accepted_count)
print("Uncertain predictions:", uncertain_count)
print("Coverage:", f"{coverage:.2%}")
print(
    "Accuracy among accepted predictions:",
    f"{accepted_accuracy:.2%}",
)