from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


MODEL_PATH = Path("models/few_32_shot_model.pkl")
VALIDATION_PATH = Path(
    "data/processed/validation_clean.csv"
)
RESULT_PATH = Path(
    "outputs/results/threshold_validation_results.csv"
)

TARGET_ACCURACY = 0.98


model = joblib.load(MODEL_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

texts = validation_df["clean_text"].astype(str)
true_labels = validation_df["label"]

probabilities = model.predict_proba(texts)
predicted_indexes = np.argmax(
    probabilities,
    axis=1,
)

predicted_labels = model.classes_[
    predicted_indexes
]

confidence_scores = np.max(
    probabilities,
    axis=1,
)

results = []

for threshold in np.arange(0.50, 0.96, 0.01):
    accepted_mask = (
        confidence_scores >= threshold
    )

    accepted_count = int(
        accepted_mask.sum()
    )

    if accepted_count == 0:
        continue

    accepted_accuracy = accuracy_score(
        true_labels[accepted_mask],
        predicted_labels[accepted_mask],
    )

    coverage = (
        accepted_count / len(validation_df)
    )

    results.append(
        {
            "threshold": round(
                float(threshold),
                2,
            ),
            "accepted_predictions":
                accepted_count,
            "uncertain_predictions":
                len(validation_df)
                - accepted_count,
            "coverage": coverage,
            "accepted_accuracy":
                accepted_accuracy,
        }
    )


results_df = pd.DataFrame(results)

RESULT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    RESULT_PATH,
    index=False,
)

print(results_df.to_string(index=False))

qualified_results = results_df[
    results_df["accepted_accuracy"]
    >= TARGET_ACCURACY
]

if qualified_results.empty:
    print(
        "\nNo threshold reached the",
        f"{TARGET_ACCURACY:.0%}",
        "target accuracy.",
    )
else:
    selected_result = (
        qualified_results
        .sort_values(
            by=[
                "coverage",
                "threshold",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    print("\nRecommended threshold:")
    print(
        f"{selected_result['threshold']:.2f}"
    )
    print(
        "Validation accepted accuracy:",
        f"{selected_result['accepted_accuracy']:.2%}",
    )
    print(
        "Validation coverage:",
        f"{selected_result['coverage']:.2%}",
    )

print("\nResults saved:", RESULT_PATH)