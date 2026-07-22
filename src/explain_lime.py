"""
Generate an HTML LIME explanation for one test article.

The HTML output is saved in:

    outputs/xai/
"""

from pathlib import Path

import joblib
import pandas as pd

from lime.lime_text import LimeTextExplainer


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
XAI_DIR = Path("outputs/xai")

XAI_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Change this to baseline_tfidf_logistic.pkl
# to explain the baseline model.
MODEL_FILE = "few_32_shot_model.pkl"


def main(
    index: int = 0,
) -> None:
    test_path = (
        DATA_DIR /
        "test_clean.csv"
    )

    model_path = (
        MODEL_DIR /
        MODEL_FILE
    )

    if not test_path.exists():
        raise FileNotFoundError(
            "test_clean.csv was not found. "
            "Run preprocess.py first."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model was not found: {model_path}"
        )

    test_df = pd.read_csv(
        test_path
    )

    test_df = test_df.dropna(
        subset=["clean_text", "label"]
    ).reset_index(drop=True)

    if index < 0 or index >= len(test_df):
        raise IndexError(
            f"Index must be between 0 and "
            f"{len(test_df) - 1}."
        )

    model = joblib.load(
        model_path
    )

    text_instance = test_df.loc[
        index,
        "clean_text",
    ]

    actual_label = test_df.loc[
        index,
        "label",
    ]

    predicted_label = model.predict(
        [text_instance]
    )[0]

    probability_values = model.predict_proba(
        [text_instance]
    )[0]

    probability_dictionary = {
        str(class_name): round(
            float(probability),
            4,
        )
        for class_name, probability in zip(
            model.classes_,
            probability_values,
        )
    }

    explainer = LimeTextExplainer(
        class_names=[
            str(class_name)
            for class_name in model.classes_
        ],
        random_state=42,
    )

    explanation = explainer.explain_instance(
        text_instance,
        model.predict_proba,
        num_features=12,
        num_samples=1000,
    )

    output_path = (
        XAI_DIR /
        f"lime_explanation_{index}.html"
    )

    explanation.save_to_file(
        str(output_path)
    )

    print("Test article index:", index)
    print("Actual label:", actual_label)
    print("Predicted label:", predicted_label)

    print("\nClass probabilities:")
    print(probability_dictionary)

    print("\nLIME features:")
    for feature, weight in explanation.as_list():
        print(
            f"{feature!r}: {weight:+.4f}"
        )

    print("\nLIME explanation saved:")
    print(output_path)


if __name__ == "__main__":
    # Change 0 to another test row number when needed
    main(index=0)