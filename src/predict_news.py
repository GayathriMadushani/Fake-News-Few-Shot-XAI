"""
Interactive fake-news prediction system.

Features:
- Displays real and fake probabilities
- Returns "uncertain" for low-confidence predictions
- Uses LIME to explain important features
- Displays features supporting and opposing the prediction
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

# Few-shot model
MODEL_PATH = Path("models/few_32_shot_model.pkl")

# For the stronger full-data baseline model, comment the line above
# and uncomment the line below:
# MODEL_PATH = Path("models/baseline_tfidf_logistic.pkl")

UNCERTAIN_THRESHOLD = 0.55
MINIMUM_WORDS = 8


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text: object) -> str:
    """
    Apply the same basic cleaning used during preprocessing.
    """

    text = str(text)
    text = text.replace("\u00a0", " ")
    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text,
    )

    text = re.sub(
        r"\S+@\S+",
        " EMAIL ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9.,!?%$'\-\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model was not found: {MODEL_PATH}\n"
        "Run train_few_shot.py or train_baseline.py first."
    )

model = joblib.load(MODEL_PATH)

class_names = [
    str(class_name)
    for class_name in model.classes_
]

explainer = LimeTextExplainer(
    class_names=class_names,
    random_state=42,
)


# ---------------------------------------------------------
# FEATURE CLEANING
# ---------------------------------------------------------

def clean_lime_feature(feature: object) -> str:
    """
    Convert LIME or NumPy string values into clean Python strings.
    """

    feature_text = str(feature)

    if feature_text.startswith("np.str_("):
        feature_text = feature_text.replace(
            "np.str_(",
            "",
            1,
        )

        if (
            feature_text.startswith("'")
            and feature_text.endswith("')")
        ):
            feature_text = feature_text[1:-2]

        elif (
            feature_text.startswith('"')
            and feature_text.endswith('")')
        ):
            feature_text = feature_text[1:-2]

    return feature_text


# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------

def predict_with_reason(news_text: str) -> dict:
    """
    Predict whether the input resembles real or fake news.

    Important:
    This model detects learned language patterns.
    It does not independently verify facts online.
    """

    cleaned_text = clean_text(news_text)
    word_count = len(cleaned_text.split())

    if word_count < MINIMUM_WORDS:
        return {
            "prediction": "invalid input",
            "raw_prediction": None,
            "confidence": 0.0,
            "probabilities": {},
            "important_features": [],
            "supporting_features": [],
            "opposing_features": [],
            "reason": (
                f"Enter at least {MINIMUM_WORDS} words. "
                "A complete news paragraph produces a better result."
            ),
            "warning": (
                "The classifier recognizes writing patterns "
                "and does not verify facts against external sources."
            ),
        }

    # Get class probabilities
    probabilities = model.predict_proba(
        [cleaned_text]
    )[0]

    winning_class_index = int(
        np.argmax(probabilities)
    )

    raw_prediction = class_names[
        winning_class_index
    ]

    confidence = float(
        probabilities[
            winning_class_index
        ]
    )

    # Return uncertain when confidence is below the threshold
    if confidence >= UNCERTAIN_THRESHOLD:
        final_prediction = raw_prediction
    else:
        final_prediction = "uncertain"

    probability_dictionary = {
        str(class_name): round(
            float(probability),
            4,
        )
        for class_name, probability in zip(
            class_names,
            probabilities,
        )
    }

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        cleaned_text,
        model.predict_proba,
        labels=[winning_class_index],
        num_features=10,
        num_samples=500,
    )

    explanation_features = explanation.as_list(
        label=winning_class_index
    )

    important_features = []

    for feature, weight in explanation_features:
        important_features.append(
            {
                "feature": clean_lime_feature(feature),
                "weight": round(
                    float(weight),
                    4,
                ),
            }
        )

    supporting_features = [
        item
        for item in important_features
        if item["weight"] > 0
    ]

    opposing_features = [
        item
        for item in important_features
        if item["weight"] < 0
    ]

    supporting_words = [
        item["feature"]
        for item in supporting_features[:5]
    ]

    opposing_words = [
        item["feature"]
        for item in opposing_features[:5]
    ]

    if final_prediction == "uncertain":
        reason = (
            "The real and fake probabilities are too close. "
            "The system therefore returns 'uncertain' instead "
            "of forcing a possibly incorrect decision."
        )
    else:
        reason = (
            f"The model predicted '{raw_prediction}' because "
            f"features such as "
            f"{supporting_words or ['none clearly identified']} "
            "supported that class."
        )

        if opposing_words:
            reason += (
                f" Features such as {opposing_words} "
                "opposed the prediction."
            )

    return {
        "prediction": final_prediction,
        "raw_prediction": raw_prediction,
        "confidence": round(
            confidence,
            4,
        ),
        "probabilities": probability_dictionary,
        "important_features": important_features,
        "supporting_features": supporting_features,
        "opposing_features": opposing_features,
        "reason": reason,
        "warning": (
            "This classifier recognizes language patterns. "
            "It does not verify whether the event actually happened."
        ),
    }


# ---------------------------------------------------------
# DISPLAY FUNCTION
# ---------------------------------------------------------

def display_result(result: dict) -> None:
    """
    Display the prediction result clearly.
    """

    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)

    print(
        "Prediction:",
        result["prediction"].upper(),
    )

    if result.get("raw_prediction"):
        print(
            "Highest-probability class:",
            result["raw_prediction"],
        )

    print(
        "Confidence:",
        result["confidence"],
    )

    print("\nClass probabilities:")

    if result["probabilities"]:
        for class_name, probability in (
            result["probabilities"].items()
        ):
            print(
                f"  {class_name}: "
                f"{probability:.4f}"
            )
    else:
        print("  Not available")

    print("\nImportant LIME features:")

    if result["important_features"]:
        for item in result["important_features"]:
            weight = item["weight"]

            if weight > 0:
                direction = (
                    "supports highest-probability class"
                )
            else:
                direction = (
                    "opposes highest-probability class"
                )

            print(
                f"  {item['feature']!r}: "
                f"{weight:+.4f} "
                f"({direction})"
            )
    else:
        print("  No explanation available")

    print("\nReason:")
    print(result["reason"])

    print("\nWarning:")
    print(result["warning"])


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("FAKE NEWS DETECTION SYSTEM")
    print("=" * 60)

    print("Loaded model:", MODEL_PATH)
    print("\nEnter a full news sentence or paragraph.")

    user_news = input(
        "\nEnter news text: "
    ).strip()

    result = predict_with_reason(
        user_news
    )

    display_result(
        result
    )


if __name__ == "__main__":
    main()


    #.\.venv\Scripts\activate
    #python .\src\predict_news.py run code 


    #real
    #Minister stated that government funding is sufficient for ongoing infrastructure and development projects, according to official reports.

    #fake
    #Breaking news: Scientists have discovered a new planet made entirely of chocolate, defying all
