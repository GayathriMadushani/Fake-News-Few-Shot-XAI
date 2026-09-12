"""
Interactive fake-news prediction using the saved 32-shot SetFit model.

Features:
- Displays fake and real probabilities
- Returns "uncertain" for low-confidence predictions
- Uses LIME to explain important words
- Does not retrain or download the model
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

# Prevent accidental Hugging Face downloads.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from lime.lime_text import LimeTextExplainer
from setfit import SetFitModel


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MODEL_PATH = Path(
    "models/setfit_distilroberta_32_shot"
)

EXPECTED_LABELS = ["fake", "real"]

UNCERTAIN_THRESHOLD = 0.55
MINIMUM_WORDS = 8

LIME_FEATURES = 10
LIME_SAMPLES = 300


# ---------------------------------------------------------
# TEXT PROCESSING
# ---------------------------------------------------------

def clean_text(text: object) -> str:
    """Remove unnecessary whitespace without changing words."""

    cleaned = re.sub(
        r"\s+",
        " ",
        str(text),
    )

    return cleaned.strip()


def validate_text(text: str) -> tuple[bool, str]:
    """Check whether the input contains enough information."""

    if not text:
        return False, "No news text was entered."

    word_count = len(text.split())

    if word_count < MINIMUM_WORDS:
        return (
            False,
            f"Please enter at least {MINIMUM_WORDS} words. "
            f"Current word count: {word_count}",
        )

    return True, ""


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

def load_model() -> tuple[SetFitModel, list[str]]:
    """Load the locally saved SetFit model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model was not found: {MODEL_PATH}"
        )

    print("Loading saved 32-shot SetFit model...")

    model = SetFitModel.from_pretrained(
        str(MODEL_PATH),
        device="cpu",
        local_files_only=True,
    )

    label_order = list(
        getattr(model, "labels", None)
        or EXPECTED_LABELS
    )

    if set(label_order) != set(EXPECTED_LABELS):
        raise ValueError(
            f"Unexpected model labels: {label_order}"
        )

    print("Model loaded successfully.")
    print("Model label order:", label_order)

    return model, label_order


# ---------------------------------------------------------
# PROBABILITY PREDICTION
# ---------------------------------------------------------

def convert_to_numpy(values: object) -> np.ndarray:
    """Convert model output into a NumPy array."""

    if hasattr(values, "detach"):
        return (
            values
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        values,
        dtype=float,
    )


def create_probability_function(
    model: SetFitModel,
    model_label_order: list[str],
) -> Callable[[list[str]], np.ndarray]:
    """
    Create the probability function required by LIME.

    Returned columns are always ordered as:
    fake, real
    """

    column_indices = [
        model_label_order.index(label)
        for label in EXPECTED_LABELS
    ]

    def predict_probabilities(
        texts: list[str],
    ) -> np.ndarray:
        cleaned_texts = [
            clean_text(text)
            for text in texts
        ]

        raw_probabilities = model.predict_proba(
            cleaned_texts
        )

        probabilities = convert_to_numpy(
            raw_probabilities
        )

        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(
                1,
                -1,
            )

        return probabilities[:, column_indices]

    return predict_probabilities


# ---------------------------------------------------------
# RESULT DISPLAY
# ---------------------------------------------------------

def display_prediction(
    probabilities: np.ndarray,
) -> int:
    """Display probabilities and return the highest class index."""

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_label = EXPECTED_LABELS[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    print("\nPrediction results")
    print("-" * 50)
    print(
        "Fake probability:",
        f"{probabilities[0]:.2%}",
    )
    print(
        "Real probability:",
        f"{probabilities[1]:.2%}",
    )

    if confidence < UNCERTAIN_THRESHOLD:
        print(
            "Final prediction: UNCERTAIN "
            f"(leans {predicted_label.upper()})"
        )
        print(
            "Reason: The model confidence is below "
            f"{UNCERTAIN_THRESHOLD:.0%}."
        )
    else:
        print(
            "Final prediction:",
            predicted_label.upper(),
        )
        print(
            "Confidence:",
            f"{confidence:.2%}",
        )

    return predicted_index


# ---------------------------------------------------------
# LIME EXPLANATION
# ---------------------------------------------------------

def display_lime_explanation(
    text: str,
    predicted_index: int,
    predict_probabilities: Callable[
        [list[str]],
        np.ndarray,
    ],
) -> None:
    """Generate and display a LIME word explanation."""

    print("\nGenerating LIME explanation...")
    print(
        "This may take a few minutes on the CPU."
    )

    explainer = LimeTextExplainer(
        class_names=EXPECTED_LABELS,
        random_state=42,
    )

    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_probabilities,
        labels=[predicted_index],
        num_features=LIME_FEATURES,
        num_samples=LIME_SAMPLES,
    )

    feature_weights = explanation.as_list(
        label=predicted_index
    )

    supporting_features = [
        (feature, weight)
        for feature, weight in feature_weights
        if weight > 0
    ]

    opposing_features = [
        (feature, weight)
        for feature, weight in feature_weights
        if weight < 0
    ]

    predicted_label = EXPECTED_LABELS[
        predicted_index
    ]

    print(
        f"\nWords supporting the "
        f"{predicted_label.upper()} prediction"
    )
    print("-" * 50)

    if supporting_features:
        for feature, weight in supporting_features:
            print(
                f"{feature:<25} "
                f"+{weight:.4f}"
            )
    else:
        print("No supporting words were identified.")

    print(
        f"\nWords opposing the "
        f"{predicted_label.upper()} prediction"
    )
    print("-" * 50)

    if opposing_features:
        for feature, weight in opposing_features:
            print(
                f"{feature:<25} "
                f"{weight:.4f}"
            )
    else:
        print("No opposing words were identified.")

    print(
        "\nPositive weights support the predicted class. "
        "Negative weights oppose it."
    )


# ---------------------------------------------------------
# INTERACTIVE PROGRAM
# ---------------------------------------------------------

def main() -> None:
    model, model_label_order = load_model()

    predict_probabilities = (
        create_probability_function(
            model,
            model_label_order,
        )
    )

    print("\n" + "=" * 60)
    print("SETFIT FAKE-NEWS PREDICTION WITH LIME")
    print("=" * 60)
    print(
        "Enter a news article or headline."
    )
    print(
        "Enter Q to close the program."
    )

    while True:
        print("\n" + "=" * 60)

        user_text = input(
            "News text: "
        ).strip()

        if user_text.lower() in {
            "q",
            "quit",
            "exit",
        }:
            print("Prediction program closed.")
            break

        cleaned_text = clean_text(user_text)

        is_valid, message = validate_text(
            cleaned_text
        )

        if not is_valid:
            print(message)
            continue

        probabilities = predict_probabilities(
            [cleaned_text]
        )[0]

        predicted_index = display_prediction(
            probabilities
        )

        display_lime_explanation(
            cleaned_text,
            predicted_index,
            predict_probabilities,
        )

        print(
            "\nImportant: This model detects learned "
            "language patterns. It does not independently "
            "verify whether every factual claim is true."
        )


if __name__ == "__main__":
    main()