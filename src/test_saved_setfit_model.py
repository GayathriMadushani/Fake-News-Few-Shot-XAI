from __future__ import annotations

import os
from pathlib import Path

# Prevent accidental Hugging Face downloads.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

from setfit import SetFitModel


MODEL_PATH = Path(
    "models/setfit_distilroberta_32_shot"
)

EXPECTED_LABELS = ["fake", "real"]

TEST_ARTICLES = [
    (
        "The local council approved a new public "
        "transport plan during its meeting on Monday."
    ),
    (
        "Scientists confirmed that drinking only hot "
        "water can permanently cure every known disease."
    ),
]


def main() -> None:
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
    print("Probability label order:", label_order)

    predictions = model.predict(TEST_ARTICLES)
    probabilities = model.predict_proba(
        TEST_ARTICLES
    )

    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()

    if hasattr(probabilities, "detach"):
        probabilities = (
            probabilities
            .detach()
            .cpu()
            .numpy()
        )
    else:
        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

    for number, (
        article,
        prediction,
        probability_row,
    ) in enumerate(
        zip(
            TEST_ARTICLES,
            predictions,
            probabilities,
        ),
        start=1,
    ):
        scores = dict(
            zip(label_order, probability_row)
        )

        print("\n" + "=" * 60)
        print(f"ARTICLE {number}")
        print("=" * 60)
        print(article)
        print("\nPrediction:", prediction)
        print(
            "Fake probability:",
            f"{scores['fake']:.2%}",
        )
        print(
            "Real probability:",
            f"{scores['real']:.2%}",
        )

    print(
        "\nSaved-model test completed successfully."
    )


if __name__ == "__main__":
    main()