from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path

# Keep model caches on the E: drive.
os.environ.setdefault("HF_HOME", r"E:\AI_Cache\huggingface")
os.environ.setdefault("HF_HUB_CACHE", r"E:\AI_Cache\huggingface\hub")
os.environ.setdefault("TORCH_HOME", r"E:\AI_Cache\torch")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_PATH = Path("models/xlmr_english_sinhala_cpu_v2")

LABELS = ["fake", "real"]
CLASS_NAMES = ["FAKE/MISLEADING", "REAL/CREDIBLE"]
MAX_LENGTH = 128
MINIMUM_WORDS = 8
UNCERTAIN_THRESHOLD = 0.75

# More LIME samples can make the explanation more stable, but slower on CPU.
LIME_NUM_SAMPLES = 300
LIME_NUM_FEATURES = 10
LIME_BATCH_SIZE = 8
RANDOM_SEED = 42


def configure_terminal_encoding() -> None:
    """Allow Sinhala characters to print correctly on Windows."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def clean_text(text: object) -> str:
    """Normalize whitespace while preserving Sinhala characters."""

    normalized = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", normalized).strip()


def detect_language(text: str) -> str:
    """Detect Sinhala script or English Latin script."""

    sinhala_count = len(re.findall(r"[\u0D80-\u0DFF]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))

    if sinhala_count > 0 and sinhala_count >= latin_count:
        return "si"
    if latin_count > 0:
        return "en"
    if sinhala_count > 0:
        return "si"
    return "unknown"


def display_language(language: str) -> str:
    return {
        "en": "English",
        "si": "Sinhala",
        "unknown": "Unknown",
    }[language]


def validate_text(text: str) -> tuple[bool, str]:
    if not text:
        return False, "No news text was entered."

    word_count = len(text.split())
    if word_count < MINIMUM_WORDS:
        return (
            False,
            f"Please enter at least {MINIMUM_WORDS} words. "
            f"Current word count: {word_count}",
        )

    if detect_language(text) == "unknown":
        return (
            False,
            "The input language could not be identified. "
            "Enter English or Sinhala Unicode text.",
        )

    return True, ""


def load_model() -> tuple:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved bilingual model was not found: {MODEL_PATH}"
        )

    print("Loading saved English-Sinhala XLM-R model...")

    # Do not add fix_mistral_regex=True. The warning shown for this locally
    # saved XLM-R tokenizer is unrelated to the XLM-R tokenizer used here.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    print("Device:", device)
    return tokenizer, model, device


def predict_batch_probabilities(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    """Return [fake, real] probabilities for a batch of texts."""

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=1)

    return probabilities.cpu().numpy()


def predict_probabilities(
    text: str,
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    return predict_batch_probabilities(
        [text], tokenizer, model, device
    )[0]


def display_prediction(
    text: str,
    probabilities: np.ndarray,
) -> tuple[int, float]:
    language = detect_language(text)
    fake_probability = float(probabilities[0])
    real_probability = float(probabilities[1])
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    print("\nPrediction results")
    print("-" * 60)
    print("Detected language:", display_language(language))
    print("Fake/misleading probability:", f"{fake_probability:.2%}")
    print("Real/credible probability:", f"{real_probability:.2%}")

    if confidence < UNCERTAIN_THRESHOLD:
        print(
            "Final prediction: UNCERTAIN "
            f"(leans {CLASS_NAMES[predicted_index]})"
        )
        print(
            "Reason: Confidence is below "
            f"{UNCERTAIN_THRESHOLD:.0%}."
        )
    else:
        print("Final prediction:", CLASS_NAMES[predicted_index])
        print("Model confidence:", f"{confidence:.2%}")

    if language == "si":
        print(
            "Sinhala note: The misinformation class includes "
            "both FALSE and PARTIAL training articles."
        )

    return predicted_index, confidence


def lime_classifier_function(
    tokenizer,
    model,
    device: torch.device,
):
    """Build the probability function required by LIME."""

    def classifier(texts) -> np.ndarray:
        text_list = [clean_text(text) for text in texts]
        probability_batches = []

        for start in range(0, len(text_list), LIME_BATCH_SIZE):
            batch = text_list[start : start + LIME_BATCH_SIZE]
            probability_batches.append(
                predict_batch_probabilities(
                    batch,
                    tokenizer,
                    model,
                    device,
                )
            )

        return np.concatenate(probability_batches, axis=0)

    return classifier


def display_weighted_features(
    title: str,
    weighted_features: list[tuple[str, float]],
) -> None:
    print("\n" + title)
    print("-" * 60)

    if not weighted_features:
        print("No strong features were identified in this section.")
        return

    for feature, weight in weighted_features:
        print(f"{feature:<35} {weight:+.4f}")


def generate_lime_explanation(
    text: str,
    predicted_index: int,
    tokenizer,
    model,
    device: torch.device,
) -> None:
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as exc:
        raise RuntimeError(
            "LIME is not installed. Run: "
            "python -m pip install lime"
        ) from exc

    print("\nGenerating LIME explanation...")
    print("This may take several minutes on the CPU.")

    # Split only at whitespace so Sinhala Unicode word forms remain intact.
    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES,
        split_expression=r"\s+",
        bow=True,
        random_state=RANDOM_SEED,
    )

    explanation = explainer.explain_instance(
        text,
        lime_classifier_function(tokenizer, model, device),
        labels=[predicted_index],
        num_features=min(LIME_NUM_FEATURES, len(text.split())),
        num_samples=LIME_NUM_SAMPLES,
    )

    feature_weights = explanation.as_list(label=predicted_index)
    supporting = [
        (feature, weight)
        for feature, weight in feature_weights
        if weight > 0
    ]
    opposing = [
        (feature, weight)
        for feature, weight in feature_weights
        if weight < 0
    ]

    print("\nLIME MODEL-EVIDENCE EXPLANATION")
    print("=" * 60)
    print("Class being explained:", CLASS_NAMES[predicted_index])

    display_weighted_features(
        f"Words/phrases supporting {CLASS_NAMES[predicted_index]}",
        supporting,
    )
    display_weighted_features(
        f"Words/phrases opposing {CLASS_NAMES[predicted_index]}",
        opposing,
    )

    print(
        "\nInterpretation: Positive weights pushed this prediction "
        "toward the explained class; negative weights pushed against it."
    )
    print(
        "Evidence limitation: These are influential language patterns, "
        "not external proof that the claim is true or false."
    )


def main() -> None:
    configure_terminal_encoding()
    tokenizer, model, device = load_model()

    print("\n" + "=" * 60)
    print("ENGLISH-SINHALA FAKE-NEWS SCREENING WITH LIME XAI")
    print("=" * 60)
    print("Enter an English or Sinhala news article/headline.")
    print("Sinhala must be entered using Sinhala Unicode script.")
    print("A LIME explanation is generated after every prediction.")
    print("Enter Q to close the program.")

    while True:
        print("\n" + "=" * 60)
        user_text = input("News text: ").strip()

        if user_text.lower() in {"q", "quit", "exit"}:
            print("Prediction program closed.")
            break

        cleaned_text = clean_text(user_text)
        is_valid, message = validate_text(cleaned_text)
        if not is_valid:
            print(message)
            continue

        probabilities = predict_probabilities(
            cleaned_text,
            tokenizer,
            model,
            device,
        )
        predicted_index, _ = display_prediction(
            cleaned_text,
            probabilities,
        )

        try:
            generate_lime_explanation(
                cleaned_text,
                predicted_index,
                tokenizer,
                model,
                device,
            )
        except RuntimeError as error:
            print("\nXAI error:", error)

        print(
            "\nImportant: The classifier and LIME do not independently "
            "verify facts against trusted sources."
        )


if __name__ == "__main__":
    main()
