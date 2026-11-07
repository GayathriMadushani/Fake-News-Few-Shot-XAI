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

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_PATH = Path(
    "models/xlmr_english_sinhala_cpu_v2"
)

LABELS = ["fake", "real"]
MAX_LENGTH = 128
MINIMUM_WORDS = 8
UNCERTAIN_THRESHOLD = 0.55


def configure_terminal_encoding() -> None:
    """Allow Sinhala characters to print correctly on Windows."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def clean_text(text: object) -> str:
    """Normalize whitespace while preserving Sinhala characters."""

    normalized = unicodedata.normalize(
        "NFC",
        str(text),
    )

    return re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()


def detect_language(text: str) -> str:
    """Detect Sinhala script or English Latin script."""

    sinhala_count = len(
        re.findall(r"[\u0D80-\u0DFF]", text)
    )
    latin_count = len(
        re.findall(r"[A-Za-z]", text)
    )

    if sinhala_count > 0 and sinhala_count >= latin_count:
        return "si"

    if latin_count > 0:
        return "en"

    if sinhala_count > 0:
        return "si"

    return "unknown"


def display_language(language: str) -> str:
    names = {
        "en": "English",
        "si": "Sinhala",
        "unknown": "Unknown",
    }
    return names[language]


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

    language = detect_language(text)

    if language == "unknown":
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

    # Do not add fix_mistral_regex=True. The warning shown for a
    # locally saved XLM-R tokenizer is a known false-positive warning.
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


def predict_probabilities(
    text: str,
    tokenizer,
    model,
    device: torch.device,
) -> torch.Tensor:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(**encoded)
        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )[0]

    return probabilities.cpu()


def display_prediction(
    text: str,
    probabilities: torch.Tensor,
) -> None:
    language = detect_language(text)

    fake_probability = float(probabilities[0])
    real_probability = float(probabilities[1])

    predicted_index = int(
        torch.argmax(probabilities).item()
    )
    predicted_label = LABELS[predicted_index]
    confidence = float(probabilities[predicted_index])

    print("\nPrediction results")
    print("-" * 60)
    print("Detected language:", display_language(language))
    print(
        "Fake/misleading probability:",
        f"{fake_probability:.2%}",
    )
    print(
        "Real/credible probability:",
        f"{real_probability:.2%}",
    )

    if confidence < UNCERTAIN_THRESHOLD:
        print(
            "Final prediction: UNCERTAIN "
            f"(leans {predicted_label.upper()})"
        )
        print(
            "Reason: Confidence is below "
            f"{UNCERTAIN_THRESHOLD:.0%}."
        )
    else:
        display_label = (
            "FAKE/MISLEADING"
            if predicted_label == "fake"
            else "REAL/CREDIBLE"
        )
        print("Final prediction:", display_label)
        print("Model confidence:", f"{confidence:.2%}")

    print(
        "\nImportant: This is a learned language-pattern score, "
        "not independent factual verification."
    )

    if language == "si":
        print(
            "Sinhala note: The misinformation class includes "
            "both FALSE and PARTIAL training articles."
        )


def main() -> None:
    configure_terminal_encoding()
    tokenizer, model, device = load_model()

    print("\n" + "=" * 60)
    print("ENGLISH-SINHALA FAKE-NEWS SCREENING")
    print("=" * 60)
    print("Enter an English or Sinhala news article/headline.")
    print("Sinhala must be entered using Sinhala Unicode script.")
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

        display_prediction(
            cleaned_text,
            probabilities,
        )


if __name__ == "__main__":
    main()
