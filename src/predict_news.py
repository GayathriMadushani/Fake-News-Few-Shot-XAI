import re
import joblib
from pathlib import Path
from lime.lime_text import LimeTextExplainer

# Load model
MODEL_PATH = Path("models/few_32_shot_model.pkl")
model = joblib.load(MODEL_PATH)

# Class names (fake/real)
class_names = list(model.classes_)
explainer = LimeTextExplainer(class_names=class_names)


# -----------------------------
# TEXT CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# PREDICTION + EXPLANATION
# -----------------------------
def predict_with_reason(news_text):

    # safety check (VERY IMPORTANT for LIME crash fix)
    if len(news_text.split()) < 3:
        return {
            "prediction": "Invalid Input",
            "confidence": 0.0,
            "important_words": [],
            "reason": "Please enter a longer news sentence for prediction."
        }

    cleaned = clean_text(news_text)

    # prediction
    prediction = model.predict([cleaned])[0]
    probability = model.predict_proba([cleaned]).max()

    # LIME explanation (FIXED)
    explanation = explainer.explain_instance(
        cleaned,
        model.predict_proba,
        num_features=5,
        num_samples=100
    )

    important_features = explanation.as_list()
    important_words = [word for word, score in important_features[:5]]

    result = {
        "prediction": prediction,
        "confidence": round(float(probability), 3),
        "important_words": important_words,
        "reason": f"The model predicted '{prediction}' based on key words: {important_words}"
    }

    return result


# -----------------------------
# MAIN PROGRAM
# -----------------------------
if __name__ == "__main__":
    print("Fake News Detection System")
    print("-" * 40)

    user_news = input("Enter news text: ")

    result = predict_with_reason(user_news)

    print("\nFinal Output")
    print("-" * 40)
    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"])
    print("Important Words:", result["important_words"])
    print("Reason:", result["reason"])