import re
import joblib
from pathlib import Path
from lime.lime_text import LimeTextExplainer

MODEL_PATH = Path("models/few_32_shot_model.pkl")

model = joblib.load(MODEL_PATH)

class_names = list(model.classes_)
explainer = LimeTextExplainer(class_names=class_names)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_with_reason(news_text):
    cleaned = clean_text(news_text)

    prediction = model.predict([cleaned])[0]
    probability = model.predict_proba([cleaned]).max()

    explanation = explainer.explain_instance(
        cleaned,
        model.predict_proba,
        num_features=8
    )

    important_features = explanation.as_list()
    important_words = [word for word, score in important_features[:5]]

    result = {
        "prediction": prediction,
        "confidence": round(float(probability), 3),
        "important_words": important_words,
        "reason": f"The model predicted this news as '{prediction}' because the words/features {important_words} strongly influenced the prediction."
    }

    return result


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