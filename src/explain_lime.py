import pandas as pd
import joblib
from pathlib import Path
from lime.lime_text import LimeTextExplainer

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
XAI_DIR = Path("outputs/xai")

XAI_DIR.mkdir(parents=True, exist_ok=True)

test_df = pd.read_csv(DATA_DIR / "test_clean.csv")

model = joblib.load(MODEL_DIR / "baseline_tfidf_logistic.pkl")

class_names = list(model.classes_)

explainer = LimeTextExplainer(class_names=class_names)

idx = 0
text_instance = test_df.iloc[idx]["clean_text"]
actual_label = test_df.iloc[idx]["label"]
predicted_label = model.predict([text_instance])[0]

exp = explainer.explain_instance(
    text_instance,
    model.predict_proba,
    num_features=10
)

print("Actual Label:", actual_label)
print("Predicted Label:", predicted_label)
print("\nImportant words/features:")
print(exp.as_list())

exp.save_to_file(str(XAI_DIR / "lime_explanation_sample.html"))

print("\nLIME explanation saved:")
print(XAI_DIR / "lime_explanation_sample.html")