
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path("outputs")
RESULTS = ROOT / "results"
OUT = ROOT / "thesis_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300
})

def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()
    print("CREATED:", name)

# F05 - Sinhala class distribution
df = pd.read_csv(
    "data/processed/sinhala/sinhala_all_clean.csv"
)
counts = df["label"].value_counts().reindex(
    ["fake", "real"], fill_value=0
)
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(["Fake", "Real"], counts.values)
ax.bar_label(bars)
ax.set_ylabel("Number of articles")
ax.set_title("Sinhala News Dataset: Class Distribution")
save("F05_sinhala_class_distribution.png")

# F06 - Train / Validation / Test
splits = {}
for name, file in [
    ("Train", "sinhala_train_clean.csv"),
    ("Validation", "sinhala_validation_clean.csv"),
    ("Test", "sinhala_test_clean.csv")
]:
    d = pd.read_csv("data/processed/sinhala/" + file)
    splits[name] = d["label"].value_counts()

s = pd.DataFrame(splits).T.fillna(0)
s = s.reindex(columns=["fake", "real"], fill_value=0)

ax = s.plot(kind="bar", figsize=(8, 4.5), rot=0)
ax.set_ylabel("Number of articles")
ax.set_title("Sinhala Dataset Split by Class")
ax.legend(["Fake", "Real"])
save("F06_train_validation_test.png")

# F07 - Few-shot sample distribution
shots = [8, 16, 32, 64]
x = np.arange(len(shots))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - 0.18, shots, 0.36, label="Fake")
ax.bar(x + 0.18, shots, 0.36, label="Real")
ax.set_xticks(x, [str(n) for n in shots])
ax.set_xlabel("Shots per class")
ax.set_ylabel("Training samples")
ax.set_title("Balanced Sinhala Few-Shot Training Sets")
ax.legend()
save("F07_fewshot_distribution.png")

# F08 - Sinhala learning curve
comparison = pd.read_csv(
    RESULTS / "final_research_model_comparison.csv"
)
standard = comparison[
    comparison["model"].isin(["SinBERT", "XLM-R"])
    & (comparison["evaluation_scope"] == "Sinhala")
]
pivot = standard.pivot(
    index="shots_per_class",
    columns="model",
    values="macro_f1"
).sort_index()

ax = pivot.plot(marker="o", figsize=(8, 4.5))
ax.set_xlabel("Training shots per class")
ax.set_ylabel("Test Macro-F1")
ax.set_title("Sinhala Few-Shot Learning Curve")
ax.set_ylim(0, 1)
ax.grid(alpha=0.25)
save("F08_learning_curve.png")

# F09 - Sinhala-only model accuracy
ax = pivot.copy()
acc = standard.pivot(
    index="shots_per_class",
    columns="model",
    values="accuracy"
).sort_index()

acc.plot(kind="bar", figsize=(8, 4.5), rot=0)
plt.xlabel("Shots per class")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)
plt.title("Sinhala Model Accuracy Comparison")
save("F09_accuracy_comparison.png")

# F10 - Sinhala-only Macro-F1
pivot.plot(kind="bar", figsize=(8, 4.5), rot=0)
plt.xlabel("Shots per class")
plt.ylabel("Test Macro-F1")
plt.ylim(0, 1)
plt.title("SinBERT vs XLM-R: Macro-F1")
save("F10_macro_f1_comparison.png")

# F11 - Bilingual model by language
b = pd.read_csv(
    RESULTS / "bilingual_per_language_comparison.csv"
)
b = b[b["language"].isin(["English", "Sinhala"])]
b.set_index("language")[["accuracy", "macro_f1"]].plot(
    kind="bar", figsize=(8, 4.5), rot=0
)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Bilingual XLM-R V2: Performance by Language")
save("F11_bilingual_language_comparison.png")

# F12 - Confusion matrices
with open(
    RESULTS / "xlmr_english_sinhala_cpu_v2_results.json",
    encoding="utf-8"
) as f:
    data = json.load(f)

for language, key in [("English", "en"), ("Sinhala", "si")]:
    matrix = np.array(
        data["test"]["by_language"][key]["confusion_matrix"]
    )
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Fake", "Real"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Bilingual XLM-R V2: {language}")
    save(f"F12_confusion_matrix_{key}.png")

print("\nSUCCESS: Thesis result figures generated.")
print("Folder:", OUT.resolve())
