
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("outputs/thesis_figures")
OUT.mkdir(parents=True, exist_ok=True)

def draw_diagram(filename, title, steps):
    fig, ax = plt.subplots(figsize=(11, 2.2 * len(steps)))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 2)
    ax.axis("off")

    for i, (heading, description) in enumerate(steps):
        y = (len(steps) - i - 1) * 2 + 0.4

        box = FancyBboxPatch(
            (1, y), 8, 1.2,
            boxstyle="round,pad=0.15",
            linewidth=1.5,
            edgecolor="#25476A",
            facecolor="#EAF2F8"
        )
        ax.add_patch(box)

        ax.text(
            5, y + 0.82, heading,
            ha="center", va="center",
            fontsize=13, fontweight="bold"
        )
        ax.text(
            5, y + 0.35, description,
            ha="center", va="center",
            fontsize=10
        )

        if i < len(steps) - 1:
            arrow = FancyArrowPatch(
                (5, y - 0.05), (5, y - 0.7),
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=1.6,
                color="#25476A"
            )
            ax.add_patch(arrow)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    for ext in ["png", "svg"]:
        path = OUT / f"{filename}.{ext}"
        fig.savefig(
            path, dpi=300,
            bbox_inches="tight",
            facecolor="white"
        )
        print("CREATED:", path)

    plt.close(fig)


draw_diagram(
    "F01_proposed_research_framework",
    "Proposed Research Framework",
    [
        ("News Dataset", "Sinhala and English news articles"),
        ("Data Preprocessing", "Text cleaning and label preparation"),
        ("Few-Shot Learning", "Balanced 8, 16, 32 and 64-shot experiments"),
        ("Model Training", "SinBERT and XLM-RoBERTa"),
        ("Model Evaluation", "Accuracy, Macro-F1, Precision and Recall"),
        ("Explainable AI", "LIME-based word-level explanations"),
    ]
)

draw_diagram(
    "F02_system_architecture",
    "Fake News Detection System Architecture",
    [
        ("News Text Input", "Sinhala or English news article"),
        ("Text Preparation", "Preprocessing and tokenization"),
        ("Trained XLM-R Model", "Binary fake / real classification"),
        ("Prediction Output", "Predicted label and class probabilities"),
        ("LIME Explanation", "Word-level contribution analysis"),
    ]
)

draw_diagram(
    "F03_data_preprocessing",
    "Data Preprocessing Workflow",
    [
        ("Raw News Articles", "Sinhala and English datasets"),
        ("Data Cleaning", "Text normalization and cleaning"),
        ("Label Preparation", "Fake and Real classes"),
        ("Dataset Partitioning", "Training, validation and testing"),
        ("Few-Shot Sampling", "Balanced training subsets"),
    ]
)

draw_diagram(
    "F04_fewshot_workflow",
    "Few-Shot Learning Experimental Workflow",
    [
        ("Training Dataset", "Labeled news articles"),
        ("Few-Shot Selection", "8, 16, 32 and 64 examples per class"),
        ("Model Fine-Tuning", "SinBERT and XLM-R"),
        ("Validation", "Model selection using validation data"),
        ("Testing", "Evaluation on held-out test data"),
        ("Performance Analysis", "Macro-F1 and class-wise metrics"),
    ]
)

print("\nSUCCESS: All architecture diagrams generated.")
