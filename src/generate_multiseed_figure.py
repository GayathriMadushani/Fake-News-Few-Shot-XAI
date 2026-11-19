
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

source = Path("outputs/results/sinhala_multiseed_comparison.csv")
output = Path("outputs/thesis_figures")
output.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source)

models = ["SinBERT", "XLM-R"]
seeds = [42, 123]

x = np.arange(len(models))
width = 0.32

fig, ax = plt.subplots(figsize=(9, 5.5))

for i, seed in enumerate(seeds):
    values = [
        df.loc[
            (df["model"] == model) & (df["seed"] == seed),
            "macro_f1"
        ].iloc[0]
        for model in models
    ]

    bars = ax.bar(
        x + (i - 0.5) * width,
        values,
        width,
        label=f"Seed {seed}"
    )

    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)

ax.set_title(
    "Multi-Seed Performance Comparison (64-Shot)",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Model")
ax.set_ylabel("Macro-F1 Score")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 0.85)
ax.legend(title="Random Seed")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

fig.tight_layout()

for ext in ["png", "svg"]:
    path = output / f"F13_multiseed_performance.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("CREATED:", path)

plt.close(fig)

print("\nSUCCESS: F13 Multi-Seed Performance Graph generated.")
