"""
Step 2: Group-wise shortcut ablation.

Run from the project root (needs audit_shortcuts.py in the same folder):
    python src/audit_shortcut_groups.py

For each group of suspicious words, it hides the group and measures:
  B = original model on masked test text (does the model rely on it?)
  C = model retrained on masked text (can it find other cues?)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from audit_shortcuts import evaluate, load, new_model

OUT_DIR = Path("outputs/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]
WEEKDAYS = [
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
]

# group name -> (plain words, extra regex patterns)
# Edit these lists freely after you see the results.
GROUPS = {
    "source_markers": (
        ["reuters", "video", "via", "breitbart", "sputnik",
         "featured", "image", "getty"],
        [r"\bpictwittercom\w*", r"\bhttps?\w*", r"\bwww\w*"],
    ),
    "dates_and_time": (
        MONTHS + WEEKDAYS,
        [r"\b(?:19|20)\d{2}\b"],
    ),
    "clickbait_words": (
        ["watch", "share", "breaking", "click", "please", "wire",
         "truth", "you", "your", "this", "what", "why", "how",
         "like", "just", "more"],
        [],
    ),
    "attribution_style": (
        ["said", "spokesman", "spokeswoman", "spokesperson",
         "minister", "ministry", "official", "officials", "told",
         "according", "reported", "statement", "briefing",
         "lawmakers", "senate", "parliament"],
        [],
    ),
    "election_topics": (
        ["hillary", "hillarys", "clinton", "obama", "trump",
         "donald", "america", "american", "americans"],
        [],
    ),
}


def compile_group(words: list[str], patterns: list[str]) -> re.Pattern:
    parts = []
    if words:
        parts.append(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b")
    parts.extend(patterns)
    return re.compile("|".join(parts))


def apply_mask(texts: pd.Series, pattern: re.Pattern) -> pd.Series:
    masked = texts.map(lambda t: pattern.sub(" ", t))
    return masked.map(lambda t: re.sub(r"\s+", " ", t).strip())


def main() -> None:
    train_df = load("train_clean.csv")
    test_df = load("test_clean.csv")
    y_train = train_df["label"]
    y_test = test_df["label"]

    base_model = new_model()
    base_model.fit(train_df["clean_text"], y_train)
    base_acc, _ = evaluate(base_model, test_df["clean_text"], y_test)
    print(f"Original accuracy: {base_acc:.4f}\n")

    compiled = {
        name: compile_group(words, patterns)
        for name, (words, patterns) in GROUPS.items()
    }

    all_words = [w for words, _ in GROUPS.values() for w in words]
    all_patterns = [p for _, patterns in GROUPS.values() for p in patterns]
    compiled["ALL_GROUPS"] = compile_group(all_words, all_patterns)

    rows = []
    for name, pattern in compiled.items():
        masked_test = apply_mask(test_df["clean_text"], pattern)
        masked_train = apply_mask(train_df["clean_text"], pattern)

        acc_b, _ = evaluate(base_model, masked_test, y_test)

        retrained = new_model()
        retrained.fit(masked_train, y_train)
        acc_c, f1_c = evaluate(retrained, masked_test, y_test)

        rows.append(
            {
                "masked_group": name,
                "acc_original_model": acc_b,
                "drop_B": base_acc - acc_b,
                "acc_retrained": acc_c,
                "drop_C": base_acc - acc_c,
                "macro_f1_retrained": f1_c,
            }
        )

    result = pd.DataFrame(rows).sort_values("drop_B", ascending=False)
    result.to_csv(OUT_DIR / "shortcut_group_ablation.csv", index=False)

    print(result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {OUT_DIR / 'shortcut_group_ablation.csv'}")
    print(
        "\nHow to read it: a large drop_B means the original model "
        "depends on that group.\nA small drop_C means the model can "
        "still find other cues without it."
    )


if __name__ == "__main__":
    main()
