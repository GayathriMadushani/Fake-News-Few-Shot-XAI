"""
Project statistics: dataset sizes (English vs Sinhala) and saved confusion matrices.

Run from the project root (the Fake-News-Few-Shot-XAI folder):
    python src/project_stats.py

It only READS your files. It writes one new file:
    outputs/results/dataset_sizes.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(".")
DATA_DIR = ROOT / "data"
SEARCH_DIRS = [ROOT / "outputs", ROOT / "models"]
OUT_PATH = ROOT / "outputs" / "results" / "dataset_sizes.csv"

SINHALA = re.compile(r"[\u0D80-\u0DFF]")
TEXT_COLS = ["clean_text", "text", "Text", "content", "article", "statement", "body", "title"]
LABEL_COLS = ["label", "Label", "class", "target", "type", "category"]
MAX_JSON_BYTES = 5_000_000


def read_csv_any(path: Path) -> pd.DataFrame | None:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as error:  # noqa: BLE001
            print(f"  could not read {path}: {error}")
            return None
    return None


def first_present(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def dataset_sizes() -> None:
    print("=" * 78)
    print("PART 1: DATASET SIZES (every CSV and JSONL under data/)")
    print("=" * 78)

    rows = []
    for path in sorted(DATA_DIR.rglob("*")):
        if path.suffix.lower() == ".jsonl":
            with open(path, encoding="utf-8", errors="ignore") as handle:
                count = sum(1 for _ in handle)
            rows.append({"file": str(path), "rows": count, "english_rows": "", "sinhala_rows": "", "labels": ""})
            continue
        if path.suffix.lower() != ".csv":
            continue

        df = read_csv_any(path)
        if df is None:
            continue

        text_col = first_present(df.columns, TEXT_COLS)
        label_col = first_present(df.columns, LABEL_COLS)

        sinhala_rows = english_rows = ""
        if text_col is not None:
            has_sinhala = df[text_col].fillna("").astype(str).map(lambda t: bool(SINHALA.search(t)))
            sinhala_rows = int(has_sinhala.sum())
            english_rows = int(len(df) - sinhala_rows)

        labels = ""
        if label_col is not None:
            labels = ", ".join(f"{k}={v}" for k, v in df[label_col].value_counts().items())

        rows.append(
            {
                "file": str(path),
                "rows": len(df),
                "english_rows": english_rows,
                "sinhala_rows": sinhala_rows,
                "labels": labels,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        print("No CSV files found under data/. Run this from the project root.")
        return

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(table.to_string(index=False))
    print(
        "\nNote: 'sinhala_rows' counts rows whose text contains Sinhala letters; "
        "'english_rows' counts the other rows.\n"
        "A row that mixes both languages is counted as Sinhala."
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


def find_keys(obj, needle: str, path: str = ""):
    """Yield (path, value) for every key that contains the needle."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else str(key)
            if needle in str(key).lower():
                yield here, value
            yield from find_keys(value, needle, here)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from find_keys(value, needle, f"{path}[{i}]")


def confusion_matrices() -> None:
    print("\n" + "=" * 78)
    print("PART 2: SAVED CONFUSION MATRICES (outputs/ and models/)")
    print("=" * 78)

    found = 0

    for base in SEARCH_DIRS:
        if not base.exists():
            continue

        for path in sorted(base.rglob("*.json")):
            if path.stat().st_size > MAX_JSON_BYTES:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:  # noqa: BLE001
                continue

            hits = list(find_keys(data, "confusion"))
            if not hits:
                continue

            labels = None
            for _, value in find_keys(data, "confusion_matrix_labels"):
                labels = value

            print(f"\nFile: {path}")
            for key_path, value in hits:
                if key_path.endswith("labels"):
                    continue
                print(f"  {key_path}  (rows = actual, columns = predicted)")
                if labels:
                    print(f"  label order: {labels}")
                if isinstance(value, list):
                    for row in value:
                        print("    ", row)
                else:
                    print("    ", value)
                found += 1

        for path in sorted(base.rglob("*.txt")):
            if path.stat().st_size > MAX_JSON_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "onfusion" not in text:
                continue
            lines = text.splitlines()
            print(f"\nFile: {path}")
            for i, line in enumerate(lines):
                if "onfusion" in line:
                    print("  " + "\n  ".join(lines[i : i + 4]))
                    found += 1

    if found == 0:
        print("No saved confusion matrices were found in outputs/ or models/.")
    else:
        print(f"\nFound {found} confusion matrix entries.")


if __name__ == "__main__":
    dataset_sizes()
    confusion_matrices()
