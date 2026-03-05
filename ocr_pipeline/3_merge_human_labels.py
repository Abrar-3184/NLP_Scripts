"""
OCR Pipeline — Step 3: Merge with Human Labels

Merges the filtered_unfiltered.csv (from Step 2) with human_labeled.csv on
'Screenshot Filename', producing merged_results.csv with columns:

    Screenshot Filename | Unfiltered Text | Filtered Text | Human Labelled

Rows from the OCR CSV that have no matching human label get an empty string.

Usage:
    python 3_merge_human_labels.py
    (edit the config section below to match your paths)
"""

import csv
import os

# ── Configuration (edit these) ─────────────────────────────────────────────────
# Paths are relative to this script's directory so the pipeline works from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))

FILTERED_UNFILTERED_CSV = os.path.join(_HERE, "filtered_unfiltered.csv")
HUMAN_LABELED_CSV       = os.path.join(_HERE, "..", "human_labeled.csv")
OUTPUT_MERGED_CSV       = os.path.join(_HERE, "merged_results.csv")
# ──────────────────────────────────────────────────────────────────────────────


def load_csv_as_dict(path: str, key_col: str) -> dict:
    """Load a CSV into a dict keyed by key_col."""
    result = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            result[row[key_col].strip()] = row
    return result


def main():
    # Validate inputs
    for path, label in [(FILTERED_UNFILTERED_CSV, "filtered_unfiltered.csv"),
                        (HUMAN_LABELED_CSV,       "human_labeled.csv")]:
        if not os.path.exists(path):
            print(f"Error: '{label}' not found at: {path}")
            return

    # Load human labels keyed by Screenshot Filename
    human_labels = load_csv_as_dict(HUMAN_LABELED_CSV, 'Screenshot Filename')
    print(f"Loaded {len(human_labels)} human label(s) from '{HUMAN_LABELED_CSV}'")

    # Merge
    merged = []
    unmatched = 0

    with open(FILTERED_UNFILTERED_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['Screenshot Filename'].strip()
            human_row = human_labels.get(fname)
            human_label = human_row['Human Labelled'] if human_row else ''
            if not human_row:
                unmatched += 1
            merged.append({
                'Screenshot Filename': fname,
                'Unfiltered Text':     row['Unfiltered Text'],
                'Filtered Text':       row['Filtered Text'],
                'Human Labelled':      human_label,
            })

    # Write output
    with open(OUTPUT_MERGED_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'Screenshot Filename', 'Unfiltered Text',
            'Filtered Text', 'Human Labelled'
        ])
        w.writeheader()
        w.writerows(merged)

    print(f"Merged {len(merged)} row(s) → '{OUTPUT_MERGED_CSV}'")
    if unmatched:
        print(f"  Warning: {unmatched} row(s) had no matching human label (left as empty).")
    else:
        print("  All rows matched successfully.")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 3 — Merge with Human Labels")
    print("=" * 60 + "\n")
    main()
