"""
OCR Pipeline — Step 2: Keyboard & Status Bar Filtering

Reads all JSON files from OCR_DATA_FOLDER, applies keyboard and status-bar filtering, and writes
three CSV output files.

Outputs:
  filtered_only.csv             — Screenshot Filename | Filtered Text
  filtered_unfiltered.csv       — Screenshot Filename | Unfiltered Text | Filtered Text
  filtered_unfiltered_diff.csv  — Screenshot Filename | Unfiltered Text | Filtered Text
                                   | Status Bar Text | Keyboard Text

Usage:
    python 2_filter_and_export.py
    (edit the config section below to match your paths)
"""

import os
import re
import csv
import json
from typing import List, Tuple

from keyboard_detector import ImprovedKeyboardDetector

# ── Configuration (edit these) ─────────────────────────────────────────────────
# Paths are relative to this script's directory so the pipeline works from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))

OCR_DATA_FOLDER                = os.path.join(_HERE, "OCR_data")
OUTPUT_FILTERED_ONLY_CSV       = os.path.join(_HERE, "filtered_only.csv")
OUTPUT_FILTERED_UNFILTERED_CSV = os.path.join(_HERE, "filtered_unfiltered.csv")
OUTPUT_DIFF_CSV                = os.path.join(_HERE, "filtered_unfiltered_diff.csv")
MIN_CONFIDENCE                 = 0.30
STATUS_BAR_RATIO               = 0.05   # Top 5% of image height = status bar
# ──────────────────────────────────────────────────────────────────────────────

detector = ImprovedKeyboardDetector(
    scan_fraction=0.50,
    min_rows=2,
    min_chars_per_row=4,
    row_threshold=60
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def natural_sort_key(s: str) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def load_json(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def infer_height(items: list) -> int:
    """
    Estimate image height purely from bounding boxes.
    Takes the maximum Y coordinate across all items and adds a 1% buffer.
    Falls back to 1000 if there are no items.
    """
    if not items:
        return 1000
    max_y = max(
        max(item['box'][2][1], item['box'][3][1])
        for item in items
    )
    return max(int(max_y * 1.01), 1)


def filter_confidence(items: list, threshold: float = MIN_CONFIDENCE) -> list:
    return [item for item in items if item['conf'] >= threshold]


def center_y(item: dict) -> float:
    return (item['box'][0][1] + item['box'][2][1]) / 2


def split_by_y(items: list, threshold_px: float) -> Tuple[list, list]:
    """Return (items_above_threshold, items_below_threshold)."""
    above, below = [], []
    for item in items:
        (above if center_y(item) <= threshold_px else below).append(item)
    return above, below


def split_keyboard(items: list, regions: list, height: int) -> Tuple[list, list]:
    """Return (keyboard_items, content_items) given fractional keyboard regions."""
    kb, content = [], []
    for item in items:
        cy = center_y(item) / height
        in_kb = any(s <= cy <= e for s, e in regions)
        (kb if in_kb else content).append(item)
    return kb, content


def to_text(items: list) -> str:
    return "\n".join(item['text'].strip() for item in items if item['text'].strip())


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_json(data: dict) -> dict:
    """
    Apply filtering to a single OCR JSON payload.

    Returns a dict with keys:
        filename, unfiltered_text, filtered_text, status_bar_text, keyboard_text
    """
    items = data.get('items', [])

    # Apply confidence threshold
    confident = filter_confidence(items, MIN_CONFIDENCE)

    # Infer height from bounding boxes
    height = infer_height(confident) if confident else 1000

    # Unfiltered text (everything above confidence threshold)
    unfiltered_text = to_text(confident)

    # Status bar: top STATUS_BAR_RATIO of the image
    sb_items, body_items = split_by_y(confident, height * STATUS_BAR_RATIO)
    status_bar_text = to_text(sb_items)

    # Keyboard detection on body
    kb_regions = detector.detect_keyboard_regions(body_items, height)
    if kb_regions:
        kb_items, content_items = split_keyboard(body_items, kb_regions, height)
    else:
        kb_items, content_items = [], body_items

    keyboard_text  = to_text(kb_items)
    filtered_text  = to_text(content_items)

    return {
        'filename':       data.get('filename', os.path.basename('')),
        'unfiltered_text': unfiltered_text,
        'filtered_text':   filtered_text,
        'status_bar_text': status_bar_text,
        'keyboard_text':   keyboard_text,
    }


def main():
    if not os.path.isdir(OCR_DATA_FOLDER):
        print(f"Error: OCR data folder '{OCR_DATA_FOLDER}' not found.")
        return

    json_files = sorted(
        [f for f in os.listdir(OCR_DATA_FOLDER) if f.endswith('.json')],
        key=natural_sort_key
    )

    if not json_files:
        print(f"No JSON files found in '{OCR_DATA_FOLDER}'")
        return

    print(f"Found {len(json_files)} JSON file(s) in '{OCR_DATA_FOLDER}'\n")

    rows = []
    for idx, jf in enumerate(json_files, 1):
        json_path = os.path.join(OCR_DATA_FOLDER, jf)
        print(f"[{idx}/{len(json_files)}] {jf}")
        try:
            data = load_json(json_path)
            result = process_json(data)
            rows.append(result)
            kb_found = bool(result['keyboard_text'])
            sb_found = bool(result['status_bar_text'])
            print(f"  Status bar: {'yes' if sb_found else 'no'}  "
                  f"Keyboard: {'yes' if kb_found else 'no'}")
        except Exception as e:
            print(f"  Error processing {jf}: {e}")

    # Write filtered_only.csv
    with open(OUTPUT_FILTERED_ONLY_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Screenshot Filename', 'Filtered Text'])
        for r in rows:
            w.writerow([r['filename'], r['filtered_text']])
    print(f"\nSaved → {OUTPUT_FILTERED_ONLY_CSV}")

    # Write filtered_unfiltered.csv
    with open(OUTPUT_FILTERED_UNFILTERED_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Screenshot Filename', 'Unfiltered Text', 'Filtered Text'])
        for r in rows:
            w.writerow([r['filename'], r['unfiltered_text'], r['filtered_text']])
    print(f"Saved → {OUTPUT_FILTERED_UNFILTERED_CSV}")

    # Write filtered_unfiltered_diff.csv
    with open(OUTPUT_DIFF_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Screenshot Filename', 'Unfiltered Text', 'Filtered Text',
                    'Status Bar Text', 'Keyboard Text'])
        for r in rows:
            w.writerow([
                r['filename'],
                r['unfiltered_text'],
                r['filtered_text'],
                r['status_bar_text'],
                r['keyboard_text'],
            ])
    print(f"Saved → {OUTPUT_DIFF_CSV}")

    print(f"\nDone. {len(rows)} screenshot(s) processed.")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 2 — Keyboard & Status Bar Filtering")
    print("=" * 60 + "\n")
    main()
