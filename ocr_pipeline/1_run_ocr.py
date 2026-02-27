"""
OCR Pipeline — Step 1: Run EasyOCR and save results as JSON

Reads images from INPUT_FOLDER, runs EasyOCR on each, and saves one JSON file
per image into OCR_DATA_FOLDER. Already-processed images are skipped.

Usage:
    python 1_run_ocr.py
    (edit the config section below to point to your folders)
"""

import os
import re
import json

import easyocr

# ── Configuration (edit these) ─────────────────────────────────────────────────
# Paths are relative to this script's directory so the pipeline works from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER    = os.path.join(_HERE, "..", "screenshot_data")  # Folder with screenshots
OCR_DATA_FOLDER = os.path.join(_HERE, "OCR_data")             # Where to save JSON files
MIN_CONFIDENCE  = 0.30                                         # Min OCR confidence
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG')


def natural_sort_key(s: str) -> list:
    """Sort filenames with embedded numbers."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def main():
    os.makedirs(OCR_DATA_FOLDER, exist_ok=True)

    images = sorted(
        [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(IMAGE_EXTENSIONS)],
        key=natural_sort_key
    )

    if not images:
        print(f"No images found in '{INPUT_FOLDER}'")
        return

    print(f"Found {len(images)} image(s) in '{INPUT_FOLDER}'")
    print(f"Output folder: '{OCR_DATA_FOLDER}'\n")

    # Initialise EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)

    skipped = 0
    processed = 0

    for idx, fname in enumerate(images, 1):
        json_name = os.path.splitext(fname)[0] + ".json"
        json_path = os.path.join(OCR_DATA_FOLDER, json_name)

        # Skip already-processed files
        if os.path.exists(json_path):
            print(f"[{idx}/{len(images)}] SKIP (already done): {fname}")
            skipped += 1
            continue

        image_path = os.path.join(INPUT_FOLDER, fname)
        print(f"[{idx}/{len(images)}] Processing: {fname}")

        try:
            raw_results = reader.readtext(image_path)
        except Exception as e:
            print(f"  OCR error: {e}")
            continue

        # Convert EasyOCR output to a clean list of items
        items = []
        for bbox, text, prob in raw_results:
            if float(prob) < MIN_CONFIDENCE:
                continue
            if not text.strip():
                continue
            # EasyOCR returns numpy types for coordinates; cast to plain Python
            items.append({
                "text": text,
                "box":  [[float(pt[0]), float(pt[1])] for pt in bbox],
                "conf": round(float(prob), 4)
            })

        payload = {
            "filename": fname,
            "items":    items
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"  Saved {len(items)} item(s) → {json_path}")
        processed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1 — EasyOCR → JSON")
    print("=" * 60 + "\n")
    main()
