"""
OCR Pipeline — Full Runner

Runs all three pipeline steps in sequence:
    Step 1 — EasyOCR → JSON files (1_run_ocr.py)
    Step 2 — JSON → filtered CSVs (2_filter_and_export.py)
    Step 3 — Merge with human labels (3_merge_human_labels.py)

Usage:
    python run_pipeline.py           # run all 3 steps (clears OCR_data first)
    python run_pipeline.py 2 3       # run only steps 2 and 3 (no clearing)

To run a subset of steps, call the individual scripts directly.
"""

import sys
import os
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
OCR_DATA_DIR = os.path.join(_HERE, "OCR_data")

def banner(step: str, title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {step}: {title}")
    print("=" * 60 + "\n")


def run_step1():
    banner("STEP 1", "EasyOCR → JSON")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_ocr",
        os.path.join(_HERE, "1_run_ocr.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def run_step2():
    banner("STEP 2", "Keyboard & Status Bar Filtering → CSVs")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "filter_and_export",
        os.path.join(_HERE, "2_filter_and_export.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def run_step3():
    banner("STEP 3", "Merge with Human Labels → merged_results.csv")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "merge_human_labels",
        os.path.join(_HERE, "3_merge_human_labels.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    print("=" * 60)
    print("  OCR Pipeline — Full Run")
    print("=" * 60)

    steps = {
        "1": run_step1,
        "2": run_step2,
        "3": run_step3,
    }

    # Allow running specific steps: python run_pipeline.py 2 3
    requested = sys.argv[1:] if len(sys.argv) > 1 else ["1", "2", "3"]

    # On a full run (all 3 steps), clear OCR_data so results are always fresh.
    # On a partial run (e.g. just steps 2 3), leave it intact.
    if requested == ["1", "2", "3"] and os.path.isdir(OCR_DATA_DIR):
        print(f"Clearing OCR_data folder for fresh run: {OCR_DATA_DIR}")
        shutil.rmtree(OCR_DATA_DIR)
        os.makedirs(OCR_DATA_DIR, exist_ok=True)

    for step_id in requested:
        if step_id in steps:
            steps[step_id]()
        else:
            print(f"Unknown step: '{step_id}'. Valid steps: 1, 2, 3")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)
