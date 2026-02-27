# NLP_Scripts

# OCR Pipeline
## Pipeline Components

| Script | Run command | What it does |
|---|---|---|
| `1_run_ocr.py` | `python 1_run_ocr.py` | Runs EasyOCR on images and saves one JSON per image to `OCR_data/` |
| `2_filter_and_export.py` | `python 2_filter_and_export.py` | Reads JSONs from `OCR_data/`, removes keyboard and status bar text, writes 3 CSVs |
| `3_merge_human_labels.py` | `python 3_merge_human_labels.py` | Merges filtered results with human-labeled data into a single CSV |
| `run_pipeline.py` | `python run_pipeline.py` | Runs all 3 steps in sequence (clears `OCR_data/` first for a fresh run) |

---

## Running the Pipeline

### Full run (all 3 steps, fresh OCR)
```
python ocr_pipeline/run_pipeline.py
```
> Clears `OCR_data/` before Step 1 so every run starts fresh.

### Partial run (skip OCR, re-filter existing JSONs)
```
python ocr_pipeline/run_pipeline.py 2 3
```
> Use this when the OCR was run on a different machine and the JSONs are already in `OCR_data/`.

### Run a single step
```
python ocr_pipeline/1_run_ocr.py
python ocr_pipeline/2_filter_and_export.py
python ocr_pipeline/3_merge_human_labels.py
```

---

## Outputs

| File | Columns |
|---|---|
| `filtered_only.csv` | Screenshot Filename, Filtered Text |
| `filtered_unfiltered.csv` | Screenshot Filename, Unfiltered Text, Filtered Text |
| `filtered_unfiltered_diff.csv` | Screenshot Filename, Unfiltered Text, Filtered Text, Status Bar Text, Keyboard Text |
| `merged_results.csv` | Screenshot Filename, Unfiltered Text, Filtered Text, Human Labelled |

All output CSVs are saved inside the `ocr_pipeline/` folder.

---

## Changing the Input Data

### Changing the screenshot folder (Step 1)
Edit `INPUT_FOLDER` at the top of `1_run_ocr.py`:
```python
INPUT_FOLDER = os.path.join(_HERE, "..", "keyboard_test")  # ← change this
```

### Changing the OCR data folder
Edit `OCR_DATA_FOLDER` in `1_run_ocr.py` (where JSONs are saved):
```python
OCR_DATA_FOLDER = os.path.join(_HERE, "OCR_data")  # ← change this
```
And update `OCR_DATA_FOLDER` in `2_filter_and_export.py` to match (where JSONs are read from):
```python
OCR_DATA_FOLDER = os.path.join(_HERE, "OCR_data")  # ← change this
```

### Changing the human labels file (Step 3)
Edit `HUMAN_LABELED_CSV` at the top of `3_merge_human_labels.py`:
```python
HUMAN_LABELED_CSV = os.path.join(_HERE, "..", "human_labeled.csv")  # ← change this
```

### Changing output CSV filenames or locations
Edit the output path variables at the top of `2_filter_and_export.py` or `3_merge_human_labels.py`:
```python
OUTPUT_FILTERED_ONLY_CSV       = os.path.join(_HERE, "filtered_only.csv")
OUTPUT_FILTERED_UNFILTERED_CSV = os.path.join(_HERE, "filtered_unfiltered.csv")
OUTPUT_DIFF_CSV                = os.path.join(_HERE, "filtered_unfiltered_diff.csv")
OUTPUT_MERGED_CSV              = os.path.join(_HERE, "merged_results.csv")
```

### Changing the OCR confidence threshold
Edit `MIN_CONFIDENCE` at the top of `1_run_ocr.py` or `2_filter_and_export.py`:
```python
MIN_CONFIDENCE = 0.30  # ← raise to be stricter, lower to include more text
```

---
# Running emotion classification and evaluation

## Run emotion classification

### Run Grouped
```
python go_emotion_analysis.py
```

### Run Individual
```
python go_emotion_analysis_ratios.py
```

### Run ICC eval (Grouped)
```
python icc_grouped_analysis.py
```

### Run ICC eval (Individual)
```
python icc_analysis.py
```

### Change paths for the above scripts mentioned in the scripts 




