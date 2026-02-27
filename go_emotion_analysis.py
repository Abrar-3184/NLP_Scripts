import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
INPUT_FILE = "All_Screenshots_Filtered_Unfiltered_Human.csv"
MODEL_NAME = "cirimus/modernbert-base-go-emotions"
BATCH_SIZE = 16  # Processes 16 rows at a time (perfect for GPU)

TEXT_COLUMNS = {
    "Unfiltered": "Unfiltered PaddleOCR Result",
    "Filtered": "Filtered PaddleOCR Result",
    "Human": "Human Labelled"
}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Check for GPU
    device = 0 if torch.cuda.is_available() else -1
    print(f"--- Running on: {'GPU (Fast)' if device == 0 else 'CPU (Slow)'} ---")

    # Initialize pipeline with batching
    classifier = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        top_k=1,
        device=device,
        batch_size=BATCH_SIZE,
        truncation=True
    )

    # Analyze each column
    for label, col_name in TEXT_COLUMNS.items():
        if col_name not in df.columns:
            continue
            
        print(f"\nProcessing {label} text...")
        # Prepare list and handle empty cells
        texts = df[col_name].fillna("").astype(str).tolist()
        
        results = []
        # tqdm shows the progress bar [##########] 100%
        for out in tqdm(classifier(texts), total=len(texts)):
            results.append(out[0]['label'])
            
        df[f'emotion_{label.lower()}'] = results

    # Save results
    df.to_csv("emotion_comparison_results.csv", index=False)
    print("\nCSV saved: emotion_comparison_results.csv")

    # Visualization
    summary = []
    for label in TEXT_COLUMNS.keys():
        col = f'emotion_{label.lower()}'
        if col in df.columns:
            pcts = df[col].value_counts(normalize=True) * 100
            for emotion, val in pcts.items():
                summary.append({'Source': label, 'Emotion': emotion, 'Percentage': val})

    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    sns.barplot(data=pd.DataFrame(summary), x='Emotion', y='Percentage', hue='Source')
    plt.title('Emotion Distribution Across OCR Stages', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('emotion_analysis_comparison.png')
    print("Graph saved: emotion_analysis_comparison.png")

if __name__ == "__main__":
    main()
