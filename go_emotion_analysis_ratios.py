import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import os

# --- CONFIG ---
INPUT_FILE = "merged_results.csv"
MODEL_NAME = "cirimus/modernbert-base-go-emotions"
BATCH_SIZE = 8 

# Including all three stages now
TEXT_COLUMNS = {
    "Unfiltered": "Unfiltered Text",
    "Filtered": "Filtered Text",
    "Human": "Human Labelled"
}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    device = 0 if torch.cuda.is_available() else -1
    
    print(f"Loading ModernBERT on: {'GPU' if device == 0 else 'CPU'}...")

    classifier = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        top_k=None,  # This ensures we get the probability for ALL 28 emotions
        device=device,
        batch_size=BATCH_SIZE,
        truncation=True,
        max_length=512
    )

    # Process Unfiltered, Filtered, and Human columns
    for label_prefix, col_name in TEXT_COLUMNS.items():
        if col_name not in df.columns:
            print(f"Skipping {col_name} (not found in CSV)")
            continue
            
        print(f"\nCalculating emotion ratios for {label_prefix} text...")
        texts = df[col_name].fillna("").astype(str).tolist()
        
        all_scores = []
        for out in tqdm(classifier(texts), total=len(texts), desc=f"Scoring {label_prefix}"):
            # Flatten the list of dictionaries into one row of columns
            # Example: {'Unfiltered_joy': 0.12, 'Unfiltered_anger': 0.05...}
            row_scores = {f"{label_prefix}_{item['label']}": round(item['score'], 4) for item in out}
            all_scores.append(row_scores)
        
        # Merge these scores into the main dataframe
        scores_df = pd.DataFrame(all_scores)
        df = pd.concat([df, scores_df], axis=1)

    # Save the final file
    output_file = "new_emotion_ratios.csv" # Update output filename if needed
    df.to_csv(output_file, index=False)
    
    print(f"\nSuccess! Full breakdown saved to {output_file}")
    print(f"The file now has {len(df.columns)} columns.")

if __name__ == "__main__":
    main()
