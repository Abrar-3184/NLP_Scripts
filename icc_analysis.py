
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Load dataset
    file_path = 'new_emotion_ratios.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)

    # Identify emotions (assuming columns format: Filtered_<emotion>, Human_<emotion>)
    human_cols = [c for c in df.columns if c.startswith('Human_')]
    emotions = [c.replace('Human_', '') for c in human_cols]
    
    icc_results = []

    print(f"Found {len(emotions)} emotions to analyze.")

    # Prepare plots
    num_cols = 4
    num_rows = (len(emotions) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
    axes = axes.flatten()

    for idx, emotion in enumerate(emotions):
        filtered_col = f'Filtered_{emotion}'
        human_col = f'Human_{emotion}'

        if filtered_col not in df.columns:
            print(f"Skipping {emotion}: {filtered_col} missing.")
            continue
        
        # Create temp dataframe for emotion
        temp_df = pd.DataFrame({
            'Target': df.index, # Assuming rows are subjects/screenshots
            'Filtered': df[filtered_col],
            'Human': df[human_col]
        })
        
        # Melt to long format
        df_long = temp_df.melt(id_vars=['Target'], var_name='Rater', value_name='Score')
        
        # Calculate ICC
        try:
            icc = pg.intraclass_corr(data=df_long, targets='Target', raters='Rater', ratings='Score')
            
            # Extract ICC3 value (Single fixed raters)
            icc_val = icc.set_index('Type').loc['ICC3', 'ICC']
            p_val = icc.set_index('Type').loc['ICC3', 'pval']
            
            icc_results.append({
                'Emotion': emotion,
                'ICC': icc_val,
                'p-value': p_val
            })
        except Exception as e:
            print(f"Error calculating ICC for {emotion}: {e}")
            icc_results.append({
                'Emotion': emotion,
                'ICC': 0.0,
                'p-value': 1.0
            })

        # Plotting Scatter
        ax = axes[idx]
        sns.scatterplot(x=df[human_col], y=df[filtered_col], ax=ax, alpha=0.6)
        
        # Add regression line
        sns.regplot(x=df[human_col], y=df[filtered_col], ax=ax, scatter=False, color='red')
        
        # Ideal line
        min_val = min(df[human_col].min(), df[filtered_col].min())
        max_val = max(df[human_col].max(), df[filtered_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], ls="--", c=".3")
        
        ax.set_title(f'{emotion.capitalize()} (ICC: {icc_val:.2f})')
        ax.set_xlabel('Human Score')
        ax.set_ylabel('Filtered OCR Score')

    for i in range(len(emotions), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('icc_scatter_plots.png')
    print("Saved scatter plots to icc_scatter_plots.png")

    # Save Results
    results_df = pd.DataFrame(icc_results).sort_values(by='ICC', ascending=False)
    results_df.to_csv('icc_results.csv', index=False)
    print("Saved ICC results to icc_results.csv")
    
    # Summary Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='ICC', y='Emotion', data=results_df, palette='viridis')
    plt.title('Intraclass Correlation Coefficient (ICC) by Emotion')
    plt.xlim(-0.1, 1.0)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Moderate')
    plt.axvline(x=0.75, color='green', linestyle='--', label='Good')
    plt.legend()
    plt.tight_layout()
    plt.savefig('icc_summary_plot.png')
    print("Saved summary plot to icc_summary_plot.png")

    print("\nTop 5 Emotions by Agreement:")
    print(results_df.head())

if __name__ == "__main__":
    main()
