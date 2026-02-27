import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Load dataset
    file_path = 'new_grouped_emotion_ratios.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    emotions = ['positive', 'negative', 'neutral', 'ambiguous']
    icc_results = []

    print(f"Analyzing {len(emotions)} grouped emotions: {emotions}")

    # Prepare plots
    num_cols = 2
    num_rows = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))
    axes = axes.flatten()

    for idx, emotion in enumerate(emotions):
        filtered_col = f'Filtered_{emotion}'
        human_col = f'Human_{emotion}'

        if filtered_col not in df.columns or human_col not in df.columns:
            print(f"Skipping {emotion}: Column missing ({filtered_col} or {human_col}).")
            continue

        # Prepare DataFrame for Pingouin ICC
        temp_df = pd.DataFrame({
            'Target': df.index,
            'Filtered': df[filtered_col],
            'Human': df[human_col]
        })
        
        df_long = temp_df.melt(id_vars=['Target'], var_name='Rater', value_name='Score')
        
        # Calculate ICC
        try:
            icc = pg.intraclass_corr(data=df_long, targets='Target', raters='Rater', ratings='Score')
            
            # Extract ICC3 value and p-value
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

    plt.tight_layout()
    plt.savefig('icc_grouped_scatter_plots.png')
    print("Saved scatter plots to icc_grouped_scatter_plots.png")

    # Save Results
    results_df = pd.DataFrame(icc_results).sort_values(by='ICC', ascending=False)
    results_df.to_csv('icc_grouped_results.csv', index=False)
    print("Saved ICC results to icc_grouped_results.csv")
    
    # Summary Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x='ICC', y='Emotion', data=results_df, palette='viridis')
    plt.title('ICC by Grouped Emotion')
    plt.xlim(-0.1, 1.0)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Moderate')
    plt.axvline(x=0.75, color='green', linestyle='--', label='Good')
    plt.legend()
    plt.tight_layout()
    plt.savefig('icc_grouped_summary_plot.png')
    print("Saved summary plot to icc_grouped_summary_plot.png")

    print("\nResults:")
    print(results_df)

if __name__ == "__main__":
    main()
