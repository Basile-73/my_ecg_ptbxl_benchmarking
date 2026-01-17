import pandas as pd
from pathlib import Path
from maps import NAME_MAP

# Directory containing the CSV files
csv_dir = Path('/local/home/bamorel/my_ecg_ptbxl_benchmarking/new_code/outputs/AAA_tabels')

# Get all CSV files in the directory
csv_files = list(csv_dir.glob('*.csv'))

dataset_names = {
    'ptb-xl': 'PTB-XL',
    'sinus' : 'MIT-BIH NSR'
}

def get_ci_max(mean, ci_low, ci_high):
    return max(
        ci_high - mean,
        mean - ci_low
    )

def get_text(mean, ci_max):
    return f"{mean:.2f} ± {ci_max:.2f}"

# Read and concatenate all CSV files
dfs = []
for csv_file in csv_files:
    dataset = csv_file.stem  # Get the file name without extension
    df = pd.read_csv(csv_file)
    df['dataset'] = dataset_names[dataset]
    df['ci_max'] = df.apply(lambda row: get_ci_max(row['mean'], row['ci_low'], row['ci_high']), axis=1)
    df['text'] = df.apply(lambda row: get_text(row['mean'], row['ci_max']), axis=1)
    dfs.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

print(f"Loaded {len(csv_files)} CSV files")
print(f"Combined dataframe shape: {combined_df.shape}")
print("\nFirst few rows:")
print(combined_df.head())

# Pivot to get model as rows, and (dataset, metric) as multi-index columns
pivoted_df = combined_df.pivot(index='model', columns=['dataset', 'metric'], values='text')

# Map model names using NAME_MAP
pivoted_df.index = pivoted_df.index.map(lambda x: NAME_MAP.get(x, x))

print("\nPivoted table:")
print(pivoted_df.head())

def format_latex_table(df):
    """
    Convert pivoted DataFrame to LaTeX table with formatting:
    - Bold for best value
    - Underline for second-best value
    - For RMSE: lower is better
    - For SNR: higher is better
    """
    # Create a copy to work with
    formatted_df = df.copy()

    # For each column, extract mean values and determine best/second-best
    for col in df.columns:
        dataset, metric = col

        # Extract numerical mean values from text (format: "mean ± ci")
        means = df[col].apply(lambda x: float(x.split(' ± ')[0]) if pd.notna(x) else float('nan'))

        # Determine best and second-best based on metric type
        if metric == 'RMSE':
            # For RMSE, lower is better
            sorted_indices = means.argsort()
        else:  # SNR or other metrics where higher is better
            sorted_indices = means.argsort()[::-1]

        # Get indices of best and second-best (excluding NaN values)
        valid_indices = [idx for idx in sorted_indices if not pd.isna(means.iloc[idx])]

        if len(valid_indices) >= 1:
            best_idx = valid_indices[0]
            formatted_df.loc[df.index[best_idx], col] = f"\\textbf{{{df.iloc[best_idx][col]}}}"

        if len(valid_indices) >= 2:
            second_best_idx = valid_indices[1]
            formatted_df.loc[df.index[second_best_idx], col] = f"\\underline{{{df.iloc[second_best_idx][col]}}}"

    # Convert to LaTeX
    latex_str = formatted_df.to_latex(
        escape=False,
        column_format='l' + 'c' * len(df.columns),
        multicolumn_format='c'
    )

    return latex_str

# Generate LaTeX table
latex_table = format_latex_table(pivoted_df)
print("\nLaTeX table:")
print(latex_table)

# Optional: Save the combined dataframe
results_path = os.path.join(csv_dir, 'aggregated')
pivoted_df.to_csv(os.path.join(results_path, 'pivoted_results.csv'))
with open(os.path.join(results_path, 'results_table.tex'), 'w') as f:
    f.write(latex_table)
