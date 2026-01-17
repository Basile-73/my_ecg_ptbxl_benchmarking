import pandas as pd
from pathlib import Path
import os
from maps import NAME_MAP, OUR_MODELS, COLOR_MAP

# Models to ignore in the output table
models_to_ignore = [
    # Add model names here to exclude them from the table
    # Example: 'unet_mamba', 'unet_mamba_bidir'
]

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
pivoted_means = combined_df.pivot(index='model', columns=['dataset', 'metric'], values='mean')

# Filter out ignored models
if models_to_ignore:
    pivoted_df = pivoted_df[~pivoted_df.index.isin(models_to_ignore)]
    pivoted_means = pivoted_means[~pivoted_means.index.isin(models_to_ignore)]

# Get the order from NAME_MAP
name_map_order = list(NAME_MAP.keys())

# Separate models into two groups: regular and ours, maintaining NAME_MAP order
regular_models = [m for m in name_map_order if m in pivoted_df.index and m not in OUR_MODELS]
our_models = [m for m in name_map_order if m in pivoted_df.index and m in OUR_MODELS]

# Reorder the dataframes
new_order = regular_models + our_models
pivoted_df = pivoted_df.loc[new_order]
pivoted_means = pivoted_means.loc[new_order]

# Map model names using NAME_MAP and add "(ours)" suffix for our models
def format_model_name(model):
    name = NAME_MAP.get(model, model)
    if model in OUR_MODELS:
        name = f"{name} (ours)"
    return name

pivoted_df.index = pivoted_df.index.map(format_model_name)
pivoted_means.index = pivoted_means.index.map(format_model_name)

print("\nPivoted table:")
print(pivoted_df.head())

def format_latex_table(df, means_df):
    """
    Convert pivoted DataFrame to LaTeX table with formatting:
    - Bold for best value
    - Underline for second-best value
    - For RMSE: lower is better
    - For SNR: higher is better
    """
    # Create a copy to work with
    formatted_df = df.copy()

    # For each column, use unrounded mean values to determine best/second-best
    for col in df.columns:
        dataset, metric = col

        # Use unrounded mean values
        means = means_df[col]

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

    # Modify header to save space
    lines = latex_str.split('\n')
    # Find and modify the header lines
    # The current structure has:
    # dataset & \multicolumn{...}{c}{Dataset Name} \\
    # metric & RMSE & SNR \\
    # model &  &  \\
    # We want:
    #  & \multicolumn{...}{c}{Dataset Name} \\
    # model & RMSE & SNR \\

    new_lines = []
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        # Remove "dataset" and "metric" words
        if line.startswith('dataset &'):
            # Replace "dataset" with empty space
            new_lines.append(line.replace('dataset &', ' &', 1))
        elif line.startswith('metric &'):
            # Skip the metric line entirely, we'll merge it with model line
            metric_parts = line.split(' & ')
            metrics = metric_parts[1:]  # Get RMSE, SNR, etc.
        elif line.startswith('model &'):
            # Replace the model line with model & metrics
            model_line = 'model & ' + ' & '.join(metrics)
            new_lines.append(model_line)
        else:
            new_lines.append(line)

    latex_str = '\n'.join(new_lines)
    lines = latex_str.split('\n')

    # Add midrule before our models
    # Find the first "(ours)" model in the index
    our_model_indices = [i for i, idx in enumerate(df.index) if '(ours)' in str(idx)]
    if our_model_indices:
        first_our_model_idx = our_model_indices[0]
        # Split the LaTeX table and insert midrule
        lines = latex_str.split('\n')
        # Find the line corresponding to the first our model (accounting for header lines)
        # Typically: \begin{tabular}, \toprule, header line, \midrule, then data rows
        data_start_line = None
        for i, line in enumerate(lines):
            if '\\midrule' in line:
                data_start_line = i + 1
                break

        if data_start_line:
            insert_line = data_start_line + first_our_model_idx
            lines.insert(insert_line, '\\midrule')
            latex_str = '\n'.join(lines)

    # Wrap the table in LaTeX table environment
    wrapped_table = (
        "\\begin{table}[htbp]\n"
        "\\caption{Reconstruction Performance Metrics}\n"
        "\\begin{center}\n"
        f"{latex_str}\n"
        "\\label{tab1}\n"
        "\\end{center}\n"
        "\\end{table}"
    )

    return wrapped_table

# Generate LaTeX table
latex_table = format_latex_table(pivoted_df, pivoted_means)
print("\nLaTeX table:")
print(latex_table)

# Optional: Save the combined dataframe
results_path = os.path.join(csv_dir, 'aggregated')
pivoted_df.to_csv(os.path.join(results_path, 'pivoted_results.csv'))
with open(os.path.join(results_path, 'results_table.tex'), 'w') as f:
    f.write(latex_table)
