import pandas as pd

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from new_code.utils.getters import get_model
from new_code.visualisation.maps import COLOR_MAP, OUR_MODELS, NAME_MAP, EXCLUDE_MODELS, CLASSIFICATION_MODEL_NAMES, CLASSIFICATION_MODEL_NAMES, plot_font_sizes

classifier = 'fastai_inception1d'
df = pd.read_csv('output/all_100_nbp_strong/downstream_results/exp1.1.1/per_class_roc_results.csv')
df_raw = df
df = df[df['classifier']==classifier]

# Exclude models based on EXCLUDE_MODELS
df = df[~df['denoising_model'].isin(EXCLUDE_MODELS)]

# put column denoising_model first
cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index('denoising_model')))
df = df[cols]

# add a new columng with "roc_auc (+/- max(abs(upper-mean), abs(mean-lower)))"
def format_ci(row):
    mean = row['roc_auc']
    lower = row['lower']
    upper = row['upper']
    ci = max(abs(upper - mean), abs(mean - lower))
    return f"{mean:.3f} (+/- {ci:.3f})"

df['roc_auc_ci'] = df.apply(format_ci, axis=1)

# add a column to keep track of the highest and second highest roc_auc based on col 'roc_auc'
def rank_roc_auc(sub_df):
    # Separate clean and noisy from other models
    special_models = ['clean', 'noisy']
    is_special = sub_df['denoising_model'].isin(special_models)

    regular_df = sub_df[~is_special].copy()
    special_df = sub_df[is_special].copy()

    # Rank regular models by roc_auc
    regular_df = regular_df.sort_values(by='roc_auc', ascending=False).reset_index(drop=True)
    regular_df['rank'] = regular_df.index + 1

    # Assign last ranks to clean and noisy
    if len(special_df) > 0:
        special_df = special_df.sort_values(by='roc_auc', ascending=False).reset_index(drop=True)
        special_df['rank'] = range(len(regular_df) + 1, len(regular_df) + len(special_df) + 1)

    # Combine back
    result = pd.concat([regular_df, special_df], ignore_index=True)
    return result

df = df.groupby('diagnosis').apply(rank_roc_auc).reset_index(drop=True)

# keep only columns diagnosis, denoising_model, roc_auc_ci, rank
df = df[['diagnosis', 'denoising_model', 'roc_auc_ci', 'rank']]
# pivot table to have denoising_model as columns
df_pivot = df.pivot(index='denoising_model', columns='diagnosis', values=['roc_auc_ci', 'rank'])

# Format as latex table. Make highest roc_auc bold and second highest underlined. drop the rank columns

def format_latex(roc_auc_ci, rank):
    if pd.isna(roc_auc_ci):
        return ""
    if rank == 1:
        return r"\textbf{" + roc_auc_ci + "}"
    elif rank == 2:
        return r"\underline{" + roc_auc_ci + "}"
    else:
        return roc_auc_ci

final_df = pd.DataFrame()
for diagnosis in df['diagnosis'].unique():
    final_df[diagnosis] = df_pivot.apply(
        lambda row: format_latex(row['roc_auc_ci', diagnosis], row['rank', diagnosis]),
        axis=1
    )
final_df.index = df_pivot.index

# Keep track of original model names before renaming
original_names = final_df.index.tolist()

# Rename models using NAME_MAP and add (ours) suffix
def rename_model(original_name):
    mapped_name = NAME_MAP.get(original_name, original_name)
    if original_name in OUR_MODELS:
        return f"{mapped_name} (ours)"
    return mapped_name

final_df.index = [rename_model(name) for name in original_names]

# Sort models: our models first, then others, following NAME_MAP order
our_models_renamed = [rename_model(name) for name in original_names if name in OUR_MODELS]
other_models_renamed = [rename_model(name) for name in original_names if name not in OUR_MODELS]

# Create order based on NAME_MAP keys
name_map_order = list(NAME_MAP.keys())
def get_sort_key(original_name):
    try:
        return name_map_order.index(original_name)
    except ValueError:
        return len(name_map_order)

our_models_sorted = sorted(
    [(name, orig) for name, orig in zip([rename_model(n) for n in original_names if n in OUR_MODELS],
                                        [n for n in original_names if n in OUR_MODELS])],
    key=lambda x: get_sort_key(x[1])
)
other_models_sorted = sorted(
    [(name, orig) for name, orig in zip([rename_model(n) for n in original_names if n not in OUR_MODELS],
                                        [n for n in original_names if n not in OUR_MODELS])],
    key=lambda x: get_sort_key(x[1])
)

ordered_models = [x[0] for x in our_models_sorted] + [x[0] for x in other_models_sorted]
final_df = final_df.reindex(ordered_models)

# Generate LaTeX table and insert dashed line
latex_table = final_df.to_latex(escape=False)

# Insert dashed line after our models
if len(our_models_sorted) > 0:
    lines = latex_table.split('\n')
    # Find where to insert the dashed line (after our models)
    # Account for: \begin{tabular}, \toprule, header, \midrule (4 lines before data rows)
    insert_index = 4 + len(our_models_sorted)
    if insert_index < len(lines):
        lines.insert(insert_index, r'\hdashline')
        latex_table = '\n'.join(lines)

print(latex_table)
