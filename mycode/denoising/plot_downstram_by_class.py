import pandas as pd
import os

#csv_path = 'output/all_100_nbp_moderate/downstream_results/exp1.1.1/per_class_roc_results.csv'
csv_path = 'mycode/denoising/output/test_ls/downstream_results/exp1.1.1/per_class_roc_results_exp1.1.1.csv'
parent_dir = csv_path.rsplit('/', 1)[0]
df = pd.read_csv(csv_path)
df.rename(columns={'classifier':'classification_model',
                   'roc_auc': 'auc',
                   'lower':'auc_lower',
                   'upper': 'auc_upper'}, inplace=True)

# create a folder for each unique class
for class_name in df['diagnosis'].unique():
    class_dir = f'{parent_dir}/per_class_eval/{class_name}'
    os.makedirs(class_dir, exist_ok=True)

from evaluate_downstream import plot_metric_bars

# plot the results for each class
for class_name in df['diagnosis'].unique():
    class_dir = f'{parent_dir}/per_class_eval/{class_name}'
    class_df = df[df['diagnosis'] == class_name]
    plot_metric_bars(class_df, class_dir)
