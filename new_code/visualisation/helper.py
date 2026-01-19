import os
import yaml
import pandas as pd
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from maps import COLOR_MAP

def nested_get(d, path):
    for p in path.split("."):
        d = d[p]
    return d


def summarize_results(experiment_name: str, keys: list[str])-> pd.DataFrame:
    folders = os.listdir(experiment_name)
    folders = [f for f in folders if os.path.isdir(os.path.join(experiment_name, f))]
    all_results = pd.DataFrame()

    for folder in folders:
        results = pd.read_csv(os.path.join(experiment_name, folder, "results.csv"), index_col = 0)
        out = results.set_index('metric').stack().to_frame().T # single line, double index

        with open(os.path.join(experiment_name, folder, "config.yaml")) as f:
            config = yaml.safe_load(f)
        for key in keys:
            value = nested_get(config, key)
            out.insert(0, key, value)

        all_results= pd.concat([all_results, out])
    return all_results.sort_values(by=keys)

def plot_results(all_results, keys, save_path=None, extra_df=None, filtered_models:list[str]=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = ['RMSE', 'SNR']
    colors = plt.cm.tab10.colors

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Get unique models
        models = all_results[keys[0]].unique()
        if filtered_models is not None:
            models = [x for x in models if x in filtered_models]

        for model_idx, model in enumerate(models):
            model_data = all_results[all_results[keys[0]] == model].sort_values(keys[1])

            durations = model_data[keys[1]].values
            means = model_data[(metric, 'mean')].values
            ci_low = model_data[(metric, 'ci_low')].values
            ci_high = model_data[(metric, 'ci_high')].values

            color = COLOR_MAP.get(model, colors[model_idx % len(colors)])

            # Plot line with markers
            ax.plot(durations, means, marker='o', label=model, color=color, linewidth=2, markersize=8)

            # Plot confidence interval
            ax.fill_between(durations, ci_low, ci_high, alpha=0.2, color=color)

            if extra_df is not None:
                ed = extra_df[extra_df['model'] == model]
                ax.scatter(
                    ed['record_length'],
                    ed[metric],
                    marker='x',
                    s=80,
                    color=color,
                    zorder=3,
                    label=f"{model} w/o curriculum"
                )

        ax.set_xlabel('Record Length (360Hz)', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} vs Record Length', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis to show integer durations
        ax.set_xscale('log')
        ax.minorticks_off()
        ax.set_xticks(sorted(all_results[keys[1]].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

def add_difference(df, models):
    d1 = df[df['model.type']==models[0]].copy()
    d2 = df[df['model.type']==models[1]]

    num_cols = d1.select_dtypes('number').columns
    num_cols = num_cols.drop(('split_length', ''))
    d1[num_cols] = d1[num_cols] - d2[num_cols].values
    d1['model.type'] = f'{models[0]} - {models[1]}'
    mask = d1.columns.get_level_values(-1).str.contains('ci', case=False, na=False)
    d1.loc[:, mask] = 0
    return d1

all_results = summarize_results('../outputs/P0_curriculum_synthetic', ["model.type", "split_length"])
differences = add_difference(all_results, ['unet_mamba_block', 'unet'])
plot_results(all_results, keys = ["model.type", "split_length"], filtered_models=['unet', 'unet_mamba_block'])
plot_results(differences, keys = ["model.type", "split_length"])


def plot_losses(train_loss_history, test_loss_history, model_name, save_folder):
    epochs = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(12, 7))
    plt.plot(epochs, train_loss_history, label='Training Loss', linewidth=3)
    plt.plot(epochs, test_loss_history, label='Testing Loss', linewidth=3)

    plt.title(f'Loss History for {model_name}', fontsize=28)
    plt.xlabel('Epochs', fontsize=22)
    plt.ylabel('Loss', fontsize=22)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.ylim(0, 0.15)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig(os.path.join(save_folder, 'losses.png'))



# experiment_name = 'length_seed'
# folders = os.listdir(f'../outputs/{experiment_name}/')
# folders = [f for f in folders if os.path.isdir(os.path.join(experiment_name, f))]
# for folder in folders:
#     train_loss = np.load(f'../outputs/{experiment_name}/{folder}/train.npy')
#     test_loss = np.load(f'../outputs/{experiment_name}/{folder}/test.npy')
#     save_folder = Path(f'../outputs/{experiment_name}/{folder}/')
#     if 'mamba' in folder:
#         plot_losses(train_loss, test_loss, model_name='imunet_mamba', save_folder = save_folder)
#     else:
#         plot_losses(train_loss, test_loss, model_name='imunet', save_folder=save_folder)
