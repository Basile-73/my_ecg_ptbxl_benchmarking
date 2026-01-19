import os
import yaml
import pandas as pd
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# COLOR_MAP = {
#     'unet': '#ff9896',
#     'unet_mamba': '#cc7675',
#     'unet_mamba - unet': '#cc7675',
#     'unet_mamba_bidir': '#995555',
#     'unet_mamba_bidir - unet': '#995555',
#     'imunet': '#98df8a',
#     'imunet_mamba': '#6cbf5c',
#     'imunet_mamba_bidir': '#3d7f3d',
# }

COLOR_MAP = {
    'noisy_input': '#808080',  # Grey (baseline)
    'ant_drnn': '#ffbb78',
    'chiang_dae': '#ff7f0e',
    'fcn': '#aec7e8',         # Light blue (Stage1)
    'drnet_fcn': '#1f77b4',   # Dark blue (Stage2)
    'unet': '#ff9896',
    # 'unet_mamba': '#cc7675',
    # 'unet_mamba - unet': '#cc7675',
    # 'unet_mamba_bidir': '#995555',
    # 'unet_mamba_bidir - unet': '#995555',
    'drnet_unet': '#d62728',  # Dark red (Stage2)
    'imunet': '#98df8a',
    'imunet_mamba': '#6cbf5c',
    'imunet_mamba_bidir': '#3d7f3d',
    'drnet_imunet': '#2ca02c', # Dark green (Stage2)
    'imunet_origin': '#9467bd',    # Purple
    'mecge_phase': '#f6c453',      # warm golden yellow
    'mecge_phase_250': '#e0a800',  # deeper amber
    # 'imunet_mamba_bn': '#ff7f0e',  # Orange
    # 'imunet_mamba_bottleneck': '#1C8AC9',  # Cyan-blue
    # 'imunet_mamba_up': '#17becf',  # Cyan/Teal
    # 'imunet_mamba_early': '#391CC9', # Purple-blue
    # 'imunet_mamba_late': '#bcbd22',  # Yellow-green
    'mamba1_3blocks': '#8ecae6',      # light blue
    'drnet_mamba1_3blocks': '#005f73',# dark blue
    'mamba2_3blocks': '#c77dff',        # light purple
    'drnet_mamba2_3blocks': '#5a189a',  # dark purple
}

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

def plot_results(all_results, keys, save_path=None, extra_df=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = ['RMSE', 'SNR']
    colors = plt.cm.tab10.colors

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Get unique models
        models = all_results[keys[0]].unique()

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

# all_results = summarize_results('P0_curriculum_synthetic', ["model.type", "split_length"])
# print(all_results)
# plot_results(all_results, keys = ["model.type", "split_length"])


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
