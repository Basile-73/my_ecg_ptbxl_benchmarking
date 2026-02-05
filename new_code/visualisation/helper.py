import os
import yaml
import pandas as pd
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from maps import COLOR_MAP, NAME_MAP, plot_font_sizes

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

def _plot_metric_on_axis(ax, all_results, keys, metric, filtered_models=None, extra_df=None, x_ticks=None, show_xlabel=True, show_title=True, sampling_rate=360):
    """Helper function to plot a single metric on a given axis."""
    colors = plt.cm.tab10.colors

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
        model_label = NAME_MAP.get(model, model)

        # Plot line with markers
        ax.plot(durations, means, marker='o', label=model_label, color=color, linewidth=2, markersize=8)

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
                label=f"{model_label} w/o curriculum"
            )

    if show_xlabel:
        ax.set_xlabel(f'Record Length (s)', fontsize=plot_font_sizes['axis_labels'])
    else:
        ax.set_xlabel('')
    ax.set_ylabel(metric, fontsize=plot_font_sizes['axis_labels'])
    if show_title:
        ax.set_title(f'{metric} vs Record Length', fontsize=plot_font_sizes['title'], fontweight='bold')
    ax.legend(fontsize=plot_font_sizes['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=plot_font_sizes['ticks'])

    # Format x-axis to show integer durations
    ax.set_xscale('log')
    ax.minorticks_off()
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        # Convert to seconds for x-axis labels
        if show_xlabel:
            ax.set_xticklabels([f'{int(x/sampling_rate)}' for x in x_ticks])
        else:
            ax.set_xticklabels([])
    else:
        tick_values = sorted(all_results[keys[1]].unique())
        ax.set_xticks(tick_values)
        if show_xlabel:
            ax.set_xticklabels([f'{int(x/sampling_rate)}' for x in tick_values])
        else:
            ax.set_xticklabels([])


def plot_results(all_results, keys, save_path=None, extra_df=None, filtered_models:list[str]=None, filtered_metrics:list[str]=None):
    metrics = filtered_metrics if filtered_metrics is not None else ['RMSE', 'SNR']

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(7 * num_metrics, 5))

    # Ensure axes is always iterable (handle single subplot case)
    if num_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        _plot_metric_on_axis(axes[idx], all_results, keys, metric, filtered_models, extra_df)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_results_and_differences(all_results, differences, keys, save_path=None, extra_df=None,
                                   filtered_models:list[str]=None, filtered_metrics:list[str]=None, sampling_rate=360, show_title=True):
    """
    Plot results and differences in a 2-row layout with aligned axes.

    Args:
        all_results: DataFrame with original results
        differences: DataFrame with difference results
        keys: List of keys for grouping
        save_path: Optional path to save the figure
        extra_df: Optional extra data to plot
        filtered_models: Optional list of models to include
        filtered_metrics: Optional list of metrics to plot
        sampling_rate: Sampling rate in Hz for converting to seconds (default: 360)
        show_title: Whether to show titles on the top row plots (default: True)
    """
    metrics = filtered_metrics if filtered_metrics is not None else ['RMSE', 'SNR']
    num_metrics = len(metrics)

    # Create 2 rows, metrics on top, differences on bottom (reduced height)
    fig, axes = plt.subplots(2, num_metrics, figsize=(7 * num_metrics, 6))

    # Ensure axes is always 2D
    if num_metrics == 1:
        axes = axes.reshape(2, 1)

    # Get common x_ticks for alignment
    x_ticks = sorted(all_results[keys[1]].unique())

    # Get unique models for filling between them
    models = all_results[keys[0]].unique()
    if filtered_models is not None:
        models = [x for x in models if x in filtered_models]

    # Plot metrics on top row (no x-axis labels/ticks)
    for idx, metric in enumerate(metrics):
        _plot_metric_on_axis(axes[0, idx], all_results, keys, metric, filtered_models, extra_df, x_ticks, show_xlabel=False, show_title=show_title, sampling_rate=sampling_rate)

        # Fill between the two model curves with transparent light green
        if len(models) == 2:
            model1_data = all_results[all_results[keys[0]] == models[0]].sort_values(keys[1])
            model2_data = all_results[all_results[keys[0]] == models[1]].sort_values(keys[1])
            durations = model1_data[keys[1]].values
            means1 = model1_data[(metric, 'mean')].values
            means2 = model2_data[(metric, 'mean')].values
            axes[0, idx].fill_between(durations, means1, means2, alpha=0.2, color='lightgreen', zorder=1)

    # Plot differences on bottom row (with x-axis in seconds)
    for idx, metric in enumerate(metrics):
        ax = axes[1, idx]
        colors = plt.cm.tab10.colors

        # Get difference data
        diff_models = differences[keys[0]].unique()
        for model_idx, model in enumerate(diff_models):
            model_data = differences[differences[keys[0]] == model].sort_values(keys[1])

            durations = model_data[keys[1]].values
            means = model_data[(metric, 'mean')].values
            ci_low = model_data[(metric, 'ci_low')].values
            ci_high = model_data[(metric, 'ci_high')].values

            model_label = NAME_MAP.get(model, model)

            # Plot line with markers in black
            # Use metric name with "Difference" as legend label
            legend_label = f'{metric} Improvement'
            ax.plot(durations, means, marker='o', label=legend_label, color='grey', linewidth=2, markersize=8)

            # Fill below the curve with transparent light green
            ax.fill_between(durations, 0, means, alpha=0.2, color='lightgreen', zorder=1)

        ax.set_xlabel(f'Record Length (s)', fontsize=plot_font_sizes['axis_labels'])
        ax.set_ylabel(metric, fontsize=plot_font_sizes['axis_labels'])
        # No title for difference plots
        ax.legend(fontsize=plot_font_sizes['legend'])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=plot_font_sizes['ticks'])

        # Format x-axis
        ax.set_xscale('log')
        ax.minorticks_off()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{int(x/sampling_rate)}' for x in x_ticks])

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

all_results = summarize_results('../outputs/P0_curriculum_synthetic_extra', ["model.type", "split_length"])
differences = add_difference(all_results, ['unet_mamba_block', 'unet'])
all_results['model.type'] = all_results['model.type'].str.replace('unet_mamba_block', 'mamba1_3blocks', regex=False)
# plot_results(all_results, keys = ["model.type", "split_length"], filtered_models=['unet', 'mamba1_3blocks'], filtered_metrics=['SNR'])
# plot_results(differences, keys = ["model.type", "split_length"], filtered_metrics=['SNR'])

plot_results_and_differences(
    all_results,
    differences,
    keys=["model.type", "split_length"],
    filtered_models=['unet', 'mamba1_3blocks'],
    filtered_metrics=['SNR'],
    save_path='../outputs/AAA_plots/length_extra.png',
    show_title=False
)


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
