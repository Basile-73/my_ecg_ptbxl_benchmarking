import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent.parent))
from visualisation.maps import COLOR_MAP, NAME_MAP, plot_font_sizes

OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
SAVE_DIR = OUTPUT_DIR / 'data_volume_visualisation'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    512: OUTPUT_DIR / 'sanity_check_512' / 'european_st_t',
    1024: OUTPUT_DIR / 'sanity_check_1024' / 'european_st_t',
    2048: OUTPUT_DIR / 'sanity_check_2048' / 'european_st_t',
}


def load_results():
    all_results = pd.DataFrame()
    for n_samples, folder in EXPERIMENTS.items():
        for subfolder in sorted(folder.iterdir()):
            if not subfolder.is_dir():
                continue
            results_file = subfolder / 'results.csv'
            if not results_file.exists():
                continue
            # subfolder name is like "14400_modelname"
            model = '_'.join(subfolder.name.split('_')[1:])
            results = pd.read_csv(results_file, index_col=0)
            results['model'] = model
            results['n_samples'] = n_samples
            all_results = pd.concat([all_results, results])
    return all_results


def plot_metric(df, metric, save_path):
    metric_df = df[df['metric'] == metric]

    fig, ax = plt.subplots(figsize=(8, 7.5))
    tab_colors = plt.cm.tab10.colors

    models = sorted(metric_df['model'].unique())
    for i, model in enumerate(models):
        model_data = metric_df[metric_df['model'] == model].sort_values('n_samples')
        color = COLOR_MAP.get(model, tab_colors[i % len(tab_colors)])
        label = NAME_MAP.get(model, model)

        ax.plot(
            model_data['n_samples'], model_data['mean'],
            marker='o', label=label, color=color, linewidth=2, markersize=8,
        )
        ax.fill_between(
            model_data['n_samples'], model_data['ci_low'], model_data['ci_high'],
            alpha=0.2, color=color,
        )

    ax.set_xlabel('Number of Training Samples', fontsize=plot_font_sizes['axis_labels'])
    ax.set_ylabel(metric, fontsize=plot_font_sizes['axis_labels'])
    ax.set_title(f'{metric} vs Data Volume', fontsize=plot_font_sizes['title'], fontweight='bold')
    ax.legend(fontsize=plot_font_sizes['legend'], bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=plot_font_sizes['ticks'])

    ax.set_xscale('log', base=2)
    ax.minorticks_off()
    ax.set_xticks(sorted(metric_df['n_samples'].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.show()


if __name__ == '__main__':
    results = load_results()
    plot_metric(results, 'SNR', SAVE_DIR / 'snr_vs_data_volume.png')
    plot_metric(results, 'RMSE', SAVE_DIR / 'rmse_vs_data_volume.png')
