import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent.parent))
from visualisation.maps import COLOR_MAP, NAME_MAP, plot_font_sizes

_SCRIPT_DIR = Path(__file__).resolve().parent
_FONT_DIR = _SCRIPT_DIR.parent.parent / "fonts" / "cm-unicode-0.7.0"
for _ttf in _FONT_DIR.glob("*.ttf"):
    fm.fontManager.addfont(str(_ttf))
plt.rcParams["font.family"] = "CMU Serif"

font_sizes = {k: v * 1.872 for k, v in plot_font_sizes.items()}

OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
EXPERIMENT_DIR = OUTPUT_DIR / 'reproduce_ptbxl_z_data_experiment'
SAVE_DIR = OUTPUT_DIR / 'folds_visualisation'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = [2, 4, 6, 8]

SUBPLOT_GROUPS = {
    'Mamba-based': lambda m: 'mamba' in m or 'mecge' in m,
    'IMUNet': lambda m: 'imunet' in m,
    'Other': lambda m: 'mamba' not in m and 'mecge' not in m and 'imunet' not in m,
}


def load_results():
    all_results = pd.DataFrame()
    for n_folds in FOLDS:
        folder = EXPERIMENT_DIR / f'{n_folds}_folds'
        if not folder.exists():
            continue
        for subfolder in sorted(folder.iterdir()):
            if not subfolder.is_dir():
                continue
            results_file = subfolder / 'results.csv'
            if not results_file.exists():
                continue
            model = '_'.join(subfolder.name.split('_')[1:])
            results = pd.read_csv(results_file, index_col=0)
            results['model'] = model
            results['n_folds'] = n_folds
            all_results = pd.concat([all_results, results])
    return all_results


def plot_metrics(df, metrics, save_path, show_ci=False):
    n_rows = len(metrics)
    tab_colors = plt.cm.tab10.colors

    n_cols = len(SUBPLOT_GROUPS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows), sharey='row')
    if n_rows == 1:
        axes = [axes]

    for row, metric in enumerate(metrics):
        metric_df = df[df['metric'] == metric]
        for col, (group_name, filter_fn) in enumerate(SUBPLOT_GROUPS.items()):
            ax = axes[row][col]
            models = sorted(m for m in metric_df['model'].unique() if filter_fn(m))
            for i, model in enumerate(models):
                model_data = metric_df[metric_df['model'] == model].sort_values('n_folds')
                color = COLOR_MAP.get(model, tab_colors[i % len(tab_colors)])
                label = NAME_MAP.get(model, model)

                ax.plot(
                    model_data['n_folds'], model_data['mean'],
                    marker='o', label=label, color=color, linewidth=2, markersize=8,
                )
                if show_ci:
                    ax.fill_between(
                        model_data['n_folds'], model_data['ci_low'], model_data['ci_high'],
                        alpha=0.2, color=color,
                    )

            if row == n_rows - 1:
                ax.set_xlabel('Number of Folds', fontsize=font_sizes['axis_labels'])
            if row == 0:
                ax.set_title(group_name, fontsize=font_sizes['title'], fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=font_sizes['ticks'])
            ax.set_xticks(FOLDS)
            ax.set_xlim(FOLDS[0] - 0.5, FOLDS[-1] + 0.5)

        axes[row][0].set_ylabel(metric, fontsize=font_sizes['axis_labels'])

    # One legend per column, below the bottom row
    for col in range(n_cols):
        h, l = axes[0][col].get_legend_handles_labels()
        axes[n_rows - 1][col].legend(
            h, l,
            loc='upper center', ncol=1,
            fontsize=font_sizes['legend'],
            bbox_to_anchor=(0.5, -0.25),
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.show()


if __name__ == '__main__':
    results = load_results()
    plot_metrics(results, ['SNR', 'RMSE'], SAVE_DIR / 'snr_rmse_vs_folds.png', show_ci=True)
