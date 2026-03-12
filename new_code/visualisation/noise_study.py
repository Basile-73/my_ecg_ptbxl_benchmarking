import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from pathlib import Path

from maps import COLOR_MAP, NAME_MAP, OUR_MODELS

# Register CMU Serif font
_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "fonts" / "cm-unicode-0.7.0"
for _ttf in _FONT_DIR.glob("*.ttf"):
    fm.fontManager.addfont(str(_ttf))
plt.rcParams["font.family"] = "CMU Serif"


def plot_noise_study(csv_path="outputs/noise_study_results.csv", output_folder='outputs/noise_study',
                     metric='SNR', plausible_range=None):
    """Plot noise study results: side-by-side subplots for each noise_type.

    Args:
        csv_path: Path to the CSV with noise study results.
        output_folder: Directory to save output figures.
        metric: The metric column to plot on the y-axis.
        plausible_range: Optional dict mapping noise_type -> [lo, hi] in dB.
            When provided, regions outside the plausible range are shaded grey.
            For 'combined', the plausible range is derived from the per-type ranges
            using the independence formula (SNR_comb = 1/sum(1/SNR_i)).
            If None, no shading is applied (backward compatible).
    """
    # Configure font sizes
    plt.rcParams.update({
        'font.size': 29.16,        # default text size
        'axes.titlesize': 34.02,   # title size
        'axes.labelsize': 31.59,   # x and y labels
        'xtick.labelsize': 26.73,  # x tick labels
        'ytick.labelsize': 26.73,  # y tick labels
        'legend.fontsize': 29.16,  # legend
    })

    df = pd.read_csv(csv_path)
    data = df[df['metric'] == metric]

    # Determine x-axis column: prefer snr_value if present and non-null
    use_snr_value = 'snr_value' in data.columns and data['snr_value'].notna().any()
    x_col = 'snr_value' if use_snr_value else 'noise_level'
    x_label = 'Input SNR (dB)' if use_snr_value else 'Noise intensity'
    y_label = 'Output SNR (dB)' if metric == 'SNR' else metric

    noise_types = data['noise_type'].unique()

    # ------------------------------------------------------------------ helpers

    def _combined_snr_db(per_type_range):
        """Compute combined SNR (dB) from per-type SNR values assuming independence."""
        import numpy as np
        # 10^(v/10) is never 0 for finite v, so only guard against None
        inv_sum = sum(1.0 / (10 ** (v / 10.0)) for v in per_type_range if v is not None)
        if inv_sum == 0:
            return None
        return 10.0 * np.log10(1.0 / inv_sum)

    def _get_plausible_snr_range(noise_type):
        """Return (lo, hi) plausible SNR range for a given noise_type, or None."""
        if plausible_range is None:
            return None
        if noise_type == 'combined':
            lo = _combined_snr_db([plausible_range[nt][0] for nt in ['em', 'bw', 'ma', 'AWGN']])
            hi = _combined_snr_db([plausible_range[nt][1] for nt in ['em', 'bw', 'ma', 'AWGN']])
            if lo is None or hi is None:
                return None
            return (min(lo, hi), max(lo, hi))
        if noise_type in plausible_range:
            return (plausible_range[noise_type][0], plausible_range[noise_type][1])
        return None

    def _shade_implausible(ax, x_vals, plaus_lo, plaus_hi):
        """Shade x regions outside [plaus_lo, plaus_hi] with a dashed grey background."""
        x_min, x_max = min(x_vals), max(x_vals)
        shade_kw = dict(color='lightgrey', alpha=0.5, zorder=0)
        # left region (below plausible low)
        if x_min < plaus_lo:
            ax.axvspan(x_min, plaus_lo, **shade_kw)
            ax.axvline(plaus_lo, color='grey', linestyle='--', linewidth=1, zorder=1)
        # right region (above plausible high)
        if x_max > plaus_hi:
            ax.axvspan(plaus_hi, x_max, **shade_kw)
            ax.axvline(plaus_hi, color='grey', linestyle='--', linewidth=1, zorder=1)

    # --------------------------------------------------------- individual plots

    for noise_type in noise_types:
        # Set font sizes for individual plots
        plt.rcParams.update({
            'font.size': 32.08,
            'axes.titlesize': 37.42,
            'axes.labelsize': 34.75,
            'xtick.labelsize': 29.4,
            'ytick.labelsize': 29.4,
            'legend.fontsize': 32.08,
        })

        fig, ax = plt.subplots(figsize=(8, 6))
        noise_data = data[data['noise_type'] == noise_type]
        x_vals = sorted(noise_data[x_col].unique())

        # Plausible range shading
        plaus = _get_plausible_snr_range(noise_type)
        if use_snr_value and plaus is not None:
            _shade_implausible(ax, x_vals, plaus[0], plaus[1])

        for model in noise_data['model_name'].unique():
            model_data = noise_data[noise_data['model_name'] == model].sort_values(x_col)
            display_name = NAME_MAP.get(model, model)
            if model in OUR_MODELS:
                display_name += ' (ours)'
            color = COLOR_MAP.get(model, None)
            ax.plot(model_data[x_col], model_data['mean'],
                    marker='o', label=display_name, linewidth=2, color=color)
            ax.fill_between(model_data[x_col],
                            model_data['ci_low'], model_data['ci_high'], alpha=0.2, color=color)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # "E." prefix for the combined noise type individual plot
        if noise_type == 'combined':
            ax.set_title(f'E. {noise_type.upper()} Noise')
        else:
            ax.set_title(f'{noise_type.upper()} Noise')

        ax.invert_xaxis()  # high SNR (clean) on left, low SNR (noisy) on right
        if not use_snr_value:
            ax.set_xticks([min(x_vals), max(x_vals)])
            ax.set_xticklabels(['low', 'high'])  # low noise on left, high noise on right

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fig.savefig(f'{output_folder}/{noise_type}_noise_study.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig)

        # Reset font sizes for combined plot
        plt.rcParams.update({
            'font.size': 29.16,
            'axes.titlesize': 34.02,
            'axes.labelsize': 31.59,
            'xtick.labelsize': 26.73,
            'ytick.labelsize': 26.73,
            'legend.fontsize': 29.16,
        })

    # ------------------------------------------------------------ combined plot
    # Exclude 'combined' noise type from the combined subplot
    plot_noise_types = [nt for nt in noise_types if nt != 'combined']

    fig, axes = plt.subplots(1, len(plot_noise_types), figsize=(5 * len(plot_noise_types), 6), sharey=True,
                             gridspec_kw={'wspace': 0.08})
    if len(plot_noise_types) == 1:
        axes = [axes]

    _subplot_letters = 'ABCDEFGHIJ'
    for i, noise_type in enumerate(plot_noise_types):
        ax = axes[i]
        noise_data = data[data['noise_type'] == noise_type]
        x_vals = sorted(noise_data[x_col].unique())

        # Plausible range shading
        plaus = _get_plausible_snr_range(noise_type)
        if use_snr_value and plaus is not None:
            _shade_implausible(ax, x_vals, plaus[0], plaus[1])

        for model in noise_data['model_name'].unique():
            model_data = noise_data[noise_data['model_name'] == model].sort_values(x_col)
            display_name = NAME_MAP.get(model, model)
            if model in OUR_MODELS:
                display_name += ' (ours)'
            color = COLOR_MAP.get(model, None)
            ax.plot(model_data[x_col], model_data['mean'],
                    marker='o', label=display_name, linewidth=2, color=color)
            ax.fill_between(model_data[x_col],
                            model_data['ci_low'], model_data['ci_high'], alpha=0.2, color=color)

        ax.set_xlabel(x_label)
        if i == 0:
            ax.set_ylabel(y_label)
        else:
            ax.set_ylabel('')
        letter = _subplot_letters[i]
        ax.set_title(f'{letter}. {noise_type.upper()} Noise')

        ax.invert_xaxis()  # high SNR (clean) on left, low SNR (noisy) on right
        if not use_snr_value:
            ax.set_xticks([min(x_vals), max(x_vals)])
            ax.set_xticklabels(['low', 'high'])  # low noise on left, high noise on right


        ax.grid(True, alpha=0.3)

    # Grab handles/labels for the standalone legend
    handles, labels = axes[0].get_legend_handles_labels()

    fig.savefig(f'{output_folder}/noise_study_results_combined.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

    # --------------------------------------------------------- standalone legend
    fig_leg = plt.figure(figsize=(5 * len(plot_noise_types), 1))
    fig_leg.legend(handles, labels, loc='center', ncol=len(labels),
                   frameon=True, fancybox=True, shadow=True)
    fig_leg.savefig(f'{output_folder}/legend.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig_leg)


if __name__ == "__main__":
    import yaml

    results_file = 'outputs/noise_study_results_non_smooth_experiment_stage2.csv'
    output_folder = 'outputs/noise_study/noise_study_results_non_smooth_experiment_stage2'

    # Optionally load plausible_range from the study config (backward compat: no-op if key absent)
    config_name = 'non_smooth_experiment'
    _cfg = yaml.safe_load(open(f'experiments/noise_study/{config_name}.yaml'))
    _plausible_range = _cfg['noise'].get('plausible_range', None)

    plot_noise_study(results_file, output_folder, plausible_range=_plausible_range)
