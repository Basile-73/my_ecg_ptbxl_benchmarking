import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
    # Configure font sizes (increased by factor of 2)
    plt.rcParams.update({
        'font.size': 24,           # default text size
        'axes.titlesize': 28,      # title size
        'axes.labelsize': 26,      # x and y labels
        'xtick.labelsize': 22,     # x tick labels
        'ytick.labelsize': 22,     # y tick labels
        'legend.fontsize': 24,     # legend
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
        # Set reduced font sizes for individual plots
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 13,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
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
            ax.plot(model_data[x_col], model_data['mean'],
                    marker='o', label=model, linewidth=2)
            ax.fill_between(model_data[x_col],
                            model_data['ci_low'], model_data['ci_high'], alpha=0.2)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{noise_type.upper()} Noise')

        ax.invert_xaxis()  # high SNR (clean) on left, low SNR (noisy) on right
        if not use_snr_value:
            ax.set_xticks([min(x_vals), max(x_vals)])
            ax.set_xticklabels(['low', 'high'])  # low noise on left, high noise on right

        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        fig.savefig(f'{output_folder}/{noise_type}_noise_study.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Reset font sizes to original for combined plot
        plt.rcParams.update({
            'font.size': 24,
            'axes.titlesize': 28,
            'axes.labelsize': 26,
            'xtick.labelsize': 22,
            'ytick.labelsize': 22,
            'legend.fontsize': 24,
        })

    # ------------------------------------------------------------ combined plot

    fig, axes = plt.subplots(1, len(noise_types), figsize=(5 * len(noise_types), 6), sharey=True)
    if len(noise_types) == 1:
        axes = [axes]

    handles, labels = None, None

    for i, noise_type in enumerate(noise_types):
        ax = axes[i]
        noise_data = data[data['noise_type'] == noise_type]
        x_vals = sorted(noise_data[x_col].unique())

        # Plausible range shading
        plaus = _get_plausible_snr_range(noise_type)
        if use_snr_value and plaus is not None:
            _shade_implausible(ax, x_vals, plaus[0], plaus[1])

        for model in noise_data['model_name'].unique():
            model_data = noise_data[noise_data['model_name'] == model].sort_values(x_col)
            ax.plot(model_data[x_col], model_data['mean'],
                    marker='o', label=model, linewidth=2)
            ax.fill_between(model_data[x_col],
                            model_data['ci_low'], model_data['ci_high'], alpha=0.2)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{noise_type.upper()} Noise')

        ax.invert_xaxis()  # high SNR (clean) on left, low SNR (noisy) on right
        if not use_snr_value:
            ax.set_xticks([min(x_vals), max(x_vals)])
            ax.set_xticklabels(['low', 'high'])  # low noise on left, high noise on right

        ax.grid(True, alpha=0.3)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    # Add centered legend box at bottom
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.0),
               frameon=True, fancybox=True, shadow=True, ncol=len(labels))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)

    fig.savefig(f'{output_folder}/noise_study_results_combined.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import yaml

    results_file = 'outputs/noise_study_results_high_range2.csv'
    output_folder = 'outputs/noise_study/high_range2'

    # Optionally load plausible_range from the study config (backward compat: no-op if key absent)
    config_name = 'high_range2'
    _cfg = yaml.safe_load(open(f'experiments/noise_study/{config_name}.yaml'))
    _plausible_range = _cfg['noise'].get('plausible_range', None)

    plot_noise_study(results_file, output_folder, plausible_range=_plausible_range)
