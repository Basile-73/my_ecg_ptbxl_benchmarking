"""
Quantify how noise degrades downstream classification performance.

No denoisers are involved. For each combined-noise level (10 points over the
plausible_range from new_code/experiments/noise_study/report_stage1.yaml), noisy
validation signals are classified by 3 pre-trained classifiers and macro-AUC +
macro-F1 are reported with bootstrap 95% CIs.

The figure is 2 stacked subplots (AUC, F1) with combined SNR (dB) on the
x-axis and one line per classifier.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path as _Path
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Register CMU Serif font (mirrors evaluate_downstream.py)
def _find_repo_root():
    candidates = []
    try:
        candidates.append(_Path(__file__).resolve().parent)
    except (NameError, OSError):
        pass
    candidates.append(_Path.cwd())
    for start in candidates:
        p = start
        while p != p.parent:
            if (p / "fonts" / "cm-unicode-0.7.0").is_dir():
                return p
            p = p.parent
    return _Path.cwd()
_REPO_ROOT = _find_repo_root()
_FONT_DIR = _REPO_ROOT / "fonts" / "cm-unicode-0.7.0"
for _ttf in _FONT_DIR.glob("*.ttf"):
    fm.fontManager.addfont(str(_ttf))
plt.rcParams["font.family"] = "CMU Serif"

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '../classification'))
sys.path.insert(0, os.path.join(script_dir, '../../ecg_noise/source'))
sys.path.insert(0, os.path.join(script_dir, '../../'))

from evaluate_downstream import load_config, load_classification_model
from evaluate_downstream_clf_metrics import compute_binary_metrics_ci

from denoising_utils.downstream import compute_bootstrap_ci, calibrate_temperature
from ecg_noise_factory.noise import NoiseFactory
from utils.utils import load_dataset, apply_standardizer, select_data, compute_label_aggregations
from new_code.visualisation.maps import CLASSIFICATION_MODEL_NAMES, plot_font_sizes


# ---------------------------------------------------------------------------
# Noise-level construction (matches new_code/evaluate_noise_study.py)
# ---------------------------------------------------------------------------

# plausible_range from new_code/experiments/noise_study/report_stage1.yaml
PLAUSIBLE_RANGE = {
    'em':   [10, 15],
    'bw':   [0, 5],
    'ma':   [5, 10],
    'AWGN': [20, 25],
}
DEFAULT_STEPS = 10

DEFAULT_CLASSIFIERS = [
    'fastai_resnet1d_wang',
    'fastai_inception1d',
]

PRETTY_CLASSIFIER_NAMES = {
    'fastai_resnet1d_wang': 'FastAI ResNet1D-Wang',
    'fastai_inception1d':   'FastAI Inception1D',
    'fastai_xresnet1d101':  'FastAI XResNet1D-101',
    'fastai_fcn_wang':      'FastAI FCN-Wang',
    'fastai_lstm':          'FastAI LSTM',
    'fastai_lstm_bidir':    'FastAI BiLSTM',
}


def build_combined_levels(plausible_range, steps):
    """Linspace per noise type, then pair index-by-index.

    Returns a list of length ``steps``; each entry is a dict mapping noise
    type to SNR in dB.
    """
    arrs = {
        nt: np.linspace(plausible_range[nt][0], plausible_range[nt][1], steps)
        for nt in ['em', 'bw', 'ma', 'AWGN']
    }
    return [{nt: float(arrs[nt][i]) for nt in arrs} for i in range(steps)]


def combined_snr_db(snr_db_dict):
    """Combined SNR (dB) for independent noise sources.

    SNR_combined = 1 / sum(1 / SNR_linear_i). Returns None if all zero.
    """
    inv_sum = 0.0
    for v in snr_db_dict.values():
        if v is None:
            continue
        lin = 10 ** (v / 10.0)
        if lin > 0:
            inv_sum += 1.0 / lin
    if inv_sum == 0:
        return None
    return float(10.0 * np.log10(1.0 / inv_sum))


# ---------------------------------------------------------------------------
# Per-condition evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(X_scaled, y_val, classifier_models, n_bootstraps):
    """Run each classifier on X_scaled and compute AUC + F1 with CIs.

    Returns a dict: classifier_name -> metrics dict.
    """
    results = {}
    for clf_name, clf_model in classifier_models.items():
        y_pred = clf_model.predict(X_scaled)
        T = calibrate_temperature(y_pred, y_val)

        auc_point = roc_auc_score(y_val, y_pred, average='macro')
        auc_ci = compute_bootstrap_ci(y_val, y_pred, n_bootstraps=n_bootstraps)
        f1_ci = compute_binary_metrics_ci(
            y_val, y_pred, temperature=T, threshold=0.5, n_bootstraps=n_bootstraps
        )

        results[clf_name] = {
            'auc':       auc_point,
            'auc_lower': auc_ci['lower'],
            'auc_upper': auc_ci['upper'],
            'f1':        f1_ci['f1'],
            'f1_lower':  f1_ci['f1_lower'],
            'f1_upper':  f1_ci['f1_upper'],
            'temperature': T,
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_noise_degradation(
    config_path,
    base_exp='exp0',
    classification_sampling_rate=500,
    classifier_names=None,
    steps=DEFAULT_STEPS,
    n_bootstraps=None,
    output_subdir='noise_degradation',
    noise_seed=42,
):
    if classifier_names is None:
        classifier_names = DEFAULT_CLASSIFIERS

    config = load_config(config_path)
    if n_bootstraps is None:
        n_bootstraps = config['evaluation']['bootstrap_samples']

    denoising_exp_folder = os.path.join(config['outputfolder'], config['experiment_name'])
    results_folder = os.path.join(denoising_exp_folder, output_subdir, base_exp)
    os.makedirs(results_folder, exist_ok=True)

    print("=" * 80)
    print("NOISE DEGRADATION EVALUATION (no denoisers)")
    print("=" * 80)
    print(f"Base experiment: {base_exp}")
    print(f"Classifiers:     {classifier_names}")
    print(f"Sweep steps:     {steps}")
    print(f"Bootstraps:      {n_bootstraps}")
    print(f"Results folder:  {results_folder}")

    # -------- Locate classification experiment folder -----------------------
    base_exp_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        config['classification_outputfolder'], base_exp,
    )
    if not os.path.exists(base_exp_path):
        base_exp_path = os.path.join('../../output', base_exp)
    if not os.path.exists(base_exp_path):
        raise FileNotFoundError(f"Base experiment not found: {base_exp_path}")

    # -------- Load PTB-XL data ---------------------------------------------
    datafolder = config['datafolder']
    val_fold = config['val_fold']
    test_fold = config['test_fold']

    data, raw_labels = load_dataset(datafolder, classification_sampling_rate)

    experiments = {
        'exp0': 'all', 'exp1': 'diagnostic', 'exp1.1': 'subdiagnostic',
        'exp1.1.1': 'superdiagnostic', 'exp2': 'form', 'exp3': 'rhythm',
    }
    task = experiments[base_exp]
    labels = compute_label_aggregations(raw_labels, datafolder, task)
    pickle_folder = os.path.join(base_exp_path, 'data')
    data, labels, _, _ = select_data(data, labels, task, 0, pickle_folder)

    X_val_raw = data[labels.strat_fold == val_fold]
    y_val = np.load(os.path.join(base_exp_path, 'data', 'y_val.npy'), allow_pickle=True)
    n_classes = y_val.shape[1]
    print(f"Validation samples: {len(X_val_raw)}  |  classes: {n_classes}")

    with open(os.path.join(base_exp_path, 'data', 'standard_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # -------- Load classifiers ---------------------------------------------
    classifier_models = {}
    for clf_name in classifier_names:
        try:
            clf_models = load_classification_model(
                clf_name, base_exp_path, n_classes,
                X_val_raw[0].shape, classification_sampling_rate,
            )
            classifier_models[clf_name] = clf_models
            print(f"  loaded: {clf_name}")
        except Exception as e:
            print(f"  FAILED to load {clf_name}: {e}")
    if not classifier_models:
        raise RuntimeError("No classifiers loaded.")

    # -------- Noise factory -------------------------------------------------
    noise_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        config['noise_data_path'],
    )
    noise_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        config['noise_config_path'],
    )
    factory = NoiseFactory(
        data_path=noise_data_path,
        sampling_rate=classification_sampling_rate,
        config_path=noise_config_path,
        mode='eval',
        seed=noise_seed,
    )
    # Ensure all four noise types are active; we override SNRs per level.
    for nt in ['em', 'bw', 'ma', 'AWGN']:
        if nt not in factory.noise_types:
            factory.noise_types.append(nt)
    factory.config.setdefault('SNR', {})

    # -------- Clean baseline ------------------------------------------------
    rows = []
    print("\n--- clean baseline ---")
    X_clean_scaled = apply_standardizer(X_val_raw.copy(), scaler)
    clean_metrics = evaluate_condition(X_clean_scaled, y_val, classifier_models, n_bootstraps)
    for clf_name, m in clean_metrics.items():
        rows.append({
            'step': -1, 'snr_combined_db': np.inf, 'condition': 'clean',
            'snr_em': np.inf, 'snr_bw': np.inf, 'snr_ma': np.inf, 'snr_AWGN': np.inf,
            'classification_model': clf_name, **m,
        })
        print(f"  {clf_name}: AUC={m['auc']:.4f} [{m['auc_lower']:.4f}, {m['auc_upper']:.4f}]  "
              f"F1={m['f1']:.4f} [{m['f1_lower']:.4f}, {m['f1_upper']:.4f}]")

    # -------- Noise sweep ---------------------------------------------------
    levels = build_combined_levels(PLAUSIBLE_RANGE, steps)
    print(f"\n--- sweep: {steps} combined levels ---")
    for i, snr_dict in enumerate(tqdm(levels, desc="noise levels")):
        # Override SNRs in place
        for nt, v in snr_dict.items():
            factory.config['SNR'][nt] = v
        # Reseed per-level so noise realization is deterministic and identical
        # across classifiers (the three classifiers see the same noisy signals).
        factory.rng = np.random.default_rng(noise_seed + i)

        snr_comb = combined_snr_db(snr_dict)
        X_noisy = factory.add_noise(
            x=X_val_raw.copy(), batch_axis=0, channel_axis=2, length_axis=1,
        )
        X_noisy_scaled = apply_standardizer(X_noisy, scaler)
        metrics = evaluate_condition(X_noisy_scaled, y_val, classifier_models, n_bootstraps)

        for clf_name, m in metrics.items():
            rows.append({
                'step': i,
                'snr_combined_db': snr_comb,
                'condition': f'step_{i}',
                'snr_em':   snr_dict['em'],
                'snr_bw':   snr_dict['bw'],
                'snr_ma':   snr_dict['ma'],
                'snr_AWGN': snr_dict['AWGN'],
                'classification_model': clf_name,
                **m,
            })
        print(f"  step {i:2d}  combined={snr_comb:+.2f} dB  "
              + "  ".join(f"{n.split('_')[-1]}:AUC={metrics[n]['auc']:.3f},F1={metrics[n]['f1']:.3f}"
                          for n in classifier_names if n in metrics))

    # -------- Save CSV ------------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_folder, 'noise_degradation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved to: {csv_path}")

    # -------- Plot ----------------------------------------------------------
    plot_noise_degradation(df, results_folder)

    print(f"\n✓ Done. Results in {results_folder}")
    return df


def plot_noise_degradation(df, output_folder):
    """Two stacked subplots: AUC (top), F1 (bottom), x = combined SNR (dB).

    X-axis is inverted so low-noise (high SNR) is on the left and high-noise
    (low SNR) is on the right. Legend is placed in a box below the lower plot.
    """
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "CMU Serif"

    sweep = df[df['condition'] != 'clean'].copy()
    clean = df[df['condition'] == 'clean'].copy()
    sweep = sweep.sort_values('snr_combined_db')

    classifiers = sorted(sweep['classification_model'].unique())
    # Modern non-default palette (teal + vermillion), cycled if more classifiers
    custom_palette = ['#2a9d8f', '#e63946', '#264653', '#f4a261', '#6a4c93']
    palette = [custom_palette[i % len(custom_palette)] for i in range(len(classifiers))]
    color_map = dict(zip(classifiers, palette))

    # Font sizes scaled +30% then +20% (total 1.56x)
    fs = {k: v * 1.56 for k, v in plot_font_sizes.items()}
    # Legend should not be smaller than other labels
    fs['legend'] = fs['axis_labels']

    # Original width 9 → 20% narrower = 7.2; extra height reserved for legend
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 10.0), sharex=True)

    for metric, ax, label in [
        ('auc', axes[0], 'AUC (macro)'),
        ('f1',  axes[1], 'F1 (macro)'),
    ]:
        for clf_name in classifiers:
            d = sweep[sweep['classification_model'] == clf_name]
            color = color_map[clf_name]
            pretty = PRETTY_CLASSIFIER_NAMES.get(clf_name, clf_name)

            ax.plot(
                d['snr_combined_db'], d[metric],
                marker='o', color=color, label=pretty, linewidth=2, markersize=6,
            )
            ax.fill_between(
                d['snr_combined_db'], d[f'{metric}_lower'], d[f'{metric}_upper'],
                color=color, alpha=0.20, linewidth=0,
            )
            clean_row = clean[clean['classification_model'] == clf_name]
            if len(clean_row):
                ax.axhline(
                    clean_row[metric].values[0],
                    color=color, linestyle='--', linewidth=1.2, alpha=0.7,
                )

        ax.set_ylabel(label, fontsize=fs['axis_labels'])
        ax.tick_params(axis='both', labelsize=fs['ticks'])
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel('Combined SNR (dB)', fontsize=fs['axis_labels'])
    # Tighten x-axis so first/last points sit flush with the subplot edges
    x_min = sweep['snr_combined_db'].min()
    x_max = sweep['snr_combined_db'].max()
    axes[1].set_xlim(x_min, x_max)
    # Flip so low noise (high SNR) → high noise (low SNR) left-to-right
    axes[1].invert_xaxis()

    axes[0].set_title(
        'Classification performance vs noise level\n'
        '(dashed = clean baseline)',
        fontsize=fs['title'],
    )

    # Legend as a box below the lower subplot (no title — classifier names speak for themselves)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(labels_),
        fontsize=fs['legend'],
        frameon=True,
        framealpha=0.95,
    )

    # Reserve just enough bottom strip so the legend sits flush under the x-label
    plt.tight_layout(rect=[0, 0.09, 1, 0.97])

    out_path = os.path.join(output_folder, 'noise_degradation.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Plot saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str,
                        default='mycode/denoising/configs/report_default.yaml')
    parser.add_argument('--base-exp', type=str, default='exp1.1.1')
    parser.add_argument('--sampling-rate', type=int, default=100)
    parser.add_argument('--classifiers', nargs='+', default=DEFAULT_CLASSIFIERS)
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    parser.add_argument('--n-bootstraps', type=int, default=None)
    parser.add_argument('--output-subdir', type=str, default='noise_degradation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip evaluation and regenerate the plot from an existing '
                             'noise_degradation_results.csv in the output folder')
    args = parser.parse_args()

    if args.plot_only:
        config = load_config(args.config)
        denoising_exp_folder = os.path.join(config['outputfolder'], config['experiment_name'])
        results_folder = os.path.join(denoising_exp_folder, args.output_subdir, args.base_exp)
        csv_path = os.path.join(results_folder, 'noise_degradation_results.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Results CSV not found: {csv_path}\n"
                f"Run without --plot-only first to generate the data."
            )
        print(f"Plot-only mode: loading results from {csv_path}")
        df = pd.read_csv(csv_path)
        plot_noise_degradation(df, results_folder)
        print(f"\n✓ Plot regenerated in: {results_folder}")
        return

    evaluate_noise_degradation(
        config_path=args.config,
        base_exp=args.base_exp,
        classification_sampling_rate=args.sampling_rate,
        classifier_names=args.classifiers,
        steps=args.steps,
        n_bootstraps=args.n_bootstraps,
        output_subdir=args.output_subdir,
        noise_seed=args.seed,
    )


if __name__ == '__main__':
    main()
