"""
Evaluate denoising models using sensitivity, specificity and F1 score.

This script mirrors evaluate_downstream.py but computes threshold-based binary
classification metrics instead of AUC/BCE/Brier. Threshold is fixed at 0.5 on
temperature-calibrated probabilities (sigmoid(logits / T)), which is the
Bayes-optimal boundary under equal FP/FN costs when probabilities are calibrated.

Results are saved as a CSV in the same downstream_results/<exp>/ folder as
evaluate_downstream.py output.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Shared pipeline helpers (data loading, denoising, classification) are
# imported from evaluate_downstream to avoid duplication.
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from evaluate_downstream import (
    load_config,
    resample_signal,
    load_classification_model,
    validate_lead_specific_configs,
    denoise_12lead_signal,
)

sys.path.insert(0, os.path.join(script_dir, '../classification'))
sys.path.insert(0, os.path.join(script_dir, '../../ecg_noise/source'))
sys.path.insert(0, os.path.join(script_dir, '../../'))

from denoising_utils.preprocessing import normalize_robust, denormalize_robust, bandpass_filter
from denoising_utils.downstream import calibrate_temperature
from ecg_noise_factory.noise import NoiseFactory
from utils.utils import load_dataset, apply_standardizer, select_data, compute_label_aggregations
from new_code.utils.getters import get_model
from new_code.visualisation.maps import COLOR_MAP, OUR_MODELS, NAME_MAP, EXCLUDE_MODELS


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _binary_metrics(y_true, y_pred_binary):
    """Per-class sensitivity, specificity, F1; then macro-average.

    Args:
        y_true:        (n_samples, n_classes) int array
        y_pred_binary: (n_samples, n_classes) int array

    Returns:
        (sensitivity, specificity, f1) macro-averaged floats
    """
    n_classes = y_true.shape[1]
    sensitivities, specificities, f1s = [], [], []

    for c in range(n_classes):
        tp = np.sum((y_pred_binary[:, c] == 1) & (y_true[:, c] == 1))
        tn = np.sum((y_pred_binary[:, c] == 0) & (y_true[:, c] == 0))
        fp = np.sum((y_pred_binary[:, c] == 1) & (y_true[:, c] == 0))
        fn = np.sum((y_pred_binary[:, c] == 0) & (y_true[:, c] == 1))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

        sensitivities.append(sens)
        specificities.append(spec)
        f1s.append(f1)

    return np.mean(sensitivities), np.mean(specificities), np.mean(f1s)


def _binary_metrics_per_class(y_true, y_pred_binary):
    """Per-class sensitivity, specificity, F1 (no averaging).

    Returns:
        list of dicts, one per class index, with keys:
        class_idx, sensitivity, specificity, f1
    """
    rows = []
    for c in range(y_true.shape[1]):
        tp = np.sum((y_pred_binary[:, c] == 1) & (y_true[:, c] == 1))
        tn = np.sum((y_pred_binary[:, c] == 0) & (y_true[:, c] == 0))
        fp = np.sum((y_pred_binary[:, c] == 1) & (y_true[:, c] == 0))
        fn = np.sum((y_pred_binary[:, c] == 0) & (y_true[:, c] == 1))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        rows.append({'class_idx': c, 'sensitivity': sens, 'specificity': spec, 'f1': f1})
    return rows


def compute_per_class_metrics_ci(y_true, y_pred_logits, temperature, mlb,
                                  denoising_model_name, classifier_name,
                                  threshold=0.5, n_bootstraps=1000):
    """Per-class sensitivity, specificity, F1 with bootstrap CIs.

    Args:
        y_true:               (n_samples, n_classes)
        y_pred_logits:        (n_samples, n_classes)
        temperature:          scalar T
        mlb:                  fitted MultiLabelBinarizer (for class names)
        denoising_model_name: string label for the denoising condition
        classifier_name:      string label for the classifier
        threshold:            decision boundary (default 0.5)
        n_bootstraps:         bootstrap resamples

    Returns:
        list of dicts (one row per class)
    """
    probs = torch.sigmoid(torch.FloatTensor(y_pred_logits) / temperature).numpy()
    y_pred_binary = (probs >= threshold).astype(int)

    class_names = mlb.classes_ if hasattr(mlb, 'classes_') else [str(i) for i in range(y_true.shape[1])]

    point_rows = _binary_metrics_per_class(y_true, y_pred_binary)

    rng = np.random.RandomState(42)
    n = len(y_true)
    # bootstrap storage: list of lists (one inner list per bootstrap)
    bs_per_class = [{'sensitivity': [], 'specificity': [], 'f1': []}
                    for _ in range(y_true.shape[1])]

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        for row in _binary_metrics_per_class(y_true[idx], y_pred_binary[idx]):
            c = row['class_idx']
            bs_per_class[c]['sensitivity'].append(row['sensitivity'])
            bs_per_class[c]['specificity'].append(row['specificity'])
            bs_per_class[c]['f1'].append(row['f1'])

    results = []
    for row in point_rows:
        c = row['class_idx']
        bs = bs_per_class[c]
        results.append({
            'denoising_model':      denoising_model_name,
            'classification_model': classifier_name,
            'class_idx':            c,
            'class_name':           class_names[c],
            'sensitivity':          row['sensitivity'],
            'sensitivity_lower':    np.percentile(bs['sensitivity'], 2.5),
            'sensitivity_upper':    np.percentile(bs['sensitivity'], 97.5),
            'specificity':          row['specificity'],
            'specificity_lower':    np.percentile(bs['specificity'], 2.5),
            'specificity_upper':    np.percentile(bs['specificity'], 97.5),
            'f1':                   row['f1'],
            'f1_lower':             np.percentile(bs['f1'], 2.5),
            'f1_upper':             np.percentile(bs['f1'], 97.5),
        })
    return results


def compute_binary_metrics_ci(y_true, y_pred_logits, temperature, threshold=0.5, n_bootstraps=1000):
    """Point estimates + 95 % bootstrap CIs for sensitivity, specificity, F1.

    Threshold is applied to temperature-scaled probabilities:
        p = sigmoid(logits / T)  >= threshold → positive

    Args:
        y_true:          (n_samples, n_classes) int array
        y_pred_logits:   (n_samples, n_classes) float array (raw logits)
        temperature:     scalar T from calibrate_temperature
        threshold:       decision boundary (default 0.5)
        n_bootstraps:    number of bootstrap resamples

    Returns:
        dict with keys: sensitivity, sensitivity_lower, sensitivity_upper,
                        specificity, specificity_lower, specificity_upper,
                        f1, f1_lower, f1_upper
    """
    probs = torch.sigmoid(torch.FloatTensor(y_pred_logits) / temperature).numpy()
    y_pred_binary = (probs >= threshold).astype(int)

    sens_pt, spec_pt, f1_pt = _binary_metrics(y_true, y_pred_binary)

    rng = np.random.RandomState(42)
    n = len(y_true)
    bs_sens, bs_spec, bs_f1 = [], [], []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        s, sp, f = _binary_metrics(y_true[idx], y_pred_binary[idx])
        bs_sens.append(s)
        bs_spec.append(sp)
        bs_f1.append(f)

    return {
        'sensitivity':       sens_pt,
        'sensitivity_lower': np.percentile(bs_sens, 2.5),
        'sensitivity_upper': np.percentile(bs_sens, 97.5),
        'specificity':       spec_pt,
        'specificity_lower': np.percentile(bs_spec, 2.5),
        'specificity_upper': np.percentile(bs_spec, 97.5),
        'f1':                f1_pt,
        'f1_lower':          np.percentile(bs_f1, 2.5),
        'f1_upper':          np.percentile(bs_f1, 97.5),
    }


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def _fmt_val(v):
    """0.920 → '.920'"""
    return f"{v:.3f}".lstrip('0') or '.000'


def _fmt_unc(u):
    """0.012 → '.01'"""
    return f"{u:.2f}".lstrip('0') or '.00'


def _model_row_order(all_models):
    """Return (our_models, baseline_models, end_models) preserving COLOR_MAP order."""
    our    = [m for m in OUR_MODELS if m in all_models]
    bases  = [m for m in COLOR_MAP  if m in all_models
              and m not in OUR_MODELS and m not in ('clean', 'noisy') and m not in EXCLUDE_MODELS]
    extra  = [m for m in all_models if m not in our and m not in bases
              and m not in ('clean', 'noisy') and m not in EXCLUDE_MODELS]
    ends   = [m for m in ('noisy', 'clean') if m in all_models]
    return our, bases + extra, ends


def _bold_underline(model_vals):
    """Given {model: value} return (best_model, second_model), ignoring clean/noisy."""
    scoring = {m: v for m, v in model_vals.items() if m not in ('clean', 'noisy')}
    ranked  = sorted(scoring, key=lambda m: -scoring[m])
    best   = ranked[0] if len(ranked) > 0 else None
    second = ranked[1] if len(ranked) > 1 else None
    return best, second


def _decorate(cell, model, best, second):
    if model == best:
        return r'\textbf{' + cell + '}'
    if model == second:
        return r'\underline{' + cell + '}'
    return cell


def save_per_class_latex_tables(per_class_df, results_folder, base_exp):
    """One LaTeX table per classifier.

    Row level 1 : metric  (Sensitivity / Specificity / F1)
    Row level 2 : denoising model
    Columns     : class names
    """
    METRICS      = ('sensitivity', 'specificity', 'f1')
    METRIC_LABEL = {'sensitivity': 'Sensitivity', 'specificity': 'Specificity', 'f1': 'F1'}

    for clf_name in per_class_df['classification_model'].unique():
        clf_df      = per_class_df[per_class_df['classification_model'] == clf_name]
        class_names = list(clf_df['class_name'].unique())
        all_models  = [m for m in clf_df['denoising_model'].unique() if m not in EXCLUDE_MODELS]
        our, bases, ends = _model_row_order(all_models)

        n_cols   = 1 + len(class_names)          # model name col + one per class
        col_spec = 'l' + 'c' * len(class_names)

        lines = [
            r'\begin{tabular}{' + col_spec + '}',
            r'\toprule',
            ' & ' + ' & '.join(class_names) + r' \\',
        ]

        for m_idx, metric in enumerate(METRICS):
            lines.append(r'\midrule')
            lines.append(
                r'\multicolumn{' + str(n_cols) + r'}{l}{\textit{' + METRIC_LABEL[metric] + r'}} \\'
            )

            # pre-compute best/second per class within this metric
            col_best, col_second = {}, {}
            for cn in class_names:
                model_vals = {}
                for model in all_models:
                    row = clf_df[(clf_df['denoising_model'] == model) &
                                 (clf_df['class_name'] == cn)]
                    if not row.empty:
                        model_vals[model] = row[metric].values[0]
                col_best[cn], col_second[cn] = _bold_underline(model_vals)

            def _section_rows(model_list):
                for model in model_list:
                    display = NAME_MAP.get(model, model)
                    if model in OUR_MODELS:
                        display += ' (ours)'
                    cells = [display]
                    for cn in class_names:
                        row = clf_df[(clf_df['denoising_model'] == model) &
                                     (clf_df['class_name'] == cn)]
                        if row.empty:
                            cells.append('--')
                            continue
                        val = row[metric].values[0]
                        unc = max(val - row[f'{metric}_lower'].values[0],
                                  row[f'{metric}_upper'].values[0] - val)
                        cell = _decorate(
                            f"{_fmt_val(val)} $\\pm$ {_fmt_unc(unc)}",
                            model, col_best[cn], col_second[cn]
                        )
                        cells.append(cell)
                    lines.append(' & '.join(cells) + r' \\')

            lines.append(r'\midrule')
            _section_rows(our)
            if our and (bases or ends):
                lines.append(r'\midrule')
            _section_rows(bases)
            if bases and ends:
                lines.append(r'\midrule')
            _section_rows(ends)

        lines += [r'\bottomrule', r'\end{tabular}']

        safe_clf = clf_name.replace('/', '_')
        out_path = os.path.join(results_folder,
                                f'latex_per_class_{safe_clf}_{base_exp}.tex')
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f"✓ Per-class LaTeX table saved to: {out_path}")


def save_macro_latex_tables(results_df, results_folder, base_exp):
    """One LaTeX table per classifier.

    Rows    : denoising models  (OUR_MODELS / baselines / clean+noisy)
    Columns : Sensitivity | Specificity | F1
    """
    METRICS      = ('sensitivity', 'specificity', 'f1')
    METRIC_LABEL = {'sensitivity': 'Sensitivity', 'specificity': 'Specificity', 'f1': 'F1'}

    for clf_name in results_df['classification_model'].unique():
        clf_df     = results_df[results_df['classification_model'] == clf_name]
        all_models = [m for m in clf_df['denoising_model'].unique() if m not in EXCLUDE_MODELS]
        our, bases, ends = _model_row_order(all_models)

        # best/second per metric column (excluding clean/noisy)
        col_best, col_second = {}, {}
        for metric in METRICS:
            model_vals = {
                row['denoising_model']: row[metric]
                for _, row in clf_df.iterrows()
                if row['denoising_model'] not in EXCLUDE_MODELS
            }
            col_best[metric], col_second[metric] = _bold_underline(model_vals)

        header_cells = [''] + [r'\textbf{' + METRIC_LABEL[m] + '}' for m in METRICS]
        lines = [
            r'\begin{tabular}{lccc}',
            r'\toprule',
            ' & '.join(header_cells) + r' \\',
            r'\midrule',
        ]

        def _section_rows(model_list):
            for model in model_list:
                row = clf_df[clf_df['denoising_model'] == model]
                if row.empty:
                    continue
                display = NAME_MAP.get(model, model)
                if model in OUR_MODELS:
                    display += ' (ours)'
                cells = [display]
                for metric in METRICS:
                    val = row[metric].values[0]
                    unc = max(val - row[f'{metric}_lower'].values[0],
                              row[f'{metric}_upper'].values[0] - val)
                    cell = _decorate(
                        f"{_fmt_val(val)} $\\pm$ {_fmt_unc(unc)}",
                        model, col_best[metric], col_second[metric]
                    )
                    cells.append(cell)
                lines.append(' & '.join(cells) + r' \\')

        _section_rows(our)
        if our and (bases or ends):
            lines.append(r'\midrule')
        _section_rows(bases)
        if bases and ends:
            lines.append(r'\midrule')
        _section_rows(ends)

        lines += [r'\bottomrule', r'\end{tabular}']

        safe_clf = clf_name.replace('/', '_')
        out_path = os.path.join(results_folder,
                                f'latex_macro_{safe_clf}_{base_exp}.tex')
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f"✓ Macro LaTeX table saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_clf_metrics(config_path='mycode/denoising/configs/test.yaml', base_exp='exp0',
                         classification_sampling_rate=100, classifier_names=None,
                         threshold=0.5, compute_per_class=False):
    """
    Evaluate denoising models using sensitivity, specificity and F1.

    Mirrors evaluate_downstream() but replaces AUC/BCE/Brier with binary
    classification metrics at a fixed threshold on calibrated probabilities.

    Args:
        config_path:                    Path to denoising YAML config
        base_exp:                       PTB-XL experiment name (e.g. 'exp0')
        classification_sampling_rate:   Sampling rate of classification models (Hz)
        classifier_names:               List of classifier names to evaluate
        threshold:                      Decision threshold applied to calibrated probs
    """
    if classifier_names is None:
        classifier_names = ['fastai_xresnet1d101', 'fastai_inception1d']

    print("\n" + "="*80)
    print("DOWNSTREAM CLF METRICS EVALUATION (sensitivity / specificity / F1)")
    print("="*80)
    print(f"Threshold: {threshold} (applied to temperature-calibrated probabilities)")

    config = load_config(config_path)
    denoising_exp_folder = os.path.join(config['outputfolder'], config['experiment_name'])
    denoising_sampling_rate = config['sampling_frequency']
    n_bootstraps = config['evaluation']['bootstrap_samples']

    results_folder = os.path.join(denoising_exp_folder, 'downstream_results', base_exp)
    os.makedirs(results_folder, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and
                          config['hardware']['use_cuda'] else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Loading 12-lead validation data...")
    print("-"*80)

    datafolder = config['datafolder']
    val_fold   = config['val_fold']
    test_fold  = config['test_fold']

    base_exp_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        config['classification_outputfolder'], base_exp
    )
    if not os.path.exists(base_exp_path):
        base_exp_path = os.path.join('../../output', base_exp)
    if not os.path.exists(base_exp_path):
        raise FileNotFoundError(f"Base experiment not found: {base_exp_path}")

    data, raw_labels = load_dataset(datafolder, classification_sampling_rate)

    experiments = {
        'exp0':   'all',
        'exp1':   'diagnostic',
        'exp1.1': 'subdiagnostic',
        'exp1.1.1': 'superdiagnostic',
        'exp2':   'form',
        'exp3':   'rhythm',
    }
    task = experiments[base_exp]
    labels = compute_label_aggregations(raw_labels, datafolder, task)
    pickle_folder = os.path.join(base_exp_path, 'data')
    data, labels, _, _ = select_data(data, labels, task, 0, pickle_folder)
    print(f"Loaded: {data.shape[0]} samples at {classification_sampling_rate}Hz")

    X_val_12lead          = data[labels.strat_fold == val_fold]
    X_val_12lead_original = X_val_12lead.copy()
    X_train_12lead        = data[~labels.strat_fold.isin([val_fold, test_fold])]
    print(f"Validation samples: {len(X_val_12lead)}")

    y_val     = np.load(os.path.join(pickle_folder, 'y_val.npy'), allow_pickle=True)
    n_classes = y_val.shape[1]
    print(f"Number of classes: {n_classes}")

    # Robust normalization from training statistics
    median = np.median(X_train_12lead)
    iqr    = np.percentile(X_train_12lead, 75) - np.percentile(X_train_12lead, 25)
    X_val_12lead = normalize_robust(X_val_12lead, median, iqr)
    if config.get('bandpass', True):
        X_val_12lead = bandpass_filter(X_val_12lead, fs=classification_sampling_rate)
    X_val_clean = X_val_12lead.copy()

    # -------------------------------------------------------------------------
    # Add noise
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Adding noise...")
    print("-"*80)

    noise_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        config['noise_data_path']
    )
    noise_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        config['noise_config_path']
    )

    noise_factory = NoiseFactory(
        data_path=noise_data_path,
        sampling_rate=classification_sampling_rate,
        config_path=noise_config_path,
        mode='eval'
    )
    X_val_noisy = noise_factory.add_noise(x=X_val_12lead, batch_axis=0, channel_axis=2, length_axis=1)
    print("✓ Noise added")

    # -------------------------------------------------------------------------
    # Load denoising models
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Loading denoising models...")
    print("-"*80)

    denoising_models = {}
    stage1_models_cache = {}
    model_configs = config['models']
    validate_lead_specific_configs(model_configs)

    for model_config in model_configs:
        model_name     = model_config['name']
        model_type     = model_config['type']
        is_lead_specific = model_config.get('lead_specific', False)
        model_path     = model_config['model_path']
        is_stage2      = model_config['is_stage_2']
        input_length   = denoising_sampling_rate * 10

        if is_lead_specific:
            all_files_exist = all(
                os.path.exists(model_path.replace('{lead}', str(i))) for i in range(12)
            )
            if not all_files_exist:
                print(f"⚠️  Skipping {model_name}: one or more lead weights missing")
                continue

            models = []
            for lead_idx in range(12):
                current_path = model_path.replace('{lead}', str(lead_idx))
                m = get_model(model_type, sequence_length=input_length,
                              model_config=model_config, is_stage2=is_stage2)
                m.load_state_dict(torch.load(current_path, map_location=device, weights_only=True))
                m.to(device).eval()
                models.append(m)

            stage1_model = None
            if is_stage2:
                stage1_base_path = model_config['stage_1_weights_path']
                stage1_type      = model_config['stage_1_type']
                all_s1_exist     = all(
                    os.path.exists(stage1_base_path.replace('{lead}', str(i))) for i in range(12)
                )
                if not all_s1_exist:
                    print(f"⚠️  Skipping {model_name}: one or more Stage1 weights missing")
                    continue
                stage1_models = []
                for lead_idx in range(12):
                    s1_path = stage1_base_path.replace('{lead}', str(lead_idx))
                    s1 = get_model(stage1_type, sequence_length=input_length,
                                   model_config=model_config, is_stage2=False)
                    s1.load_state_dict(torch.load(s1_path, map_location=device, weights_only=True))
                    s1.to(device).eval()
                    stage1_models.append(s1)
                stage1_model = stage1_models

            denoising_models[model_name] = {
                'model': models, 'type': model_type,
                'is_stage2': is_stage2, 'stage1_model': stage1_model,
                'lead_specific': True
            }

        else:
            if not os.path.exists(model_path):
                print(f"⚠️  Skipping {model_name}: weights not found")
                continue

            m = get_model(model_type, sequence_length=input_length,
                          model_config=model_config, is_stage2=is_stage2)
            m.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            m.to(device).eval()

            stage1_model = None
            if is_stage2:
                stage1_name = model_config.get('stage_1_type', None)
                stage1_type = stage1_name
                if stage1_name:
                    if stage1_name in stage1_models_cache:
                        stage1_model = stage1_models_cache[stage1_name]
                        print(f"  Using cached Stage1 model: {stage1_name}")
                    else:
                        stage1_model_path = model_config.get('stage_1_weights_path', '')
                        if os.path.exists(stage1_model_path):
                            s1 = get_model(stage1_type, sequence_length=input_length,
                                           model_config=model_config, is_stage2=False)
                            s1.load_state_dict(
                                torch.load(stage1_model_path, map_location=device, weights_only=True)
                            )
                            s1.to(device).eval()
                            stage1_models_cache[stage1_name] = s1
                            stage1_model = s1
                            print(f"  Loaded Stage1 model: {stage1_name} (type: {stage1_type})")
                        else:
                            print(f"  ⚠️  Warning: Stage1 weights not found for {model_name}: {stage1_model_path}")

            denoising_models[model_name] = {
                'model': m, 'type': model_type,
                'is_stage2': is_stage2, 'stage1_model': stage1_model,
                'lead_specific': False
            }

        print(f"✓ Loaded: {model_name}")

    print(f"Total denoising models loaded: {len(denoising_models)}")

    # -------------------------------------------------------------------------
    # Load classification models
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Loading classification models...")
    print("-"*80)

    classification_models = {}
    for clf_name in classifier_names:
        try:
            clf = load_classification_model(
                clf_name, base_exp_path, n_classes,
                X_val_clean[0].shape, classification_sampling_rate
            )
            classification_models[clf_name] = clf
            print(f"✓ Loaded: {clf_name}")
        except Exception as e:
            print(f"⚠️  Failed to load {clf_name}: {e}")

    if not classification_models:
        print("ERROR: No classification models loaded.")
        return

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("EVALUATING")
    print("="*80)

    # Load MultiLabelBinarizer for class names (used by per-class output)
    with open(os.path.join(pickle_folder, 'mlb.pkl'), 'rb') as f:
        mlb = pickle.load(f)

    results = []
    per_class_results = []

    def _run(condition_name, y_pred_logits):
        for clf_name, clf_model in classification_models.items():
            T = calibrate_temperature(y_pred_logits, y_val)
            metrics = compute_binary_metrics_ci(
                y_val, y_pred_logits, T,
                threshold=threshold, n_bootstraps=n_bootstraps
            )
            row = {
                'denoising_model':      condition_name,
                'classification_model': clf_name,
                'temperature':          T,
                **metrics,
            }
            results.append(row)
            print(
                f"  [{condition_name}] {clf_name}  "
                f"sens={metrics['sensitivity']:.4f} "
                f"spec={metrics['specificity']:.4f} "
                f"F1={metrics['f1']:.4f}"
            )

    # --- Baseline: clean ---
    print("\n--- Baseline: Clean ---")
    with open(os.path.join(pickle_folder, 'standard_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    for clf_name, clf_model in classification_models.items():
        X_clean_p = apply_standardizer(X_val_12lead_original, scaler)
        y_pred_clean = clf_model.predict(X_clean_p)
        T = calibrate_temperature(y_pred_clean, y_val)
        metrics = compute_binary_metrics_ci(
            y_val, y_pred_clean, T, threshold=threshold, n_bootstraps=n_bootstraps
        )
        results.append({
            'denoising_model': 'clean',
            'classification_model': clf_name,
            'temperature': T,
            **metrics,
        })
        print(
            f"  [clean] {clf_name}  "
            f"sens={metrics['sensitivity']:.4f} "
            f"spec={metrics['specificity']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )
        if compute_per_class:
            per_class_results.extend(compute_per_class_metrics_ci(
                y_val, y_pred_clean, T, mlb, 'clean', clf_name,
                threshold=threshold, n_bootstraps=n_bootstraps
            ))

    # --- Baseline: noisy ---
    print("\n--- Baseline: Noisy ---")
    with open(os.path.join(base_exp_path, 'data', 'standard_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    for clf_name, clf_model in classification_models.items():
        X_noisy_p = denormalize_robust(X_val_noisy.copy(), median, iqr)
        X_noisy_p = apply_standardizer(X_noisy_p, scaler)
        y_pred_noisy = clf_model.predict(X_noisy_p)
        T = calibrate_temperature(y_pred_noisy, y_val)
        metrics = compute_binary_metrics_ci(
            y_val, y_pred_noisy, T, threshold=threshold, n_bootstraps=n_bootstraps
        )
        results.append({
            'denoising_model': 'noisy',
            'classification_model': clf_name,
            'temperature': T,
            **metrics,
        })
        print(
            f"  [noisy] {clf_name}  "
            f"sens={metrics['sensitivity']:.4f} "
            f"spec={metrics['specificity']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )
        if compute_per_class:
            per_class_results.extend(compute_per_class_metrics_ci(
                y_val, y_pred_noisy, T, mlb, 'noisy', clf_name,
                threshold=threshold, n_bootstraps=n_bootstraps
            ))

    # --- Denoised ---
    print("\n--- Denoised ---")
    with open(os.path.join(base_exp_path, 'data', 'standard_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    for denoise_name, denoise_info in tqdm(denoising_models.items(), desc="Denoising models"):
        print(f"\n{denoise_name}:")
        X_denoised = denoise_12lead_signal(
            X_val_noisy,
            denoise_info['model'],
            device,
            classification_sf=classification_sampling_rate,
            denoising_sf=denoising_sampling_rate,
            batch_size=32,
            stage1_model=denoise_info.get('stage1_model'),
        )
        X_denoised = denormalize_robust(X_denoised, median, iqr)
        X_denoised = apply_standardizer(X_denoised, scaler)

        for clf_name, clf_model in classification_models.items():
            y_pred = clf_model.predict(X_denoised)
            T = calibrate_temperature(y_pred, y_val)
            metrics = compute_binary_metrics_ci(
                y_val, y_pred, T, threshold=threshold, n_bootstraps=n_bootstraps
            )
            results.append({
                'denoising_model':      denoise_name,
                'classification_model': clf_name,
                'temperature':          T,
                **metrics,
            })
            print(
                f"  [{denoise_name}] {clf_name}  "
                f"sens={metrics['sensitivity']:.4f} "
                f"spec={metrics['specificity']:.4f} "
                f"F1={metrics['f1']:.4f}"
            )
            if compute_per_class:
                per_class_results.extend(compute_per_class_metrics_ci(
                    y_val, y_pred, T, mlb, denoise_name, clf_name,
                    threshold=threshold, n_bootstraps=n_bootstraps
                ))

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    csv_path   = os.path.join(results_folder, 'downstream_clf_metrics_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    print(results_df.to_string(index=False))

    save_macro_latex_tables(results_df, results_folder, base_exp)

    if compute_per_class:
        per_class_df   = pd.DataFrame(per_class_results)
        per_class_path = os.path.join(results_folder, f'downstream_clf_metrics_per_class_{base_exp}.csv')
        per_class_df.to_csv(per_class_path, index=False)
        print(f"✓ Per-class results saved to: {per_class_path}")
        save_per_class_latex_tables(per_class_df, results_folder, base_exp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ALL_CLASSIFIERS = [
        'fastai_xresnet1d101',
        'fastai_inception1d',
        'fastai_resnet1d_wang',
        'fastai_lstm',
        'fastai_lstm_bidir',
        'fastai_fcn_wang',
    ]

    parser = argparse.ArgumentParser(
        description='Evaluate denoising models: sensitivity, specificity, F1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_downstream_clf_metrics.py
  python evaluate_downstream_clf_metrics.py --classifiers all
  python evaluate_downstream_clf_metrics.py --base-exp exp1 --classification-fs 500
  python evaluate_downstream_clf_metrics.py --threshold 0.5
        """
    )
    parser.add_argument('--config', type=str, default='mycode/denoising/configs/test.yaml',
                        help='Path to denoising config file')
    parser.add_argument('--base-exp', type=str, default='exp0',
                        help='PTB-XL experiment (exp0, exp1, exp1.1, exp2, exp3)')
    parser.add_argument('--classification-fs', type=int, default=100,
                        help='Sampling frequency of classification models (Hz)')
    parser.add_argument('--classifiers', type=str, nargs='+',
                        default=['fastai_xresnet1d101', 'fastai_inception1d'],
                        help=f'Use "all" or space-separated names. '
                             f'Available: {", ".join(ALL_CLASSIFIERS)}')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold on calibrated probabilities (default: 0.5)')
    parser.add_argument('-per_class', action='store_true',
                        help='Also save per-class metrics to a second CSV')
    args = parser.parse_args()

    if len(args.classifiers) == 1 and args.classifiers[0].lower() == 'all':
        classifier_names = ALL_CLASSIFIERS
    else:
        classifier_names = args.classifiers

    evaluate_clf_metrics(
        config_path=args.config,
        base_exp=args.base_exp,
        classification_sampling_rate=args.classification_fs,
        classifier_names=classifier_names,
        threshold=args.threshold,
        compute_per_class=args.per_class,
    )


if __name__ == '__main__':
    main()
