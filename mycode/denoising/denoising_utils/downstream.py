from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar


def calibrate_temperature(logits, labels):
    """
    Learn optimal temperature T for temperature scaling (Guo et al., 2017).

    Finds T that minimizes BCE on the given logits/labels, so that
    sigmoid(logits / T) produces well-calibrated probabilities.

    Args:
        logits: Raw model logits, shape (n_samples, n_classes)
        labels: Binary ground truth labels, shape (n_samples, n_classes)

    Returns:
        Optimal temperature scalar (float)
    """
    logits_tensor = torch.FloatTensor(logits)
    labels_tensor = torch.FloatTensor(labels)

    def nll(T):
        scaled = logits_tensor / T
        return nn.BCEWithLogitsLoss()(scaled, labels_tensor).item()

    result = minimize_scalar(nll, bounds=(0.1, 20.0), method='bounded')
    print(f"  Temperature scaling: T = {result.x:.4f} (BCE {nll(1.0):.4f} -> {result.fun:.4f})")
    return result.x


def compute_bootstrap_ci(y_true, y_pred, n_bootstraps=100, confidence_level=0.95, metric='auc', temperature=1.0):
    """
    Compute bootstrap confidence intervals for AUC or BCE.

    Args:
        y_true: True labels (binary multi-label format)
        y_pred: Raw model logits
        n_bootstraps: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        metric: 'auc', 'bce', or 'brier' to specify which metric to compute
        temperature: Temperature for scaling logits before BCE/Brier (default 1.0 = no scaling)

    Returns:
        Dictionary with mean, lower, and upper bounds
    """
    scores = []
    n_samples = len(y_true)

    np.random.seed(42)

    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # For AUC: check if we have at least one positive sample per class
        if metric == 'auc' and y_true_boot.sum(axis=0).min() == 0:
            continue

        try:
            if metric == 'auc':
                score = roc_auc_score(y_true_boot, y_pred_boot, average='macro')
            elif metric == 'bce':
                bce_loss = nn.BCEWithLogitsLoss()
                y_true_tensor = torch.FloatTensor(y_true_boot)
                y_pred_tensor = torch.FloatTensor(y_pred_boot) / temperature
                score = bce_loss(y_pred_tensor, y_true_tensor).item()
            elif metric == 'brier':
                probs = torch.sigmoid(torch.FloatTensor(y_pred_boot) / temperature).numpy()
                score = np.mean((probs - y_true_boot) ** 2)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            scores.append(score)
        except:
            continue

    if len(scores) == 0:
        return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}

    scores = np.array(scores)
    alpha = 1 - confidence_level

    return {
        'mean': np.mean(scores),
        'lower': np.percentile(scores, 100 * alpha / 2),
        'upper': np.percentile(scores, 100 * (1 - alpha / 2))
    }


def roc_by_class(y_val:np.ndarray, y_pred:np.ndarray, mlb:MultiLabelBinarizer, n_bootstraps=1000, classifyer_name = None, densoising_model_name = None) -> dict:
    assert y_val.shape == y_pred.shape, "Shapes of y_val and y_pred must match"
    class_names = mlb.classes_
    assert len(class_names) == y_val.shape[1], "Number of classes must match the second dimension of y_val"
    n_classes = y_val.shape[1]

    scores = roc_auc_score(y_val, y_pred, average=None)
    point_dict = {class_names[i]: scores[i] for i in range(n_classes)}

    ci = {}
    for i, class_name in enumerate(class_names):
        y_val_class = y_val[:, i]
        y_pred_class = y_pred[:, i]
        ci[class_name] = compute_bootstrap_ci(y_val_class, y_pred_class, n_bootstraps=n_bootstraps, confidence_level=0.95, metric='auc')

    # convert to dataframe
    ci_df = pd.DataFrame.from_dict(ci, orient='index', columns=['mean', 'lower', 'upper'])
    # add point estimate
    ci_df['roc_auc'] = pd.Series(point_dict)
    if classifyer_name is not None:
        ci_df['classifier'] = classifyer_name
    if densoising_model_name is not None:
        ci_df['denoising_model'] = densoising_model_name

    # Convert to flat list of dicts with 'diagnosis' column
    ci_df = ci_df.reset_index()
    ci_df = ci_df.rename(columns={'index': 'diagnosis'})
    return ci_df.to_dict(orient='records')


def plot_reliability_diagram(logits_dict, y_true, temperature, output_folder, clf_name, n_bins=15):
    """
    Plot reliability diagrams comparing calibration before/after temperature scaling.

    Shows one row per condition (clean, noisy, denoised models) with
    before-T (left) and after-T (right) columns.

    Args:
        logits_dict: Dict mapping condition name -> raw logits array (n_samples, n_classes)
        y_true: Binary ground truth labels (n_samples, n_classes)
        temperature: Temperature scalar learned on clean data
        output_folder: Path to save the plot
        clf_name: Classifier name (for title and filename)
        n_bins: Number of bins for the reliability diagram
    """
    n_conditions = len(logits_dict)
    fig, axes = plt.subplots(n_conditions, 2, figsize=(10, 3 * n_conditions), squeeze=False)

    for row, (name, logits) in enumerate(logits_dict.items()):
        for col, (label, T) in enumerate([('Before calibration (T=1)', 1.0),
                                           (f'After calibration (T={temperature:.2f})', temperature)]):
            ax = axes[row, col]

            # Convert logits to probabilities
            probs = torch.sigmoid(torch.FloatTensor(logits) / T).numpy()

            # Flatten across all classes for the reliability diagram
            y_flat = y_true.flatten()
            p_flat = probs.flatten()

            # Compute calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_flat, p_flat, n_bins=n_bins, strategy='uniform'
            )

            # Plot
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
            ax.plot(mean_predicted_value, fraction_of_positives, 's-', markersize=4, label=name)
            ax.fill_between(mean_predicted_value, fraction_of_positives, mean_predicted_value,
                          alpha=0.15, color='red')

            # Expected Calibration Error
            bin_counts = np.histogram(p_flat, bins=n_bins, range=(0, 1))[0]
            ece = np.sum(np.abs(fraction_of_positives - mean_predicted_value) *
                        bin_counts[bin_counts > 0] / len(p_flat))

            ax.set_xlabel('Mean predicted probability')
            ax.set_ylabel('Fraction of positives')
            ax.set_title(f'{name} - {label}\nECE = {ece:.4f}')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'Reliability Diagram - {clf_name}', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    safe_clf_name = clf_name.replace('/', '_').replace('\\', '_')
    plot_path = os.path.join(output_folder, f'reliability_diagram_{safe_clf_name}.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Reliability diagram saved to: {plot_path}")
