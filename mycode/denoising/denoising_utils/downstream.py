from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import numpy as np
import pandas as pd

import sys
from pathlib import Path
import torch
import torch.nn as nn


def compute_bootstrap_ci(y_true, y_pred, n_bootstraps=100, confidence_level=0.95, metric='auc'):
    """
    Compute bootstrap confidence intervals for AUC or BCE.

    Args:
        y_true: True labels (binary multi-label format)
        y_pred: Model predictions. Must be:
                - Probabilities (0-1) for metric='auc'
                - Unnormalized logits for metric='bce' (BCEWithLogitsLoss expects raw model outputs)
        n_bootstraps: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        metric: 'auc' or 'bce' to specify which metric to compute

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
        # For BCE: no filtering needed, all samples are valid
        if metric == 'auc' and y_true_boot.sum(axis=0).min() == 0:
            continue

        try:
            if metric == 'auc':
                score = roc_auc_score(y_true_boot, y_pred_boot, average='macro')
            elif metric == 'bce':
                # BCE expects unnormalized logits (raw model outputs before sigmoid)
                bce_loss = nn.BCEWithLogitsLoss()
                y_true_tensor = torch.FloatTensor(y_true_boot)
                y_pred_tensor = torch.FloatTensor(y_pred_boot)
                score = bce_loss(y_pred_tensor, y_true_tensor).item()
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
