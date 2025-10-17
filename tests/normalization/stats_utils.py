"""
Utility functions for computing normalization statistics.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


def compute_standard_stats(data: np.ndarray, axis: int = None) -> Dict[str, Any]:
    """
    Compute statistics for standard normalization (z-score).

    Args:
        data: Input array
        axis: Axis along which to compute statistics

    Returns:
        Dictionary with mean, std, and deviation metrics
    """
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)

    stats = {
        'mean': mean,
        'std': std,
        'mean_abs_deviation': np.abs(mean),
        'std_deviation': np.abs(std - 1.0)
    }

    return stats


def compute_minmax_stats(data: np.ndarray, axis: int = None) -> Dict[str, Any]:
    """
    Compute statistics for min-max normalization.

    Args:
        data: Input array
        axis: Axis along which to compute statistics

    Returns:
        Dictionary with min, max, and deviation metrics
    """
    min_val = np.min(data, axis=axis)
    max_val = np.max(data, axis=axis)

    stats = {
        'min': min_val,
        'max': max_val,
        'min_deviation': np.abs(min_val - 0.0),
        'max_deviation': np.abs(max_val - 1.0)
    }

    return stats


def compute_robust_stats(data: np.ndarray, axis: int = None) -> Dict[str, Any]:
    """
    Compute statistics for robust normalization (median and IQR).

    Args:
        data: Input array
        axis: Axis along which to compute statistics

    Returns:
        Dictionary with median, IQR, and deviation metrics
    """
    median = np.median(data, axis=axis)
    q75 = np.percentile(data, 75, axis=axis)
    q25 = np.percentile(data, 25, axis=axis)
    iqr = q75 - q25

    stats = {
        'median': median,
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
        'median_abs_deviation': np.abs(median),
        'iqr_deviation': np.abs(iqr - 1.0)
    }

    return stats


def get_extreme_indices(metric: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get indices of the n highest and n lowest values in a metric array.

    Args:
        metric: Array of metric values
        n: Number of extreme examples to extract

    Returns:
        Tuple of (highest_indices, lowest_indices)
    """
    # Flatten the metric array and get indices
    flat_metric = metric.flatten()
    flat_indices = np.arange(len(flat_metric))

    # Sort by metric value
    sorted_idx = np.argsort(flat_metric)

    # Get n lowest and n highest
    lowest_idx = sorted_idx[:n]
    highest_idx = sorted_idx[-n:][::-1]  # Reverse to get descending order

    return highest_idx, lowest_idx


def analyze_normalization(data: np.ndarray, norm_type: str, axis: int = None) -> Dict[str, Any]:
    """
    Analyze normalization of data based on specified type.

    Args:
        data: Input array
        norm_type: Type of normalization ('standard', 'minmax', 'robust')
        axis: Axis along which to check normalization

    Returns:
        Dictionary with statistics and analysis results
    """
    if norm_type == "standard":
        stats = compute_standard_stats(data, axis)
    elif norm_type == "minmax":
        stats = compute_minmax_stats(data, axis)
    elif norm_type == "robust":
        stats = compute_robust_stats(data, axis)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

    return stats


def format_stats_report(stats: Dict[str, Any], norm_type: str, axis: int = None) -> str:
    """
    Format statistics into a human-readable report.

    Args:
        stats: Statistics dictionary
        norm_type: Type of normalization
        axis: Axis along which normalization was checked

    Returns:
        Formatted string report
    """
    report = []
    report.append(f"Normalization Type: {norm_type}")
    report.append(f"Analysis Axis: {axis if axis is not None else 'Global'}")
    report.append("-" * 60)

    if norm_type == "standard":
        report.append(f"Mean: {np.mean(stats['mean']):.6f} (expected: 0.0)")
        report.append(f"Mean Range: [{np.min(stats['mean']):.6f}, {np.max(stats['mean']):.6f}]")
        report.append(f"Std: {np.mean(stats['std']):.6f} (expected: 1.0)")
        report.append(f"Std Range: [{np.min(stats['std']):.6f}, {np.max(stats['std']):.6f}]")
        report.append(f"Max Abs Mean Deviation: {np.max(stats['mean_abs_deviation']):.6f}")
        report.append(f"Max Std Deviation: {np.max(stats['std_deviation']):.6f}")

    elif norm_type == "minmax":
        report.append(f"Min: {np.mean(stats['min']):.6f} (expected: 0.0)")
        report.append(f"Min Range: [{np.min(stats['min']):.6f}, {np.max(stats['min']):.6f}]")
        report.append(f"Max: {np.mean(stats['max']):.6f} (expected: 1.0)")
        report.append(f"Max Range: [{np.min(stats['max']):.6f}, {np.max(stats['max']):.6f}]")
        report.append(f"Max Min Deviation: {np.max(stats['min_deviation']):.6f}")
        report.append(f"Max Max Deviation: {np.max(stats['max_deviation']):.6f}")

    elif norm_type == "robust":
        report.append(f"Median: {np.mean(stats['median']):.6f} (expected: 0.0)")
        report.append(f"Median Range: [{np.min(stats['median']):.6f}, {np.max(stats['median']):.6f}]")
        report.append(f"IQR: {np.mean(stats['iqr']):.6f} (expected: 1.0)")
        report.append(f"IQR Range: [{np.min(stats['iqr']):.6f}, {np.max(stats['iqr']):.6f}]")
        report.append(f"Max Abs Median Deviation: {np.max(stats['median_abs_deviation']):.6f}")
        report.append(f"Max IQR Deviation: {np.max(stats['iqr_deviation']):.6f}")

    return "\n".join(report)
