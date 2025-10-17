"""
Utility functions for plotting normalization analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Any, List, Tuple
import os


def plot_distribution_histograms(stats: Dict[str, Any], norm_type: str,
                                 output_path: str, axis: int = None) -> None:
    """
    Plot histograms of relevant statistics based on normalization type.

    Args:
        stats: Statistics dictionary
        norm_type: Type of normalization
        output_path: Path to save the plot
        axis: Axis along which normalization was checked
    """
    if norm_type == "standard":
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Standard Normalization Analysis (Axis: {axis})', fontsize=16)

        # Mean distribution
        axes[0, 0].hist(stats['mean'].flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Expected: 0')
        axes[0, 0].set_xlabel('Mean')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Means')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Std distribution
        axes[0, 1].hist(stats['std'].flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(1, color='red', linestyle='--', linewidth=2, label='Expected: 1')
        axes[0, 1].set_xlabel('Standard Deviation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Standard Deviations')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Mean absolute deviation
        axes[1, 0].hist(stats['mean_abs_deviation'].flatten(), bins=50,
                       edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Absolute Deviation from 0')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Mean Absolute Deviation')
        axes[1, 0].grid(True, alpha=0.3)

        # Std deviation
        axes[1, 1].hist(stats['std_deviation'].flatten(), bins=50,
                       edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Absolute Deviation from 1')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Std Absolute Deviation')
        axes[1, 1].grid(True, alpha=0.3)

    elif norm_type == "minmax":
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Min-Max Normalization Analysis (Axis: {axis})', fontsize=16)

        # Min distribution
        axes[0, 0].hist(stats['min'].flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Expected: 0')
        axes[0, 0].set_xlabel('Min Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Minimum Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Max distribution
        axes[0, 1].hist(stats['max'].flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(1, color='red', linestyle='--', linewidth=2, label='Expected: 1')
        axes[0, 1].set_xlabel('Max Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Maximum Values')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Min deviation
        axes[1, 0].hist(stats['min_deviation'].flatten(), bins=50,
                       edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Absolute Deviation from 0')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Min Absolute Deviation')
        axes[1, 0].grid(True, alpha=0.3)

        # Max deviation
        axes[1, 1].hist(stats['max_deviation'].flatten(), bins=50,
                       edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Absolute Deviation from 1')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Max Absolute Deviation')
        axes[1, 1].grid(True, alpha=0.3)

    elif norm_type == "robust":
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Robust Normalization Analysis (Axis: {axis})', fontsize=16)

        # Median distribution
        axes[0, 0].hist(stats['median'].flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Expected: 0')
        axes[0, 0].set_xlabel('Median')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Medians')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # IQR distribution
        axes[0, 1].hist(stats['iqr'].flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(1, color='red', linestyle='--', linewidth=2, label='Expected: 1')
        axes[0, 1].set_xlabel('IQR')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of IQR')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Q25 and Q75
        axes[0, 2].hist(stats['q25'].flatten(), bins=50, edgecolor='black',
                       alpha=0.5, label='Q25', color='blue')
        axes[0, 2].hist(stats['q75'].flatten(), bins=50, edgecolor='black',
                       alpha=0.5, label='Q75', color='green')
        axes[0, 2].set_xlabel('Value')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Distribution of Q25 and Q75')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Median deviation
        axes[1, 0].hist(stats['median_abs_deviation'].flatten(), bins=50,
                       edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Absolute Deviation from 0')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Median Absolute Deviation')
        axes[1, 0].grid(True, alpha=0.3)

        # IQR deviation
        axes[1, 1].hist(stats['iqr_deviation'].flatten(), bins=50,
                       edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Absolute Deviation from 1')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('IQR Absolute Deviation')
        axes[1, 1].grid(True, alpha=0.3)

        # Hide the last subplot
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plots to {output_path}")


def plot_extreme_examples(data: np.ndarray, stats: Dict[str, Any], norm_type: str,
                         extreme_indices: Tuple[np.ndarray, np.ndarray],
                         output_path: str, axis: int = None) -> None:
    """
    Plot extreme examples based on the relevant metric.

    Args:
        data: Original data array
        stats: Statistics dictionary
        norm_type: Type of normalization
        extreme_indices: Tuple of (highest_indices, lowest_indices)
        output_path: Path to save the plot
        axis: Axis along which normalization was checked
    """
    highest_idx, lowest_idx = extreme_indices
    n = len(highest_idx)

    # Determine metric name and get metric values
    if norm_type == "standard":
        metric_name = "Standard Deviation"
        metric = stats['std']
    elif norm_type == "minmax":
        metric_name = "Max Value"
        metric = stats['max']
    elif norm_type == "robust":
        metric_name = "IQR"
        metric = stats['iqr']

    # Create figure
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f'Extreme Examples - {metric_name} (Axis: {axis})', fontsize=16)

    # Helper function to extract sample from data
    def get_sample(flat_idx):
        if axis is None:
            return data.flatten()
        else:
            # Convert flat index to multi-dimensional index for the metric array
            multi_idx = np.unravel_index(flat_idx, metric.shape)
            # Build index tuple for the original data
            # Insert slice(None) at the position of the reduced axis
            index_list = list(multi_idx)
            index_list.insert(axis, slice(None))
            return data[tuple(index_list)]

    # Plot highest metric examples
    for i, idx in enumerate(highest_idx):
        sample = get_sample(idx)

        if sample.ndim > 1:
            # For 2D data, plot as heatmap
            im = axes[0, i].imshow(sample.squeeze().T, aspect='auto', cmap='viridis')
            axes[0, i].set_title(f'Highest {i+1}\n{metric_name}={metric.flatten()[idx]:.4f}')
            plt.colorbar(im, ax=axes[0, i])
        else:
            # For 1D data, plot as line
            axes[0, i].plot(sample)
            axes[0, i].set_title(f'Highest {i+1}\n{metric_name}={metric.flatten()[idx]:.4f}')
            axes[0, i].grid(True, alpha=0.3)

    # Plot lowest metric examples
    for i, idx in enumerate(lowest_idx):
        sample = get_sample(idx)

        if sample.ndim > 1:
            im = axes[1, i].imshow(sample.squeeze().T, aspect='auto', cmap='viridis')
            axes[1, i].set_title(f'Lowest {i+1}\n{metric_name}={metric.flatten()[idx]:.4f}')
            plt.colorbar(im, ax=axes[1, i])
        else:
            axes[1, i].plot(sample)
            axes[1, i].set_title(f'Lowest {i+1}\n{metric_name}={metric.flatten()[idx]:.4f}')
            axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved extreme examples plot to {output_path}")


def plot_extreme_examples_multi_metric(data: np.ndarray, stats: Dict[str, Any],
                                       norm_type: str, n_examples: int,
                                       output_dir: str, file_name: str,
                                       axis: int = None) -> None:
    """
    Plot extreme examples for multiple metrics (e.g., mean and std for standard normalization).

    Args:
        data: Original data array
        stats: Statistics dictionary
        norm_type: Type of normalization
        n_examples: Number of extreme examples to plot
        output_dir: Directory to save plots
        file_name: Base name for output files
        axis: Axis along which normalization was checked
    """
    # Define metrics to plot based on normalization type
    if norm_type == "standard":
        metrics = [
            ("mean", "Mean", stats['mean']),
            ("std", "Standard Deviation", stats['std'])
        ]
    elif norm_type == "minmax":
        metrics = [
            ("min", "Min Value", stats['min']),
            ("max", "Max Value", stats['max'])
        ]
    elif norm_type == "robust":
        metrics = [
            ("median", "Median", stats['median']),
            ("iqr", "IQR", stats['iqr'])
        ]

    # Plot extreme examples for each metric
    for metric_key, metric_name, metric_values in metrics:
        from stats_utils import get_extreme_indices
        extreme_indices = get_extreme_indices(metric_values, n_examples)
        extreme_plot_path = os.path.join(output_dir, f"{file_name}_extreme_{metric_key}.png")

        highest_idx, lowest_idx = extreme_indices
        n = len(highest_idx)

        # Create figure
        fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(f'Extreme Examples - {metric_name} (Axis: {axis})', fontsize=16)

        # Helper function to extract sample from data
        def get_sample(flat_idx):
            if axis is None:
                return data.flatten()
            else:
                # Convert flat index to multi-dimensional index for the metric array
                multi_idx = np.unravel_index(flat_idx, metric_values.shape)
                # Build index tuple for the original data
                # Insert slice(None) at the position of the reduced axis
                index_list = list(multi_idx)
                index_list.insert(axis, slice(None))
                return data[tuple(index_list)]

        # Plot highest metric examples
        for i, idx in enumerate(highest_idx):
            sample = get_sample(idx)

            if sample.ndim > 1:
                # For 2D data, plot as heatmap
                im = axes[0, i].imshow(sample.squeeze().T, aspect='auto', cmap='viridis')
                axes[0, i].set_title(f'Highest {i+1}\n{metric_name}={metric_values.flatten()[idx]:.4f}')
                plt.colorbar(im, ax=axes[0, i])
            else:
                # For 1D data, plot as line
                axes[0, i].plot(sample)
                axes[0, i].set_title(f'Highest {i+1}\n{metric_name}={metric_values.flatten()[idx]:.4f}')
                axes[0, i].grid(True, alpha=0.3)

        # Plot lowest metric examples
        for i, idx in enumerate(lowest_idx):
            sample = get_sample(idx)

            if sample.ndim > 1:
                im = axes[1, i].imshow(sample.squeeze().T, aspect='auto', cmap='viridis')
                axes[1, i].set_title(f'Lowest {i+1}\n{metric_name}={metric_values.flatten()[idx]:.4f}')
                plt.colorbar(im, ax=axes[1, i])
            else:
                axes[1, i].plot(sample)
                axes[1, i].set_title(f'Lowest {i+1}\n{metric_name}={metric_values.flatten()[idx]:.4f}')
                axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(extreme_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved extreme examples plot to {extreme_plot_path}")


def create_summary_plot(all_file_stats: List[Dict], file_names: List[str],
                       norm_type: str, output_path: str) -> None:
    """
    Create a summary comparison plot across multiple files.

    Args:
        all_file_stats: List of statistics dictionaries for each file
        file_names: List of file names
        norm_type: Type of normalization
        output_path: Path to save the plot
    """
    n_files = len(all_file_stats)

    if norm_type == "standard":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Normalization Comparison Across Files', fontsize=16)

        # Collect means and stds
        mean_means = [np.mean(stats['mean']) for stats in all_file_stats]
        mean_stds = [np.mean(stats['std']) for stats in all_file_stats]

        # Bar plot for means
        axes[0].bar(range(n_files), mean_means, alpha=0.7, edgecolor='black')
        axes[0].axhline(0, color='red', linestyle='--', linewidth=2, label='Expected: 0')
        axes[0].set_xlabel('File')
        axes[0].set_ylabel('Average Mean')
        axes[0].set_title('Mean Values Across Files')
        axes[0].set_xticks(range(n_files))
        axes[0].set_xticklabels([os.path.basename(f) for f in file_names], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Bar plot for stds
        axes[1].bar(range(n_files), mean_stds, alpha=0.7, edgecolor='black')
        axes[1].axhline(1, color='red', linestyle='--', linewidth=2, label='Expected: 1')
        axes[1].set_xlabel('File')
        axes[1].set_ylabel('Average Std')
        axes[1].set_title('Standard Deviation Across Files')
        axes[1].set_xticks(range(n_files))
        axes[1].set_xticklabels([os.path.basename(f) for f in file_names], rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    elif norm_type == "minmax":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Normalization Comparison Across Files', fontsize=16)

        # Collect mins and maxs
        mean_mins = [np.mean(stats['min']) for stats in all_file_stats]
        mean_maxs = [np.mean(stats['max']) for stats in all_file_stats]

        # Bar plot for mins
        axes[0].bar(range(n_files), mean_mins, alpha=0.7, edgecolor='black')
        axes[0].axhline(0, color='red', linestyle='--', linewidth=2, label='Expected: 0')
        axes[0].set_xlabel('File')
        axes[0].set_ylabel('Average Min')
        axes[0].set_title('Minimum Values Across Files')
        axes[0].set_xticks(range(n_files))
        axes[0].set_xticklabels([os.path.basename(f) for f in file_names], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Bar plot for maxs
        axes[1].bar(range(n_files), mean_maxs, alpha=0.7, edgecolor='black')
        axes[1].axhline(1, color='red', linestyle='--', linewidth=2, label='Expected: 1')
        axes[1].set_xlabel('File')
        axes[1].set_ylabel('Average Max')
        axes[1].set_title('Maximum Values Across Files')
        axes[1].set_xticks(range(n_files))
        axes[1].set_xticklabels([os.path.basename(f) for f in file_names], rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    elif norm_type == "robust":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Normalization Comparison Across Files', fontsize=16)

        # Collect medians and IQRs
        mean_medians = [np.mean(stats['median']) for stats in all_file_stats]
        mean_iqrs = [np.mean(stats['iqr']) for stats in all_file_stats]

        # Bar plot for medians
        axes[0].bar(range(n_files), mean_medians, alpha=0.7, edgecolor='black')
        axes[0].axhline(0, color='red', linestyle='--', linewidth=2, label='Expected: 0')
        axes[0].set_xlabel('File')
        axes[0].set_ylabel('Average Median')
        axes[0].set_title('Median Values Across Files')
        axes[0].set_xticks(range(n_files))
        axes[0].set_xticklabels([os.path.basename(f) for f in file_names], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Bar plot for IQRs
        axes[1].bar(range(n_files), mean_iqrs, alpha=0.7, edgecolor='black')
        axes[1].axhline(1, color='red', linestyle='--', linewidth=2, label='Expected: 1')
        axes[1].set_xlabel('File')
        axes[1].set_ylabel('Average IQR')
        axes[1].set_title('IQR Across Files')
        axes[1].set_xticks(range(n_files))
        axes[1].set_xticklabels([os.path.basename(f) for f in file_names], rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary comparison plot to {output_path}")
