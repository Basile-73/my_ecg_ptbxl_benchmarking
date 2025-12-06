"""
Evaluate ECG Denoising Models by Diagnostic Class

This script computes SNR and RMSE metrics grouped by diagnostic classes
(diagnostic, subdiagnostic, superdiagnostic) with bootstrap confidence intervals
across multiple noise configurations.

Key Features:
- Handles multi-label classification (samples can belong to multiple classes)
- Computes bootstrap confidence intervals (90% CI using 5th/95th percentiles)
- Supports multiple noise configurations with prediction caching (compute once, reuse across classes)
- Generates per-class performance metrics for all models across noise configs
- Creates grouped bar charts comparing models within each noise configuration
- Saves results as CSV files for downstream analysis
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '../'))
sys.path.insert(0, os.path.join(script_dir, '../classification'))
sys.path.insert(0, os.path.join(script_dir, '../../ecg_noise/source'))

# Import required modules
import torch
from denoising_utils.utils import calculate_snr, calculate_rmse, get_model, run_denoise_inference
from denoising_utils.preprocessing import remove_bad_labels_labels_only
from denoising_utils.qui_plot import MODEL_COLOR_MAP
from utils.utils import compute_label_aggregations, load_labels_only
from ecg_noise_factory.noise import NoiseFactory

# Import bootstrap function from evaluate_similarity to avoid code duplication
sys.path.insert(0, script_dir)  # Ensure we can import from same directory
from evaluate_similarity import get_appropriate_bootstrap_samples


def compute_bootstrap_stats(metric_values, bootstrap_samples, class_mask):
    """
    Compute bootstrap statistics for a specific class.

    Args:
        metric_values: Array of per-sample metric values (all samples)
        bootstrap_samples: List of bootstrap index arrays
        class_mask: Boolean mask indicating class membership

    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    boot_means = []

    for boot_idx in bootstrap_samples:
        # Apply class mask to bootstrap indices
        boot_class_mask = class_mask[boot_idx]
        boot_class_values = metric_values[boot_idx][boot_class_mask]

        # Only compute mean if we have samples
        if len(boot_class_values) > 0:
            boot_means.append(boot_class_values.mean())

    # Compute point estimate and confidence intervals
    if len(boot_means) > 0:
        mean = metric_values[class_mask].mean()
        lower_ci = np.percentile(boot_means, 5)
        upper_ci = np.percentile(boot_means, 95)
        return mean, lower_ci, upper_ci
    else:
        # No valid bootstrap samples
        return np.nan, np.nan, np.nan


def generate_class_qui_plot(class_name, class_type, class_results_df, noise_configs, output_folder):
    """
    Generate per-class visualization with noise configuration grouping.

    NOTE: This is a custom, per-class version that mimics the visual style of qui_plot()
    from denoising_utils/qui_plot.py. This function creates grouped bar charts where bars
    are grouped by noise configuration, with models within each noise config group.
    Supports both single and multiple noise configurations.

    Args:
        class_name: Name of the diagnostic class
        class_type: Type of classification (diagnostic/subdiagnostic/superdiagnostic)
        class_results_df: DataFrame with columns: noise_config, model, n_samples, snr_mean,
                         snr_lower_ci, snr_upper_ci, rmse_mean, rmse_lower_ci, rmse_upper_ci
        noise_configs: List of noise config dicts with 'name' keys (or empty list for single config)
        output_folder: Folder to save outputs

    Returns:
        None (saves figure and CSV to output_folder)
    """
    # Determine if we have multiple noise configs
    if noise_configs and len(noise_configs) > 1:
        # Multi-noise config mode: grouped bar chart
        _generate_multi_noise_class_plot(class_name, class_type, class_results_df, noise_configs, output_folder)
    else:
        # Single config mode: simple comparison
        _generate_single_noise_class_plot(class_name, class_type, class_results_df, output_folder)


def _generate_single_noise_class_plot(class_name, class_type, class_results_df, output_folder):
    """Generate simple 2-panel bar chart for single noise configuration."""
    # Sort models by shared color map order for consistency
    model_order = [m for m in MODEL_COLOR_MAP.keys() if m in class_results_df['model'].values]
    other_models = [m for m in class_results_df['model'].values if m not in MODEL_COLOR_MAP]
    model_order.extend(sorted(other_models))

    # Filter and sort DataFrame
    df_sorted = class_results_df[class_results_df['model'].isin(model_order)].copy()
    df_sorted['model'] = pd.Categorical(df_sorted['model'], categories=model_order, ordered=True)
    df_sorted = df_sorted.sort_values('model')

    # Prepare data for plotting
    models = df_sorted['model'].values
    rmse_means = df_sorted['rmse_mean'].values
    rmse_lower = df_sorted['rmse_lower_ci'].values
    rmse_upper = df_sorted['rmse_upper_ci'].values
    snr_means = df_sorted['snr_mean'].values
    snr_lower = df_sorted['snr_lower_ci'].values
    snr_upper = df_sorted['snr_upper_ci'].values

    # Get number of samples (should be same for all models in this class)
    n_samples = df_sorted['n_samples'].values[0] if len(df_sorted) > 0 else 0

    # Compute error bars (distance from mean to CI bounds)
    rmse_err_lower = rmse_means - rmse_lower
    rmse_err_upper = rmse_upper - rmse_means
    snr_err_lower = snr_means - snr_lower
    snr_err_upper = snr_upper - snr_means

    # Get colors for each model
    colors = [MODEL_COLOR_MAP.get(m, '#cccccc') for m in models]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # X positions for bars
    x_pos = np.arange(len(models))

    # Left panel: RMSE
    ax1.bar(x_pos, rmse_means, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.errorbar(x_pos, rmse_means,
                yerr=[rmse_err_lower, rmse_err_upper],
                fmt='none', ecolor='black', capsize=3, linewidth=1.5)

    # Add value labels above bars
    for i, (x, y, y_err) in enumerate(zip(x_pos, rmse_means, rmse_err_upper)):
        ax1.text(x, y + y_err + 0.01, f'{y:.3f}',
                ha='center', va='bottom', fontsize=8, color='black',
                rotation=90,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='none', alpha=0.7))

    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_title(f'RMSE Comparison\n{class_name} ({class_type})\nn={n_samples} samples', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_yscale('log')

    # Right panel: Output SNR
    ax2.bar(x_pos, snr_means, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.errorbar(x_pos, snr_means,
                yerr=[snr_err_lower, snr_err_upper],
                fmt='none', ecolor='black', capsize=3, linewidth=1.5)

    # Add value labels above bars
    for i, (x, y, y_err) in enumerate(zip(x_pos, snr_means, snr_err_upper)):
        ax2.text(x, y + y_err + 0.5, f'{y:.2f}',
                ha='center', va='bottom', fontsize=8, color='black',
                rotation=90,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='none', alpha=0.7))

    ax2.set_ylabel('Output SNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_title(f'Output SNR Comparison\n{class_name} ({class_type})\nn={n_samples} samples', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_folder, f'{class_name}_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create summary DataFrame with point estimates and CI
    summary_data = []
    for _, row in df_sorted.iterrows():
        summary_data.append({
            'model': row['model'],
            'n_samples': row['n_samples'],
            'point_rmse': row['rmse_mean'],
            'mean_rmse': row['rmse_mean'],
            'lower_rmse': row['rmse_lower_ci'],
            'upper_rmse': row['rmse_upper_ci'],
            'point_output_snr': row['snr_mean'],
            'mean_output_snr': row['snr_mean'],
            'lower_output_snr': row['snr_lower_ci'],
            'upper_output_snr': row['snr_upper_ci']
        })

    summary_df = pd.DataFrame(summary_data)

    # Save summary CSV
    csv_path = os.path.join(output_folder, f'{class_name}_summary.csv')
    summary_df.to_csv(csv_path, index=False)


def _generate_multi_noise_class_plot(class_name, class_type, class_results_df, noise_configs, output_folder):
    """Generate grouped bar chart for multiple noise configurations."""
    # Sort models by shared color map order for consistency
    model_order = [m for m in MODEL_COLOR_MAP.keys() if m in class_results_df['model'].values]
    other_models = [m for m in class_results_df['model'].values if m not in MODEL_COLOR_MAP]
    model_order.extend(sorted(other_models))

    # Get number of samples (should be same for all models/configs in this class)
    n_samples = class_results_df['n_samples'].values[0] if len(class_results_df) > 0 else 0

    # Create figure with 2 subplots (RMSE left, SNR right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{class_name} ({class_type}) - Multi-Noise Comparison\nn={n_samples} samples',
                fontsize=16, fontweight='bold', y=0.98)

    # Setup for grouped bar chart
    n_models = len(model_order)
    n_configs = len(noise_configs)
    group_width = 0.8
    bar_width = group_width / n_models

    config_positions = []
    config_labels = []

    # Plot 1: RMSE
    x_pos = 0
    max_rmse_y = 0
    min_rmse_y = float('inf')

    for config_idx, noise_config in enumerate(noise_configs):
        config_name = noise_config['name']
        config_df = class_results_df[class_results_df['noise_config'] == config_name]

        for model_idx, model_name in enumerate(model_order):
            model_df = config_df[config_df['model'] == model_name]

            if not model_df.empty:
                point_rmse = model_df['rmse_mean'].values[0]
                lower_rmse = model_df['rmse_lower_ci'].values[0]
                upper_rmse = model_df['rmse_upper_ci'].values[0]

                # Compute error bars (asymmetric)
                lower_err = point_rmse - lower_rmse
                upper_err = upper_rmse - point_rmse

                color = MODEL_COLOR_MAP.get(model_name, '#cccccc')

                bar_pos = x_pos + model_idx * bar_width
                ax1.bar(bar_pos, point_rmse, bar_width,
                       yerr=[[lower_err], [upper_err]], capsize=3,
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
                       label=model_name if config_idx == 0 else '')

                # Add text label on top of bar with 90° rotation
                ax1.text(bar_pos, point_rmse + upper_err + 0.01, f'{point_rmse:.3f}',
                        ha='center', va='bottom', fontsize=8, color='black',
                        rotation=90,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='none', alpha=0.7))

                # Track maximum and minimum y values
                max_rmse_y = max(max_rmse_y, point_rmse + upper_err)
                min_rmse_y = min(min_rmse_y, point_rmse - lower_err)

        group_center = x_pos + (n_models - 1) * bar_width / 2
        config_positions.append(group_center)
        config_labels.append(config_name.upper())
        x_pos += group_width + 0.3

    ax1.set_xlabel('Noise Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_title('RMSE Comparison (90% Bootstrap CI)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(config_positions)
    ax1.set_xticklabels(config_labels, fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    # Ensure positive values for log scale
    min_rmse_y = max(min_rmse_y, 1e-6)
    ax1.set_ylim(bottom=min_rmse_y * 0.95, top=max_rmse_y * 1.15)
    ax1.set_yscale('log')

    # Add vertical lines between noise config groups
    for i in range(1, len(noise_configs)):
        line_x = config_positions[i-1] + (config_positions[i] - config_positions[i-1]) / 2
        ax1.axvline(x=line_x, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Plot 2: Output SNR
    x_pos = 0
    max_snr_y = 0
    min_snr_y = float('inf')

    for config_idx, noise_config in enumerate(noise_configs):
        config_name = noise_config['name']
        config_df = class_results_df[class_results_df['noise_config'] == config_name]

        for model_idx, model_name in enumerate(model_order):
            model_df = config_df[config_df['model'] == model_name]

            if not model_df.empty:
                point_snr = model_df['snr_mean'].values[0]
                lower_snr = model_df['snr_lower_ci'].values[0]
                upper_snr = model_df['snr_upper_ci'].values[0]

                # Compute error bars (asymmetric)
                lower_err = point_snr - lower_snr
                upper_err = upper_snr - point_snr

                color = MODEL_COLOR_MAP.get(model_name, '#cccccc')

                bar_pos = x_pos + model_idx * bar_width
                ax2.bar(bar_pos, point_snr, bar_width,
                       yerr=[[lower_err], [upper_err]], capsize=3,
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
                       label=model_name if config_idx == 0 else '')

                # Add text label on top of bar with 90° rotation
                ax2.text(bar_pos, point_snr + upper_err + 0.5, f'{point_snr:.2f}',
                        ha='center', va='bottom', fontsize=8, color='black',
                        rotation=90,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='none', alpha=0.7))

                # Track maximum and minimum y values
                max_snr_y = max(max_snr_y, point_snr + upper_err)
                min_snr_y = min(min_snr_y, point_snr - lower_err)

        x_pos += group_width + 0.3

    ax2.set_xlabel('Noise Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Output SNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_title('Output SNR Comparison (90% Bootstrap CI)', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(config_positions)
    ax2.set_xticklabels(config_labels, fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=min_snr_y * 0.95, top=max_snr_y * 1.15)

    # Add vertical lines between noise config groups
    for i in range(1, len(noise_configs)):
        line_x = config_positions[i-1] + (config_positions[i] - config_positions[i-1]) / 2
        ax2.axvline(x=line_x, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.95, 1])

    # Save figure
    fig_path = os.path.join(output_folder, f'{class_name}_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create summary DataFrame
    summary_df = class_results_df.copy()
    summary_df = summary_df.rename(columns={
        'rmse_mean': 'point_rmse',
        'rmse_lower_ci': 'lower_rmse',
        'rmse_upper_ci': 'upper_rmse',
        'snr_mean': 'point_output_snr',
        'snr_lower_ci': 'lower_output_snr',
        'snr_upper_ci': 'upper_output_snr'
    })
    summary_df['mean_rmse'] = summary_df['point_rmse']
    summary_df['mean_output_snr'] = summary_df['point_output_snr']

    # Save summary CSV
    csv_path = os.path.join(output_folder, f'{class_name}_summary.csv')
    summary_df.to_csv(csv_path, index=False)


def evaluate_model_by_class(model_name, predictions_cache, noise_configs, clean_test,
                            test_labels_dict, bootstrap_samples):
    """
    Evaluate a single model's performance grouped by diagnostic classes across noise configs.

    Args:
        model_name: Name of the model
        predictions_cache: Dict mapping {noise_config_name: {model_name: predictions_array}}
        noise_configs: List of noise config dicts (empty for single config)
        clean_test: Clean test signals
        test_labels_dict: Dictionary with keys 'diagnostic', 'subdiagnostic', 'superdiagnostic'
                         containing respective label DataFrames
        bootstrap_samples: List of bootstrap index arrays

    Returns:
        Dictionary mapping class_type -> {noise_config -> DataFrame of results}
    """
    print(f"\nEvaluating {model_name}...")

    # Initialize nested results structure
    results_dict = {
        'diagnostic': {},
        'subdiagnostic': {},
        'superdiagnostic': {}
    }

    # Determine noise config names
    if noise_configs:
        config_names = [nc['name'] for nc in noise_configs]
    else:
        config_names = ['default']  # Single config fallback

    # Loop over noise configurations
    for config_name in config_names:
        print(f"  Processing noise config: {config_name}")

        # Retrieve predictions from cache
        predictions = predictions_cache[config_name][model_name]

        # Compute per-sample metrics for this noise config
        n_samples = len(clean_test)
        snr_per_sample = np.zeros(n_samples)
        rmse_per_sample = np.zeros(n_samples)

        print(f"    Computing per-sample metrics...")
        for i in tqdm(range(n_samples), desc=f"    Samples ({config_name})", leave=False):
            clean_signal = clean_test[i].squeeze()
            pred_signal = predictions[i].squeeze()

            snr_per_sample[i] = calculate_snr(clean_signal, pred_signal)
            rmse_per_sample[i] = calculate_rmse(clean_signal, pred_signal)

        # Evaluate for each class type
        for class_type in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
            print(f"    Processing {class_type} classes...")

            test_labels = test_labels_dict[class_type]

            # Get all unique classes from the multi-label column
            all_classes = set()
            for sample_labels in test_labels[class_type]:
                if isinstance(sample_labels, list):
                    all_classes.update(sample_labels)

            all_classes = sorted(list(all_classes))

            # Compute metrics for each class
            class_results = []

            for class_name in tqdm(all_classes, desc=f"    {class_type} classes", leave=False):
                # Create boolean mask for samples belonging to this class
                class_mask = np.array([
                    class_name in sample_labels if isinstance(sample_labels, list) else False
                    for sample_labels in test_labels[class_type]
                ])

                n_class_samples = class_mask.sum()

                # Skip classes with very few samples
                if n_class_samples < 5:
                    continue

                # Compute bootstrap statistics
                snr_mean, snr_lower, snr_upper = compute_bootstrap_stats(
                    snr_per_sample, bootstrap_samples, class_mask
                )
                rmse_mean, rmse_lower, rmse_upper = compute_bootstrap_stats(
                    rmse_per_sample, bootstrap_samples, class_mask
                )

                class_results.append({
                    'noise_config': config_name,
                    'model': model_name,
                    'class': class_name,
                    'n_samples': n_class_samples,
                    'snr_mean': snr_mean,
                    'snr_lower_ci': snr_lower,
                    'snr_upper_ci': snr_upper,
                    'rmse_mean': rmse_mean,
                    'rmse_lower_ci': rmse_lower,
                    'rmse_upper_ci': rmse_upper
                })

            results_dict[class_type][config_name] = pd.DataFrame(class_results)

    return results_dict


def evaluate_noisy_input_by_class(clean_test, predictions_cache, noise_configs, test_labels_dict, bootstrap_samples):
    """
    Evaluate noisy input as a baseline across noise configs.

    Args:
        clean_test: Clean test data
        predictions_cache: Dict mapping {noise_config_name: {'noisy_input': noisy_test_array}}
        noise_configs: List of noise config dicts (empty for single config)
        test_labels_dict: Dictionary mapping class types to label DataFrames
        bootstrap_samples: Bootstrap sample indices for confidence intervals

    Returns:
        Dictionary mapping class_type -> {noise_config -> DataFrame with noisy input metrics}
    """
    print("\nComputing noisy input baseline metrics by class...")

    # Initialize nested results structure
    results_dict = {
        'diagnostic': {},
        'subdiagnostic': {},
        'superdiagnostic': {}
    }

    # Determine noise config names
    if noise_configs:
        config_names = [nc['name'] for nc in noise_configs]
    else:
        config_names = ['default']  # Single config fallback

    # Loop over noise configurations
    for config_name in config_names:
        print(f"  Processing noise config: {config_name}")

        # Retrieve noisy test data from cache
        noisy_test = predictions_cache[config_name]['noisy_input']

        # Compute per-sample metrics
        n_samples = len(clean_test)
        snr_per_sample = np.zeros(n_samples)
        rmse_per_sample = np.zeros(n_samples)

        print(f"    Computing per-sample metrics...")
        for i in tqdm(range(n_samples), desc=f"    Samples ({config_name})", leave=False):
            clean_signal = clean_test[i].squeeze()
            noisy_signal = noisy_test[i].squeeze()

            snr_per_sample[i] = calculate_snr(clean_signal, noisy_signal)
            rmse_per_sample[i] = calculate_rmse(clean_signal, noisy_signal)

        # Evaluate for each class type
        for class_type in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
            print(f"    Processing {class_type} classes...")

            test_labels = test_labels_dict[class_type]

            # Get all unique classes from the multi-label column
            all_classes = set()
            for sample_labels in test_labels[class_type]:
                if isinstance(sample_labels, list):
                    all_classes.update(sample_labels)

            all_classes = sorted(list(all_classes))

            # Compute metrics for each class
            class_results = []

            for class_name in tqdm(all_classes, desc=f"    {class_type} classes", leave=False):
                # Create boolean mask for samples belonging to this class
                class_mask = np.array([
                    class_name in sample_labels if isinstance(sample_labels, list) else False
                    for sample_labels in test_labels[class_type]
                ])

                n_class_samples = class_mask.sum()

                # Skip classes with very few samples
                if n_class_samples < 5:
                    continue

                # Compute bootstrap statistics
                snr_mean, snr_lower, snr_upper = compute_bootstrap_stats(
                    snr_per_sample, bootstrap_samples, class_mask
                )
                rmse_mean, rmse_lower, rmse_upper = compute_bootstrap_stats(
                    rmse_per_sample, bootstrap_samples, class_mask
                )

                class_results.append({
                    'noise_config': config_name,
                    'model': 'noisy_input',
                    'class': class_name,
                    'n_samples': n_class_samples,
                    'snr_mean': snr_mean,
                    'snr_lower_ci': snr_lower,
                    'snr_upper_ci': snr_upper,
                    'rmse_mean': rmse_mean,
                    'rmse_lower_ci': rmse_lower,
                    'rmse_upper_ci': rmse_upper
                })

            results_dict[class_type][config_name] = pd.DataFrame(class_results)

    return results_dict


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate denoising models by diagnostic class'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='mycode/denoising/configs/test_mamba_models.yaml',
        help='Path to experiment config file'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=None,
        help='Number of bootstrap samples (overrides config if specified)'
    )
    parser.add_argument(
        '--noise-configs',
        type=str,
        nargs='*',
        default=None,
        help='Noise configuration names to evaluate (reads from config if not specified)'
    )
    args = parser.parse_args()

    # Load configuration first
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Determine bootstrap sample count: CLI override or config default
    if args.n_bootstrap is not None:
        n_bootstrap = args.n_bootstrap
        bootstrap_source = "CLI argument"
    else:
        n_bootstrap = config.get('evaluation', {}).get('bootstrap_samples', 100)
        bootstrap_source = "config file"

    print("\n" + "="*80)
    print("EVALUATE DENOISING MODELS BY DIAGNOSTIC CLASS")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Bootstrap samples: {n_bootstrap} (from {bootstrap_source})")

    # Setup paths
    datafolder = config['datafolder']
    sampling_frequency = config['sampling_frequency']
    test_fold = config['test_fold']
    exp_folder = os.path.join(config['outputfolder'], config['experiment_name'])

    print(f"  Experiment folder: {exp_folder}")
    print(f"  Data folder: {datafolder}")
    print(f"  Sampling frequency: {sampling_frequency} Hz")
    print(f"  Test fold: {test_fold}")

    # Load clean test data
    print("\nLoading clean test data...")
    clean_test_path = os.path.join(exp_folder, 'data', 'clean_test.npy')
    clean_test = np.load(clean_test_path)
    print(f"  ✓ Loaded {len(clean_test)} test samples")
    print(f"  Shape: {clean_test.shape}")

    # Load noisy test data
    print("\nLoading noisy test data...")
    noisy_test_path = os.path.join(exp_folder, 'data', 'noisy_test_eval.npy')
    if not os.path.exists(noisy_test_path):
        print(f"\n❌ ERROR: Noisy test data not found at {noisy_test_path}")
        print("  Please run the evaluation script to generate noisy test data first.")
        sys.exit(1)
    noisy_test = np.load(noisy_test_path)
    print(f"  ✓ Loaded {len(noisy_test)} noisy test samples")
    print(f"  Shape: {noisy_test.shape}")

    # Verify noisy_test and clean_test alignment
    if len(noisy_test) != len(clean_test):
        print(f"\n❌ ERROR: Noisy test count ({len(noisy_test)}) != clean test count ({len(clean_test)})")
        print("  The noisy and clean test data must have the same number of samples.")
        sys.exit(1)
    if noisy_test.shape != clean_test.shape:
        print(f"\n❌ ERROR: Noisy test shape {noisy_test.shape} != clean test shape {clean_test.shape}")
        print("  The noisy and clean test data must have the same shape.")
        sys.exit(1)

    # Load raw PTB-XL labels (labels-only, no signal loading)
    print("\nLoading PTB-XL labels...")
    labels = load_labels_only(datafolder, sampling_frequency)
    print(f"  ✓ Loaded labels for {len(labels)} samples")

    # Apply bad label filtering (same as train.prepare(), labels-only variant)
    print("\nRemoving bad labels...")
    clean_labels = remove_bad_labels_labels_only(labels)
    print(f"  ✓ After filtering: {len(clean_labels)} samples")

    # Filter to test fold
    print(f"\nFiltering to test fold {test_fold}...")
    test_labels = clean_labels[clean_labels.strat_fold == test_fold].copy()
    print(f"  ✓ Test fold has {len(test_labels)} samples")

    # Verify alignment
    if len(test_labels) != len(clean_test):
        print(f"\n❌ ERROR: Label count ({len(test_labels)}) != clean test count ({len(clean_test)})")
        print("  This indicates a data preprocessing mismatch.")
        sys.exit(1)

    # Apply label aggregations for each class type
    print("\nAggregating labels by class type...")
    test_labels_dict = {}

    print("  - diagnostic...")
    test_labels_dict['diagnostic'] = compute_label_aggregations(
        test_labels.copy(), datafolder, 'diagnostic'
    )

    print("  - subdiagnostic...")
    test_labels_dict['subdiagnostic'] = compute_label_aggregations(
        test_labels.copy(), datafolder, 'subdiagnostic'
    )

    print("  - superdiagnostic...")
    test_labels_dict['superdiagnostic'] = compute_label_aggregations(
        test_labels.copy(), datafolder, 'superdiagnostic'
    )

    print("  ✓ Label aggregation complete")

    # Display class statistics
    for class_type, labels_df in test_labels_dict.items():
        unique_classes = set()
        for sample_labels in labels_df[class_type]:
            if isinstance(sample_labels, list):
                unique_classes.update(sample_labels)
        print(f"    {class_type}: {len(unique_classes)} unique classes")

    # Determine noise configurations
    print("\nDetermining noise configurations...")

    # Read available configs from config file first
    qui_plot_config = config.get('evaluation', {}).get('qui_plot', {})
    available_noise_configs = qui_plot_config.get('noise_configs', [])

    if args.noise_configs is not None:
        # Use CLI-specified noise configs: filter available configs by name
        noise_config_names = args.noise_configs

        # Resolve CLI names to full config dicts
        noise_configs = [
            nc for nc in available_noise_configs
            if nc['name'] in noise_config_names
        ]

        # Warn if some CLI names couldn't be resolved
        resolved_names = [nc['name'] for nc in noise_configs]
        unresolved_names = [name for name in noise_config_names if name not in resolved_names]
        if unresolved_names:
            print(f"  ⚠️  Warning: Could not find configs for: {', '.join(unresolved_names)}")

        if not noise_configs:
            print(f"  ❌ ERROR: No valid noise configs found for CLI arguments: {', '.join(noise_config_names)}")
            print(f"  Available configs: {', '.join([nc['name'] for nc in available_noise_configs])}")
            sys.exit(1)

        noise_configs_source = "CLI argument"
    elif available_noise_configs:
        # Use all configs from config file
        noise_configs = available_noise_configs
        noise_config_names = [nc['name'] for nc in noise_configs]
        noise_configs_source = "config file"
    else:
        # Fallback: use pre-computed predictions from disk (single config)
        noise_configs = []
        noise_config_names = []
        noise_configs_source = "fallback (pre-computed predictions)"

    if noise_config_names:
        print(f"  Noise configs: {', '.join(noise_config_names)} (from {noise_configs_source})")
    else:
        print(f"  Using pre-computed predictions (from {noise_configs_source})")

    # Generate bootstrap samples
    print(f"\nGenerating {n_bootstrap} bootstrap samples...")
    np.random.seed(config.get('random_seed', 42))
    y_dummy = np.ones((len(clean_test), 1))
    bootstrap_samples = get_appropriate_bootstrap_samples(y_dummy, n_bootstrap)
    print("  ✓ Bootstrap samples generated")

    # Create results folder structure
    print("\nCreating results folder structure...")
    results_folder = os.path.join(exp_folder, 'results')
    for class_type in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
        class_folder = os.path.join(results_folder, class_type)
        os.makedirs(class_folder, exist_ok=True)
    print("  ✓ Folder structure created")

    # ========================================================================
    # PHASE 1: Generate and Cache Predictions (ONCE per model per noise config)
    # ========================================================================
    if noise_configs:
        print("\n" + "="*80)
        print("PHASE 1: GENERATING AND CACHING PREDICTIONS")
        print("="*80)
        print("This phase generates predictions once and caches them for reuse")
        print("across all class evaluations to avoid redundant computation.")

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() and
                             config['hardware']['use_cuda'] else 'cpu')
        print(f"Using device: {device}")

        # Initialize prediction cache: {noise_config_name: {model_name: predictions_array, 'noisy_input': noisy_test_array}}
        predictions_cache = {}

        # Generate predictions for each noise configuration
        for noise_config in tqdm(noise_configs, desc="Processing noise configs", position=0):
            config_name = noise_config['name']
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                noise_config['path']
            )

            print(f"\n--- Noise Config: {config_name} ---")
            print(f"Path: {config_path}")

            # Initialize prediction cache for this noise config
            predictions_cache[config_name] = {}

            # Initialize NoiseFactory
            noise_data_path = os.path.join(os.path.dirname(__file__), '../../ecg_noise/data')
            noise_factory = NoiseFactory(
                data_path=noise_data_path,
                sampling_rate=sampling_frequency,
                config_path=config_path,
                mode='eval'  # Use eval mode for fair comparison
            )

            # Generate noisy test data
            print(f"  Generating noisy test data...")
            noisy_test_nc = noise_factory.add_noise(
                x=clean_test, batch_axis=0, channel_axis=2, length_axis=1
            )
            predictions_cache[config_name]['noisy_input'] = noisy_test_nc
            print(f"  ✓ Noisy test data generated")

            # Generate predictions for each model
            for model_config in tqdm(config['models'], desc=f"  Generating predictions ({config_name})", position=1, leave=False):
                model_name = model_config['name']
                model_type = model_config['type']
                is_stage2 = model_type.lower() in ['stage2', 'drnet']

                # Load model
                model_folder = os.path.join(exp_folder, 'models', model_name)
                model_path = os.path.join(model_folder, 'best_model.pth')

                if not os.path.exists(model_path):
                    tqdm.write(f"    Warning: Model not found at {model_path}, skipping...")
                    continue

                # For Stage2 models, load Stage1 model first
                stage1_predictions = None
                if is_stage2:
                    stage1_model_name = model_config.get('stage1_model')
                    if not stage1_model_name:
                        tqdm.write(f"    Warning: Stage2 model {model_name} missing stage1_model config, skipping...")
                        continue

                    stage1_folder = os.path.join(exp_folder, 'models', stage1_model_name)
                    stage1_path = os.path.join(stage1_folder, 'best_model.pth')

                    if not os.path.exists(stage1_path):
                        tqdm.write(f"    Warning: Stage1 model not found at {stage1_path}, skipping...")
                        continue

                    # Get Stage1 model config
                    stage1_config = next((m for m in config['models'] if m['name'] == stage1_model_name), None)
                    if not stage1_config:
                        tqdm.write(f"    Warning: Stage1 model config not found for {stage1_model_name}, skipping...")
                        continue

                    # Load and run Stage1 model
                    stage1_model = get_model(stage1_config['type'], input_length=clean_test[0].shape[0], is_stage2=False)
                    stage1_model.load_state_dict(torch.load(stage1_path, map_location=device))
                    stage1_model = stage1_model.to(device)
                    stage1_model.eval()

                    # Generate Stage1 predictions
                    stage1_predictions = []
                    with torch.no_grad():
                        for i in range(len(clean_test)):
                            noisy_sample = noisy_test_nc[i]
                            noisy_sample = torch.FloatTensor(noisy_sample).permute(1, 0).unsqueeze(0).unsqueeze(0)
                            noisy_sample = noisy_sample.to(device)

                            stage1_pred = run_denoise_inference(stage1_model, noisy_sample, is_stage2=False)
                            stage1_predictions.append(stage1_pred)

                    stage1_predictions = np.array(stage1_predictions)

                    # Clean up Stage1 model
                    del stage1_model
                    torch.cuda.empty_cache()

                # Load main model
                model = get_model(model_type, input_length=clean_test[0].shape[0], is_stage2=is_stage2)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device)
                model.eval()

                # Generate predictions
                predictions = []
                with torch.no_grad():
                    for i in range(len(clean_test)):
                        noisy_sample = noisy_test_nc[i]
                        noisy_sample = torch.FloatTensor(noisy_sample).permute(1, 0).unsqueeze(0).unsqueeze(0)

                        if is_stage2:
                            # Concatenate noisy + stage1_pred
                            stage1_sample = stage1_predictions[i]
                            stage1_sample = torch.FloatTensor(stage1_sample).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                            model_input = torch.cat([noisy_sample, stage1_sample], dim=1)
                        else:
                            model_input = noisy_sample

                        model_input = model_input.to(device)
                        pred = run_denoise_inference(model, model_input, is_stage2=is_stage2)
                        predictions.append(pred)

                predictions = np.array(predictions)
                predictions_cache[config_name][model_name] = predictions

                # Clean up model
                del model
                torch.cuda.empty_cache()

        print("\n✓ All predictions generated and cached")
    else:
        # Fallback: load pre-computed predictions from disk (single config)
        print("\n" + "="*80)
        print("USING PRE-COMPUTED PREDICTIONS (SINGLE CONFIG)")
        print("="*80)

        predictions_cache = {'default': {'noisy_input': noisy_test}}

        # Load pre-computed predictions for each model
        for model_config in config['models']:
            model_name = model_config['name']
            pred_path = os.path.join(exp_folder, 'models', model_name, 'predictions.npy')

            if os.path.exists(pred_path):
                predictions = np.load(pred_path)
                predictions_cache['default'][model_name] = predictions
                print(f"  ✓ Loaded predictions for {model_name}")
            else:
                print(f"  ⚠️  Predictions not found for {model_name}, skipping...")

        noise_configs = []  # Empty list for single config mode

    # ========================================================================
    # PHASE 2: Evaluate by Class (Reuse Cached Predictions)
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: EVALUATING BY CLASS (REUSING CACHED PREDICTIONS)")
    print("="*80)

    # Determine config names
    if noise_configs:
        config_names = [nc['name'] for nc in noise_configs]
    else:
        config_names = ['default']

    # Initialize nested results structure: {class_type: {noise_config: [DataFrames]}}
    all_results = {
        class_type: {config_name: [] for config_name in config_names}
        for class_type in ['diagnostic', 'subdiagnostic', 'superdiagnostic']
    }

    # Evaluate noisy input baseline
    print("\nEvaluating noisy input baseline...")
    noisy_input_results = evaluate_noisy_input_by_class(
        clean_test, predictions_cache, noise_configs, test_labels_dict, bootstrap_samples
    )

    # Add noisy input results first
    for class_type in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
        for config_name in config_names:
            all_results[class_type][config_name].append(noisy_input_results[class_type][config_name])

    # Evaluate each model
    for model_config in config['models']:
        model_name = model_config['name']

        # Check if predictions exist in cache
        predictions_exist = all(
            model_name in predictions_cache[config_name]
            for config_name in config_names
        )

        if not predictions_exist:
            print(f"\n⚠️  Predictions not found in cache for {model_name}, skipping...")
            continue

        print(f"\n✓ Evaluating {model_name}...")

        # Evaluate by class (using cached predictions)
        model_results = evaluate_model_by_class(
            model_name, predictions_cache, noise_configs, clean_test,
            test_labels_dict, bootstrap_samples
        )

        # Accumulate results
        for class_type in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
            for config_name in config_names:
                all_results[class_type][config_name].append(model_results[class_type][config_name])

    # ========================================================================
    # PHASE 3: Save Results and Generate Grouped Visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: SAVING RESULTS AND GENERATING VISUALIZATIONS")
    print("="*80)

    for class_type in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
        # Check if we have any results for this class_type
        has_results = any(all_results[class_type][config_name] for config_name in config_names)

        if not has_results:
            print(f"\n⚠️  No results for {class_type}, skipping...")
            continue

        print(f"\nProcessing {class_type} results...")

        # Flatten nested structure: combine all noise configs
        combined_dfs = []
        for config_name in config_names:
            if all_results[class_type][config_name]:
                combined_dfs.extend(all_results[class_type][config_name])

        if not combined_dfs:
            print(f"  ⚠️  No valid results for {class_type}, skipping...")
            continue

        combined_df = pd.concat(combined_dfs, ignore_index=True)

        # Guard: Check if combined_df is empty or lacks 'class' column
        if combined_df.empty:
            print(f"  ⚠️  No valid results for {class_type}, skipping...")
            continue

        if 'class' not in combined_df.columns:
            print(f"  ⚠️  Missing 'class' column in {class_type} results, skipping...")
            continue

        # Save aggregate CSV to class_type subfolder (includes noise_config column)
        class_folder = os.path.join(results_folder, class_type)
        aggregate_csv_path = os.path.join(class_folder, f'metrics_by_{class_type}.csv')
        combined_df.to_csv(aggregate_csv_path, index=False)
        print(f"  ✓ Saved aggregate CSV: {aggregate_csv_path}")
        if noise_configs:
            print(f"    (includes noise_config column for multi-noise comparison)")

        # Get unique classes for this class_type
        unique_classes = combined_df['class'].unique()
        print(f"  Generating visualizations for {len(unique_classes)} classes...")

        # Generate visualization for each class
        for class_name in tqdm(unique_classes, desc=f"  {class_type} visualizations"):
            # Filter results for this class
            class_results = combined_df[combined_df['class'] == class_name].copy()

            # Skip if insufficient data
            if len(class_results) == 0:
                continue

            # Generate visualization (grouped by noise config if multiple configs)
            try:
                generate_class_qui_plot(
                    class_name=class_name,
                    class_type=class_type,
                    class_results_df=class_results,
                    noise_configs=noise_configs,
                    output_folder=class_folder
                )
            except Exception as e:
                print(f"    Warning: Failed to generate plot for {class_name}: {e}")

        # Print summary table
        print(f"\n  {class_type.upper()} SUMMARY:")
        print("  " + "-" * 78)

        # Group by model and noise_config, show top 5 classes by sample count
        for model in combined_df['model'].unique():
            model_df = combined_df[combined_df['model'] == model]

            # Show breakdown by noise config if multi-config
            if noise_configs:
                for config_name in config_names:
                    config_model_df = model_df[model_df['noise_config'] == config_name]
                    if not config_model_df.empty:
                        top_classes = config_model_df.nlargest(5, 'n_samples')

                        print(f"\n  {model} ({config_name}):")
                        print(f"    {'Class':<20} {'N':<8} {'SNR (dB)':<20} {'RMSE':<20}")
                        print(f"    {'-'*20} {'-'*8} {'-'*20} {'-'*20}")

                        for _, row in top_classes.iterrows():
                            snr_str = f"{row['snr_mean']:.2f} [{row['snr_lower_ci']:.2f}, {row['snr_upper_ci']:.2f}]"
                            rmse_str = f"{row['rmse_mean']:.4f} [{row['rmse_lower_ci']:.4f}, {row['rmse_upper_ci']:.4f}]"
                            print(f"    {row['class']:<20} {row['n_samples']:<8} {snr_str:<20} {rmse_str:<20}")
            else:
                # Single config mode
                top_classes = model_df.nlargest(5, 'n_samples')

                print(f"\n  {model}:")
                print(f"    {'Class':<20} {'N':<8} {'SNR (dB)':<20} {'RMSE':<20}")
                print(f"    {'-'*20} {'-'*8} {'-'*20} {'-'*20}")

                for _, row in top_classes.iterrows():
                    snr_str = f"{row['snr_mean']:.2f} [{row['snr_lower_ci']:.2f}, {row['snr_upper_ci']:.2f}]"
                    rmse_str = f"{row['rmse_mean']:.4f} [{row['rmse_lower_ci']:.4f}, {row['rmse_upper_ci']:.4f}]"
                    print(f"    {row['class']:<20} {row['n_samples']:<8} {snr_str:<20} {rmse_str:<20}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_folder}")
    print("\nGenerated folder structure:")
    print("  results/")
    print("    diagnostic/")
    print("      - metrics_by_diagnostic.csv (with noise_config column if multi-noise)")
    if noise_configs:
        print("      - <class_name>_comparison.png (grouped bar charts by noise config)")
    else:
        print("      - <class_name>_comparison.png (simple bar charts)")
    print("      - <class_name>_summary.csv (per-class metrics)")
    print("    subdiagnostic/")
    print("      - metrics_by_subdiagnostic.csv")
    print("      - <class_name>_comparison.png")
    print("      - <class_name>_summary.csv")
    print("    superdiagnostic/")
    print("      - metrics_by_superdiagnostic.csv")
    print("      - <class_name>_comparison.png")
    print("      - <class_name>_summary.csv")

    if noise_configs:
        print(f"\n✓ Multi-noise evaluation completed across {len(noise_configs)} configurations")
        print(f"  Configs: {', '.join([nc['name'] for nc in noise_configs])}")
    else:
        print("\n✓ Single-noise evaluation completed (using pre-computed predictions)")


if __name__ == '__main__':
    main()
