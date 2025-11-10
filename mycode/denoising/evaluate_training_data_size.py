"""
Evaluate Training Data Size Experiment Results

This script evaluates denoising model performance across different training set sizes,
computing comparative metrics and generating visualizations showing how performance
scales with training data volume.

Features:
- Loads results from training_data_size_experiment.py output
- Evaluates all models across all training fold sizes (1, 4, 8)
- Computes SNR improvement, RMSE metrics with confidence intervals
- Creates comparative visualizations showing performance vs training size
- Generates summary tables and detailed reports
"""
import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datetime import datetime
from tqdm import tqdm

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '../../ecg_noise/source'))

# Import required modules
from denoising_utils.utils import calculate_snr, calculate_rmse
from ecg_noise_factory.noise import NoiseFactory

# Set plot style
sns.set_style('whitegrid')


def load_experiment_metadata(experiment_folder):
    """
    Load and parse the experiment metadata JSON file.

    Args:
        experiment_folder: Path to training_data_size_experiment folder

    Returns:
        Dictionary with metadata
    """
    print("\n" + "="*80)
    print("LOADING EXPERIMENT METADATA")
    print("="*80)

    metadata_path = os.path.join(experiment_folder, 'experiment_metadata.json')

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Please ensure the experiment has completed successfully."
        )

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Validate required fields
    required_fields = ['folds_tested', 'models_trained', 'fold_experiments']
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Required field '{field}' missing from metadata")

    # Print summary
    print(f"✓ Metadata loaded successfully")
    print(f"  Experiment date: {metadata.get('experiment_date', 'N/A')}")
    print(f"  Folds tested: {metadata['folds_tested']}")
    print(f"  Models trained: {metadata['models_trained']}")

    # Print training sizes
    print("\n  Training sizes per fold:")
    for fold in metadata['folds_tested']:
        fold_info = metadata['fold_experiments'][str(fold)]
        train_size = fold_info.get('train_size', 'N/A')
        print(f"    Fold {fold}: {train_size} samples")

    return metadata


def load_fold_data(fold_experiment_path, noise_factory):
    """
    Load clean test data and generate noisy test data for a specific fold.

    Args:
        fold_experiment_path: Path to fold experiment folder
        noise_factory: NoiseFactory instance for generating noise

    Returns:
        Tuple of (clean_test, noisy_test)
    """
    # Load clean test data
    clean_test_path = os.path.join(fold_experiment_path, 'data', 'clean_test.npy')

    if not os.path.exists(clean_test_path):
        raise FileNotFoundError(f"Clean test data not found: {clean_test_path}")

    clean_test = np.load(clean_test_path)

    # Check if noisy test data already exists (for reproducibility)
    noisy_test_path = os.path.join(fold_experiment_path, 'data', 'noisy_test_eval.npy')

    if os.path.exists(noisy_test_path):
        # Load existing noisy test data
        noisy_test = np.load(noisy_test_path)
        print(f"    ✓ Loaded existing noisy test data from: {noisy_test_path}")
    else:
        # Generate noisy test data using NoiseFactory
        noisy_test = noise_factory.add_noise(
            x=clean_test,
            batch_axis=0,
            channel_axis=2,
            length_axis=1
        )

        # Save for future reproducibility
        np.save(noisy_test_path, noisy_test)
        print(f"    ✓ Generated and saved noisy test data to: {noisy_test_path}")

    # Validate shapes match
    if clean_test.shape != noisy_test.shape:
        raise ValueError(
            f"Shape mismatch: clean_test {clean_test.shape} vs noisy_test {noisy_test.shape}"
        )

    return clean_test, noisy_test


def evaluate_model_for_fold(model_name, fold_size, fold_experiment_path,
                            clean_test, noisy_test):
    """
    Evaluate a single model from a specific fold experiment.

    Args:
        model_name: Name of the model
        fold_size: Training fold size (1, 4, 8)
        fold_experiment_path: Path to fold experiment folder
        clean_test: Clean test signals
        noisy_test: Noisy test signals

    Returns:
        pd.DataFrame with per-sample results, or None if predictions missing
    """
    # Construct path to predictions
    predictions_path = os.path.join(
        fold_experiment_path, 'models', model_name, 'predictions.npy'
    )

    if not os.path.exists(predictions_path):
        print(f"  ⚠️  Predictions not found: {predictions_path}")
        return None

    # Load predictions
    predictions = np.load(predictions_path)

    # Pre-squeeze arrays to shape (N, T) for easier per-sample processing
    # This handles cases where shapes are (N, T, 1) or (N, 1, T)
    clean_test_2d = clean_test.squeeze()
    noisy_test_2d = noisy_test.squeeze()
    predictions_2d = predictions.squeeze()

    # Ensure they are at least 2D (handle single sample case)
    if clean_test_2d.ndim == 1:
        clean_test_2d = clean_test_2d.reshape(1, -1)
    if noisy_test_2d.ndim == 1:
        noisy_test_2d = noisy_test_2d.reshape(1, -1)
    if predictions_2d.ndim == 1:
        predictions_2d = predictions_2d.reshape(1, -1)

    # Validate number of samples matches
    if len(clean_test_2d) != len(predictions_2d):
        print(f"  ⚠️  Sample count mismatch for {model_name}: "
              f"clean_test {len(clean_test_2d)} vs predictions {len(predictions_2d)}")
        return None

    # Calculate metrics for each sample
    results = []

    for sample_idx in range(len(clean_test_2d)):
        # Extract 1D signals for this sample
        clean_signal = clean_test_2d[sample_idx].squeeze()
        noisy_signal = noisy_test_2d[sample_idx].squeeze()
        denoised_signal = predictions_2d[sample_idx].squeeze()

        # Calculate SNR metrics
        input_snr_db = calculate_snr(clean_signal, noisy_signal)
        output_snr_db = calculate_snr(clean_signal, denoised_signal)
        snr_improvement_db = output_snr_db - input_snr_db

        # Calculate RMSE metrics
        rmse_noisy = calculate_rmse(clean_signal, noisy_signal)
        rmse_denoised = calculate_rmse(clean_signal, denoised_signal)
        rmse_improvement = rmse_noisy - rmse_denoised
        rmse_improvement_pct = (rmse_improvement / rmse_noisy) * 100 if rmse_noisy > 0 else 0

        results.append({
            'model': model_name,
            'fold_size': fold_size,
            'sample_idx': sample_idx,
            'input_snr_db': input_snr_db,
            'output_snr_db': output_snr_db,
            'snr_improvement_db': snr_improvement_db,
            'rmse_noisy': rmse_noisy,
            'rmse_denoised': rmse_denoised,
            'rmse_improvement': rmse_improvement,
            'rmse_improvement_pct': rmse_improvement_pct
        })

    return pd.DataFrame(results)


def evaluate_all_folds(metadata, noise_factory):
    """
    Orchestrate evaluation across all folds and models.

    Args:
        metadata: Experiment metadata dictionary
        noise_factory: NoiseFactory instance

    Returns:
        pd.DataFrame with all evaluation results
    """
    print("\n" + "="*80)
    print("EVALUATING ALL FOLDS AND MODELS")
    print("="*80)

    all_results = []
    folds = metadata['folds_tested']
    models = metadata['models_trained']

    for fold in folds:
        fold_str = str(fold)
        if fold_str not in metadata['fold_experiments']:
            print(f"\n⚠️  Fold {fold} not found in metadata, skipping...")
            continue

        fold_info = metadata['fold_experiments'][fold_str]
        fold_experiment_path = fold_info['experiment_folder']

        if not os.path.exists(fold_experiment_path):
            print(f"\n⚠️  Fold experiment path not found: {fold_experiment_path}")
            continue

        print(f"\n--- Evaluating Fold {fold} ---")
        print(f"Path: {fold_experiment_path}")

        # Load fold data
        try:
            clean_test, noisy_test = load_fold_data(fold_experiment_path, noise_factory)
            print(f"✓ Loaded test data: {clean_test.shape}")
        except Exception as e:
            print(f"❌ Error loading fold data: {e}")
            continue

        # Evaluate each model
        for model_name in tqdm(models, desc=f"Fold {fold} models"):
            print(f"  Evaluating {model_name} with {fold} fold(s) of training data...")

            results_df = evaluate_model_for_fold(
                model_name, fold, fold_experiment_path, clean_test, noisy_test
            )

            if results_df is not None:
                all_results.append(results_df)
                print(f"    ✓ Evaluated {len(results_df)} samples")
            else:
                print(f"    ⚠️  Skipping {model_name} (predictions missing)")

    if len(all_results) == 0:
        raise ValueError("No results collected! Check that predictions exist.")

    # Concatenate all results
    combined_results = pd.concat(all_results, ignore_index=True)

    print(f"\n✓ Evaluation complete: {len(combined_results)} total samples evaluated")
    print(f"  Models: {combined_results['model'].nunique()}")
    print(f"  Folds: {combined_results['fold_size'].nunique()}")

    return combined_results


def compute_summary_statistics(results_df):
    """
    Compute summary statistics for each model-fold combination.

    Args:
        results_df: DataFrame with per-sample results

    Returns:
        pd.DataFrame with summary statistics
    """
    print("\n" + "="*80)
    print("COMPUTING SUMMARY STATISTICS")
    print("="*80)

    # Group by model and fold_size
    grouped = results_df.groupby(['model', 'fold_size'])

    summary = grouped.agg({
        'snr_improvement_db': ['mean', 'std', 'median'],
        'output_snr_db': ['mean', 'std'],
        'rmse_noisy': 'mean',
        'rmse_denoised': ['mean', 'std'],
        'rmse_improvement': 'mean',
        'rmse_improvement_pct': 'mean',
        'sample_idx': 'count'
    }).reset_index()

    # Flatten column names
    summary.columns = [
        'model', 'fold_size',
        'mean_snr_improvement_db', 'std_snr_improvement_db', 'median_snr_improvement_db',
        'mean_output_snr_db', 'std_output_snr_db',
        'mean_rmse_noisy',
        'mean_rmse_denoised', 'std_rmse_denoised',
        'mean_rmse_improvement',
        'mean_rmse_improvement_pct',
        'n_samples'
    ]

    print(f"✓ Computed statistics for {len(summary)} model-fold combinations")

    return summary


def get_appropriate_bootstrap_samples(data_size, n_bootstrap=100):
    """
    Generate bootstrap sample indices.

    Args:
        data_size: Number of samples in dataset
        n_bootstrap: Number of bootstrap samples

    Returns:
        List of arrays containing bootstrap indices
    """
    np.random.seed(42)  # For reproducibility
    bootstrap_samples = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(data_size, size=data_size, replace=True)
        bootstrap_samples.append(indices)

    return bootstrap_samples


def compute_bootstrap_confidence_intervals(results_df, n_bootstrap_samples=100):
    """
    Compute bootstrap confidence intervals for key metrics.

    Args:
        results_df: DataFrame with per-sample results
        n_bootstrap_samples: Number of bootstrap samples

    Returns:
        pd.DataFrame with confidence intervals
    """
    print("\n" + "="*80)
    print("COMPUTING BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*80)

    ci_results = []

    # Group by model and fold_size
    for (model, fold_size), group in results_df.groupby(['model', 'fold_size']):
        print(f"  {model} (fold {fold_size})...", end=' ')

        n_samples = len(group)
        bootstrap_indices = get_appropriate_bootstrap_samples(n_samples, n_bootstrap_samples)

        # Bootstrap SNR improvement
        snr_bootstrap_means = []
        rmse_bootstrap_means = []
        rmse_pct_bootstrap_means = []

        for indices in bootstrap_indices:
            bootstrap_sample = group.iloc[indices]
            snr_bootstrap_means.append(bootstrap_sample['snr_improvement_db'].mean())
            rmse_bootstrap_means.append(bootstrap_sample['rmse_denoised'].mean())
            rmse_pct_bootstrap_means.append(bootstrap_sample['rmse_improvement_pct'].mean())

        # Calculate percentiles for 90% CI
        snr_ci_lower = np.percentile(snr_bootstrap_means, 5)
        snr_ci_upper = np.percentile(snr_bootstrap_means, 95)
        rmse_ci_lower = np.percentile(rmse_bootstrap_means, 5)
        rmse_ci_upper = np.percentile(rmse_bootstrap_means, 95)
        rmse_pct_ci_lower = np.percentile(rmse_pct_bootstrap_means, 5)
        rmse_pct_ci_upper = np.percentile(rmse_pct_bootstrap_means, 95)

        ci_results.append({
            'model': model,
            'fold_size': fold_size,
            'mean_snr_improvement': np.mean(snr_bootstrap_means),
            'snr_ci_lower': snr_ci_lower,
            'snr_ci_upper': snr_ci_upper,
            'mean_rmse_denoised': np.mean(rmse_bootstrap_means),
            'rmse_ci_lower': rmse_ci_lower,
            'rmse_ci_upper': rmse_ci_upper,
            'mean_rmse_improvement_pct': np.mean(rmse_pct_bootstrap_means),
            'rmse_pct_ci_lower': rmse_pct_ci_lower,
            'rmse_pct_ci_upper': rmse_pct_ci_upper
        })

        print("✓")

    ci_df = pd.DataFrame(ci_results)
    print(f"\n✓ Computed confidence intervals for {len(ci_df)} combinations")

    return ci_df


def plot_performance_vs_training_size(summary_df, ci_df, metadata, output_folder):
    """
    Create line plots showing how performance scales with training data size.

    Args:
        summary_df: Summary statistics DataFrame
        ci_df: Confidence intervals DataFrame
        metadata: Experiment metadata
        output_folder: Where to save plots
    """
    print("\n" + "="*80)
    print("CREATING PERFORMANCE VS TRAINING SIZE PLOTS")
    print("="*80)

    # Define color map for models
    color_map = {
        'fcn': '#6baed6',           # light blue
        'unet': '#fc8d62',          # light red/orange
        'imunet': '#66c2a5',        # light green
        'drnet_fcn': '#08519c',     # dark blue
        'drnet_unet': '#e31a1c',    # dark red
        'drnet_imunet': '#238b45',  # dark green
        'imunet_origin': '#9467bd',     # purple
        'imunet_mamba_bn': '#ff7f0e',   # orange
        'imunet_mamba_up': '#17becf',   # cyan/teal
        'imunet_mamba_early': '#e377c2', # magenta/pink
        'imunet_mamba_late': '#bcbd22',  # yellow-green
        'mecge_phase': '#c5b0d5'         # light purple
    }

    # Get unique fold sizes (sorted)
    fold_sizes = sorted(summary_df['fold_size'].unique())
    models = sorted(summary_df['model'].unique())

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance vs Training Data Size', fontsize=16, fontweight='bold')

    # Subplot 1: SNR Improvement vs Training Size
    ax = axes[0, 0]
    for model in models:
        model_data = summary_df[summary_df['model'] == model]
        model_ci = ci_df[ci_df['model'] == model]

        snr_values = [model_data[model_data['fold_size'] == f]['mean_snr_improvement_db'].values[0]
                     if len(model_data[model_data['fold_size'] == f]) > 0 else np.nan
                     for f in fold_sizes]

        ci_lower = [model_ci[model_ci['fold_size'] == f]['snr_ci_lower'].values[0]
                   if len(model_ci[model_ci['fold_size'] == f]) > 0 else np.nan
                   for f in fold_sizes]
        ci_upper = [model_ci[model_ci['fold_size'] == f]['snr_ci_upper'].values[0]
                   if len(model_ci[model_ci['fold_size'] == f]) > 0 else np.nan
                   for f in fold_sizes]

        color = color_map.get(model, '#000000')
        ax.plot(fold_sizes, snr_values, marker='o', label=model,
               color=color, linewidth=2, markersize=8)
        ax.fill_between(fold_sizes, ci_lower, ci_upper, alpha=0.2, color=color)

    ax.set_xlabel('Training Fold Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean SNR Improvement (dB)', fontsize=12, fontweight='bold')
    ax.set_title('SNR Improvement vs Training Data Size', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(fold_sizes)

    # Subplot 2: Output SNR vs Training Size
    ax = axes[0, 1]
    for model in models:
        model_data = summary_df[summary_df['model'] == model]

        output_snr_values = [model_data[model_data['fold_size'] == f]['mean_output_snr_db'].values[0]
                            if len(model_data[model_data['fold_size'] == f]) > 0 else np.nan
                            for f in fold_sizes]

        color = color_map.get(model, '#000000')
        ax.plot(fold_sizes, output_snr_values, marker='o', label=model,
               color=color, linewidth=2, markersize=8)

    ax.set_xlabel('Training Fold Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Output SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('Output SNR vs Training Data Size', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(fold_sizes)

    # Subplot 3: RMSE Denoised vs Training Size
    ax = axes[1, 0]
    for model in models:
        model_data = summary_df[summary_df['model'] == model]
        model_ci = ci_df[ci_df['model'] == model]

        rmse_values = [model_data[model_data['fold_size'] == f]['mean_rmse_denoised'].values[0]
                      if len(model_data[model_data['fold_size'] == f]) > 0 else np.nan
                      for f in fold_sizes]

        ci_lower = [model_ci[model_ci['fold_size'] == f]['rmse_ci_lower'].values[0]
                   if len(model_ci[model_ci['fold_size'] == f]) > 0 else np.nan
                   for f in fold_sizes]
        ci_upper = [model_ci[model_ci['fold_size'] == f]['rmse_ci_upper'].values[0]
                   if len(model_ci[model_ci['fold_size'] == f]) > 0 else np.nan
                   for f in fold_sizes]

        color = color_map.get(model, '#000000')
        ax.plot(fold_sizes, rmse_values, marker='o', label=model,
               color=color, linewidth=2, markersize=8)
        ax.fill_between(fold_sizes, ci_lower, ci_upper, alpha=0.2, color=color)

    ax.set_xlabel('Training Fold Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean RMSE (Denoised)', fontsize=12, fontweight='bold')
    ax.set_title('RMSE (Denoised) vs Training Data Size', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(fold_sizes)

    # Subplot 4: RMSE Improvement % vs Training Size
    ax = axes[1, 1]
    for model in models:
        model_data = summary_df[summary_df['model'] == model]

        rmse_imp_pct = [model_data[model_data['fold_size'] == f]['mean_rmse_improvement_pct'].values[0]
                       if len(model_data[model_data['fold_size'] == f]) > 0 else np.nan
                       for f in fold_sizes]

        color = color_map.get(model, '#000000')
        ax.plot(fold_sizes, rmse_imp_pct, marker='o', label=model,
               color=color, linewidth=2, markersize=8)

    ax.set_xlabel('Training Fold Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('RMSE Improvement (%) vs Training Data Size', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(fold_sizes)

    plt.tight_layout()

    # Save plots
    png_path = os.path.join(output_folder, 'performance_vs_training_size.png')
    pdf_path = os.path.join(output_folder, 'performance_vs_training_size.pdf')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {png_path}")
    print(f"✓ Saved: {pdf_path}")


def plot_model_comparison_per_fold(summary_df, ci_df, output_folder):
    """
    Create bar charts comparing models at each training data size.
    Shows both SNR improvement and RMSE improvement percentage with error bars.

    Args:
        summary_df: Summary statistics DataFrame
        ci_df: Confidence intervals DataFrame
        output_folder: Where to save plots
    """
    print("\n" + "="*80)
    print("CREATING MODEL COMPARISON PER FOLD PLOTS")
    print("="*80)

    fold_sizes = sorted(summary_df['fold_size'].unique())
    models = sorted(summary_df['model'].unique())

    # Color map
    color_map = {
        'fcn': '#6baed6',
        'unet': '#fc8d62',
        'imunet': '#66c2a5',
        'drnet_fcn': '#08519c',
        'drnet_unet': '#e31a1c',
        'drnet_imunet': '#238b45',
        'imunet_origin': '#9467bd',
        'imunet_mamba_bn': '#ff7f0e',
        'imunet_mamba_up': '#17becf',
        'imunet_mamba_early': '#e377c2',
        'imunet_mamba_late': '#bcbd22',
        'mecge_phase': '#c5b0d5'
    }

    # Create figure with 2 rows: SNR and RMSE
    fig, axes = plt.subplots(2, len(fold_sizes), figsize=(18, 10))
    if len(fold_sizes) == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle('Model Comparison at Different Training Data Sizes',
                fontsize=16, fontweight='bold', y=0.98)

    # Row 1: SNR Improvement
    for idx, fold in enumerate(fold_sizes):
        ax = axes[0, idx]
        fold_data = summary_df[summary_df['fold_size'] == fold]

        x_pos = np.arange(len(models))
        snr_values = [fold_data[fold_data['model'] == m]['mean_snr_improvement_db'].values[0]
                     if len(fold_data[fold_data['model'] == m]) > 0 else 0
                     for m in models]

        # Extract confidence intervals for error bars
        snr_errors_lower = []
        snr_errors_upper = []
        for m in models:
            ci_data = ci_df[(ci_df['model'] == m) & (ci_df['fold_size'] == fold)]
            if len(ci_data) > 0:
                snr_val = fold_data[fold_data['model'] == m]['mean_snr_improvement_db'].values[0]
                snr_errors_lower.append(snr_val - ci_data['snr_ci_lower'].values[0])
                snr_errors_upper.append(ci_data['snr_ci_upper'].values[0] - snr_val)
            else:
                snr_errors_lower.append(0)
                snr_errors_upper.append(0)

        snr_yerr = np.array([snr_errors_lower, snr_errors_upper])

        colors = [color_map.get(m, '#000000') for m in models]
        bars = ax.bar(x_pos, snr_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                     yerr=snr_yerr, capsize=3, error_kw={'linewidth': 1.5, 'ecolor': 'black'})

        # Add value labels
        for bar, val in zip(bars, snr_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('SNR Improvement (dB)', fontsize=10, fontweight='bold')
        ax.set_title(f'Fold {fold}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    # Row 2: RMSE Improvement %
    for idx, fold in enumerate(fold_sizes):
        ax = axes[1, idx]
        fold_data = summary_df[summary_df['fold_size'] == fold]

        x_pos = np.arange(len(models))
        rmse_values = [fold_data[fold_data['model'] == m]['mean_rmse_improvement_pct'].values[0]
                      if len(fold_data[fold_data['model'] == m]) > 0 else 0
                      for m in models]

        # Extract confidence intervals for error bars
        rmse_pct_errors_lower = []
        rmse_pct_errors_upper = []
        for m in models:
            ci_data = ci_df[(ci_df['model'] == m) & (ci_df['fold_size'] == fold)]
            if len(ci_data) > 0:
                rmse_val = fold_data[fold_data['model'] == m]['mean_rmse_improvement_pct'].values[0]
                rmse_pct_errors_lower.append(rmse_val - ci_data['rmse_pct_ci_lower'].values[0])
                rmse_pct_errors_upper.append(ci_data['rmse_pct_ci_upper'].values[0] - rmse_val)
            else:
                rmse_pct_errors_lower.append(0)
                rmse_pct_errors_upper.append(0)

        rmse_pct_yerr = np.array([rmse_pct_errors_lower, rmse_pct_errors_upper])

        colors = [color_map.get(m, '#000000') for m in models]
        bars = ax.bar(x_pos, rmse_values, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5, hatch='//',
                     yerr=rmse_pct_yerr, capsize=3, error_kw={'linewidth': 1.5, 'ecolor': 'black'})

        # Add value labels
        for bar, val in zip(bars, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('RMSE Improvement (%)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    # Add legend for metrics
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.8, edgecolor='black', label='SNR Improvement'),
        Patch(facecolor='gray', alpha=0.8, edgecolor='black', hatch='//', label='RMSE Improvement %')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10,
              bbox_to_anchor=(0.98, 0.96))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(output_folder, 'model_comparison_per_fold.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {save_path}")


def plot_training_efficiency(summary_df, metadata, output_folder):
    """
    Plot performance vs total training steps to show training efficiency.

    Args:
        summary_df: Summary statistics DataFrame
        metadata: Experiment metadata
        output_folder: Where to save plots
    """
    print("\n" + "="*80)
    print("CREATING TRAINING EFFICIENCY PLOT")
    print("="*80)

    # Extract total training steps from metadata
    efficiency_data = []

    for _, row in summary_df.iterrows():
        model = row['model']
        fold = row['fold_size']
        snr_improvement = row['mean_snr_improvement_db']

        # Get total steps from metadata
        fold_info = metadata['fold_experiments'][str(fold)]
        if 'models' in fold_info and model in fold_info['models']:
            total_steps = fold_info['models'][model]['total_steps']

            efficiency_data.append({
                'model': model,
                'fold_size': fold,
                'total_steps': total_steps,
                'snr_improvement': snr_improvement
            })

    if len(efficiency_data) == 0:
        print("⚠️  No efficiency data available, skipping plot")
        return

    eff_df = pd.DataFrame(efficiency_data)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    color_map = {
        'fcn': '#6baed6',
        'unet': '#fc8d62',
        'imunet': '#66c2a5',
        'drnet_fcn': '#08519c',
        'drnet_unet': '#e31a1c',
        'drnet_imunet': '#238b45',
        'imunet_origin': '#9467bd',
        'imunet_mamba_bn': '#ff7f0e',
        'imunet_mamba_up': '#17becf',
        'imunet_mamba_early': '#e377c2',
        'imunet_mamba_late': '#bcbd22',
        'mecge_phase': '#c5b0d5'
    }

    marker_sizes = {1: 100, 4: 200, 8: 300}

    for model in eff_df['model'].unique():
        model_data = eff_df[eff_df['model'] == model]

        colors = [color_map.get(model, '#000000')] * len(model_data)
        sizes = [marker_sizes.get(f, 150) for f in model_data['fold_size']]

        ax.scatter(model_data['total_steps'], model_data['snr_improvement'],
                  c=colors, s=sizes, alpha=0.7, edgecolors='black',
                  linewidths=2, label=model)

    ax.set_xlabel('Total Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean SNR Improvement (dB)', fontsize=12, fontweight='bold')
    ax.set_title('Training Efficiency: Performance vs Total Steps',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add fold size annotation
    legend_text = "Marker size represents fold size:\n  Small=1 fold, Medium=4 folds, Large=8 folds"
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=9)

    plt.tight_layout()

    save_path = os.path.join(output_folder, 'training_efficiency.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {save_path}")


def create_summary_tables(summary_df, metadata, output_folder):
    """
    Create formatted summary tables for reporting.

    Args:
        summary_df: Summary statistics DataFrame
        metadata: Experiment metadata
        output_folder: Where to save tables
    """
    print("\n" + "="*80)
    print("CREATING SUMMARY TABLES")
    print("="*80)

    # Create pivot table for SNR improvement
    snr_pivot = summary_df.pivot(index='model', columns='fold_size',
                                 values='mean_snr_improvement_db')
    snr_pivot.columns = [f'Fold_{c}' for c in snr_pivot.columns]

    # Create pivot table for RMSE improvement %
    rmse_pivot = summary_df.pivot(index='model', columns='fold_size',
                                  values='mean_rmse_improvement_pct')
    rmse_pivot.columns = [f'Fold_{c}' for c in rmse_pivot.columns]

    # Create training sizes DataFrame to add as columns
    training_sizes = {}
    for fold in metadata['folds_tested']:
        fold_info = metadata['fold_experiments'][str(fold)]
        training_sizes[f'TrainSize_Fold_{fold}'] = fold_info.get('train_size', 'N/A')

    # Create a row of training sizes (same for all models)
    training_size_row = pd.DataFrame([training_sizes] * len(snr_pivot), index=snr_pivot.index)

    # Combine tables with training sizes
    summary_table = pd.concat([
        training_size_row,
        snr_pivot,
        rmse_pivot
    ], axis=1, keys=['Training_Sizes', 'SNR_Improvement_dB', 'RMSE_Improvement_Pct'])

    # Save to CSV
    csv_path = os.path.join(output_folder, 'summary_table.csv')
    summary_table.to_csv(csv_path)
    print(f"✓ Saved: {csv_path}")

    # Print formatted table
    print("\n" + "-"*80)
    print("SUMMARY TABLE: SNR Improvement (dB)")
    print("-"*80)
    print(snr_pivot.round(2).to_string())

    print("\n" + "-"*80)
    print("SUMMARY TABLE: RMSE Improvement (%)")
    print("-"*80)
    print(rmse_pivot.round(1).to_string())

    print("\n" + "-"*80)
    print("TRAINING SIZES")
    print("-"*80)
    for key, value in training_sizes.items():
        print(f"  {key}: {value} samples")
    print("-"*80)


def save_detailed_results(results_df, summary_df, ci_df, metadata, output_folder):
    """
    Save all detailed results and metadata.

    Args:
        results_df: Per-sample results DataFrame
        summary_df: Summary statistics DataFrame
        ci_df: Confidence intervals DataFrame
        metadata: Experiment metadata
        output_folder: Where to save results
    """
    print("\n" + "="*80)
    print("SAVING DETAILED RESULTS")
    print("="*80)

    # Save detailed per-sample results
    detailed_path = os.path.join(output_folder, 'detailed_results.csv')
    results_df.to_csv(detailed_path, index=False)
    print(f"✓ Saved: {detailed_path}")

    # Save summary statistics
    summary_path = os.path.join(output_folder, 'summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved: {summary_path}")

    # Save confidence intervals
    ci_path = os.path.join(output_folder, 'confidence_intervals.csv')
    ci_df.to_csv(ci_path, index=False)
    print(f"✓ Saved: {ci_path}")

    # Copy experiment metadata
    metadata_copy_path = os.path.join(output_folder, 'experiment_metadata_copy.json')
    with open(metadata_copy_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved: {metadata_copy_path}")

    # Create evaluation report
    report = {
        'evaluation_date': datetime.now().isoformat(),
        'n_models_evaluated': len(summary_df['model'].unique()),
        'n_folds_evaluated': len(summary_df['fold_size'].unique()),
        'total_samples_evaluated': len(results_df),
        'key_findings': {
            'best_model_per_fold': {},
            'overall_best_model': None,
            'performance_trend': None
        }
    }

    # Find best model per fold
    for fold in summary_df['fold_size'].unique():
        fold_data = summary_df[summary_df['fold_size'] == fold]
        best_model = fold_data.loc[fold_data['mean_snr_improvement_db'].idxmax(), 'model']
        best_snr = fold_data['mean_snr_improvement_db'].max()
        report['key_findings']['best_model_per_fold'][str(fold)] = {
            'model': best_model,
            'snr_improvement_db': float(best_snr)
        }

    # Overall best
    best_overall = summary_df.loc[summary_df['mean_snr_improvement_db'].idxmax()]
    report['key_findings']['overall_best_model'] = {
        'model': best_overall['model'],
        'fold_size': int(best_overall['fold_size']),
        'snr_improvement_db': float(best_overall['mean_snr_improvement_db'])
    }

    # Save report
    report_path = os.path.join(output_folder, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved: {report_path}")

    print("\n✓ All results saved successfully!")


def main():
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description='Evaluate training data size experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python evaluate_training_data_size.py \\
    --experiment-folder mycode/denoising/output/training_data_size_experiment \\
    --n-bootstrap 100
        """
    )

    parser.add_argument(
        '--experiment-folder',
        type=str,
        required=True,
        help='Path to training_data_size_experiment folder'
    )

    parser.add_argument(
        '--noise-config',
        type=str,
        default=None,
        help='Path to noise config YAML (default: from base config)'
    )

    parser.add_argument(
        '--noise-data',
        type=str,
        default=None,
        help='Path to noise data directory (default: from base config)'
    )

    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=100,
        help='Number of bootstrap samples (default: 100)'
    )

    parser.add_argument(
        '--output-folder',
        type=str,
        default=None,
        help='Override output location (default: {experiment_folder}/evaluation_results)'
    )

    args = parser.parse_args()

    # Validate experiment folder exists
    if not os.path.exists(args.experiment_folder):
        print(f"❌ ERROR: Experiment folder not found: {args.experiment_folder}")
        sys.exit(1)

    # Print header
    print("\n" + "="*80)
    print("TRAINING DATA SIZE EXPERIMENT EVALUATION")
    print("="*80)
    print(f"Experiment folder: {args.experiment_folder}")
    print(f"Bootstrap samples: {args.n_bootstrap}")

    try:
        # Load experiment metadata
        metadata = load_experiment_metadata(args.experiment_folder)

        # Load base configuration to get noise settings and sampling rate
        base_config_path = metadata.get('base_config_path', None)

        if base_config_path and os.path.exists(base_config_path):
            print(f"\nLoading base config from: {base_config_path}")
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)

            # Get repository root (two levels up from script_dir)
            repo_root = os.path.dirname(os.path.dirname(script_dir))

            # Extract settings from config with absolute path resolution
            sampling_rate = base_config.get('sampling_frequency', 100)
            noise_config_path = args.noise_config or os.path.join(
                repo_root, base_config.get('noise_config_path', 'noise/configs/default.yaml')
            )
            noise_data_path = args.noise_data or os.path.join(
                repo_root, base_config.get('noise_data_path', 'noise/data/')
            )

            print(f"  Sampling rate: {sampling_rate} Hz")
            print(f"  Noise config: {noise_config_path}")
            print(f"  Noise data: {noise_data_path}")
        else:
            # Fallback to defaults if base config not available
            print("\n⚠️  Base config not found, using defaults")
            sampling_rate = 100
            noise_config_path = args.noise_config or 'noise/configs/default.yaml'
            noise_data_path = args.noise_data or 'noise/data/'

        # Create output folder
        if args.output_folder:
            output_folder = args.output_folder
        else:
            output_folder = os.path.join(args.experiment_folder, 'evaluation_results')

        os.makedirs(output_folder, exist_ok=True)
        print(f"\n✓ Output folder: {output_folder}")

        # Initialize NoiseFactory
        print("\n" + "="*80)
        print("INITIALIZING NOISE FACTORY")
        print("="*80)

        noise_factory = NoiseFactory(
            data_path=noise_data_path,
            sampling_rate=sampling_rate,
            config_path=noise_config_path,
            mode='eval'
        )
        print("✓ NoiseFactory initialized in 'eval' mode")

        # Run evaluation
        results_df = evaluate_all_folds(metadata, noise_factory)
        summary_df = compute_summary_statistics(results_df)
        ci_df = compute_bootstrap_confidence_intervals(results_df, args.n_bootstrap)

        # Generate visualizations
        plot_performance_vs_training_size(summary_df, ci_df, metadata, output_folder)
        plot_model_comparison_per_fold(summary_df, ci_df, output_folder)
        plot_training_efficiency(summary_df, metadata, output_folder)

        # Create summary outputs
        create_summary_tables(summary_df, metadata, output_folder)
        save_detailed_results(results_df, summary_df, ci_df, metadata, output_folder)

        # Print completion message
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        print(f"Results saved to: {output_folder}")

        # Print key findings
        print("\n" + "-"*80)
        print("KEY FINDINGS")
        print("-"*80)

        print("\nBest performing model at each training size:")
        for fold in sorted(summary_df['fold_size'].unique()):
            fold_data = summary_df[summary_df['fold_size'] == fold]
            best_model = fold_data.loc[fold_data['mean_snr_improvement_db'].idxmax(), 'model']
            best_snr = fold_data['mean_snr_improvement_db'].max()
            print(f"  Fold {fold}: {best_model} (SNR improvement: {best_snr:.2f} dB)")

        print("\nOverall best performance:")
        best_overall = summary_df.loc[summary_df['mean_snr_improvement_db'].idxmax()]
        print(f"  Model: {best_overall['model']}")
        print(f"  Fold size: {best_overall['fold_size']}")
        print(f"  SNR improvement: {best_overall['mean_snr_improvement_db']:.2f} dB")

        print("\n" + "="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
