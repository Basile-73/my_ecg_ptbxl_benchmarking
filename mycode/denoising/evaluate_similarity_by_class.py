"""
Evaluate ECG Denoising Models by Diagnostic Class

This script computes SNR and RMSE metrics grouped by diagnostic classes
(diagnostic, subdiagnostic, superdiagnostic) with bootstrap confidence intervals.

Key Features:
- Handles multi-label classification (samples can belong to multiple classes)
- Computes bootstrap confidence intervals (90% CI using 5th/95th percentiles)
- Generates per-class performance metrics for all models
- Saves results as CSV files for downstream analysis
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '../'))
sys.path.insert(0, os.path.join(script_dir, '../classification'))
sys.path.insert(0, os.path.join(script_dir, '../../ecg_noise/source'))

# Import required modules
from denoising_utils.utils import calculate_snr, calculate_rmse
from denoising_utils.preprocessing import remove_bad_labels_labels_only
from utils.utils import compute_label_aggregations, load_labels_only

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


def evaluate_model_by_class(model_name, predictions, clean_test,
                            test_labels_dict, bootstrap_samples):
    """
    Evaluate a single model's performance grouped by diagnostic classes.

    Args:
        model_name: Name of the model
        predictions: Model predictions array
        clean_test: Clean test signals
        test_labels_dict: Dictionary with keys 'diagnostic', 'subdiagnostic', 'superdiagnostic'
                         containing respective label DataFrames
        bootstrap_samples: List of bootstrap index arrays

    Returns:
        Dictionary mapping class_type -> DataFrame of results
    """
    print(f"\nEvaluating {model_name}...")

    # Compute per-sample metrics
    n_samples = len(clean_test)
    snr_per_sample = np.zeros(n_samples)
    rmse_per_sample = np.zeros(n_samples)

    print("  Computing per-sample metrics...")
    for i in tqdm(range(n_samples), desc="  Samples"):
        clean_signal = clean_test[i].squeeze()
        pred_signal = predictions[i].squeeze()

        snr_per_sample[i] = calculate_snr(clean_signal, pred_signal)
        rmse_per_sample[i] = calculate_rmse(clean_signal, pred_signal)

    # Evaluate for each class type
    results_dict = {}

    for class_type in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
        print(f"  Processing {class_type} classes...")

        test_labels = test_labels_dict[class_type]

        # Get all unique classes from the multi-label column
        all_classes = set()
        for sample_labels in test_labels[class_type]:
            if isinstance(sample_labels, list):
                all_classes.update(sample_labels)

        all_classes = sorted(list(all_classes))

        # Compute metrics for each class
        class_results = []

        for class_name in tqdm(all_classes, desc=f"  {class_type} classes"):
            # Create boolean mask for samples belonging to this class
            class_mask = np.array([
                class_name in sample_labels if isinstance(sample_labels, list) else False
                for sample_labels in test_labels[class_type]
            ])

            n_class_samples = class_mask.sum()

            # Skip classes with very few samples
            if n_class_samples < 5:
                print(f"    Warning: {class_name} has only {n_class_samples} samples (skipping)")
                continue

            # Compute bootstrap statistics
            snr_mean, snr_lower, snr_upper = compute_bootstrap_stats(
                snr_per_sample, bootstrap_samples, class_mask
            )
            rmse_mean, rmse_lower, rmse_upper = compute_bootstrap_stats(
                rmse_per_sample, bootstrap_samples, class_mask
            )

            class_results.append({
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

        results_dict[class_type] = pd.DataFrame(class_results)

    return results_dict


def evaluate_noisy_input_by_class(clean_test, noisy_test, test_labels_dict, bootstrap_samples):
    """
    Evaluate noisy input as a baseline by computing per-class metrics against clean ground truth.

    Args:
        clean_test: Clean test data
        noisy_test: Noisy test data
        test_labels_dict: Dictionary mapping class types to label DataFrames
        bootstrap_samples: Bootstrap sample indices for confidence intervals

    Returns:
        Dictionary mapping class_type -> DataFrame with noisy input metrics
    """
    print("\nComputing noisy input baseline metrics by class...")
    n_samples = len(clean_test)

    # Compute per-sample metrics
    snr_per_sample = np.zeros(n_samples)
    rmse_per_sample = np.zeros(n_samples)

    print("  Computing per-sample metrics...")
    for i in tqdm(range(n_samples), desc="  Samples"):
        clean_signal = clean_test[i].squeeze()
        noisy_signal = noisy_test[i].squeeze()

        snr_per_sample[i] = calculate_snr(clean_signal, noisy_signal)
        rmse_per_sample[i] = calculate_rmse(clean_signal, noisy_signal)

    # Evaluate for each class type
    results_dict = {}

    for class_type in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
        print(f"  Processing {class_type} classes...")

        test_labels = test_labels_dict[class_type]

        # Get all unique classes from the multi-label column
        all_classes = set()
        for sample_labels in test_labels[class_type]:
            if isinstance(sample_labels, list):
                all_classes.update(sample_labels)

        all_classes = sorted(list(all_classes))

        # Compute metrics for each class
        class_results = []

        for class_name in tqdm(all_classes, desc=f"  {class_type} classes"):
            # Create boolean mask for samples belonging to this class
            class_mask = np.array([
                class_name in sample_labels if isinstance(sample_labels, list) else False
                for sample_labels in test_labels[class_type]
            ])

            n_class_samples = class_mask.sum()

            # Skip classes with very few samples
            if n_class_samples < 5:
                print(f"    Warning: {class_name} has only {n_class_samples} samples (skipping)")
                continue

            # Compute bootstrap statistics
            snr_mean, snr_lower, snr_upper = compute_bootstrap_stats(
                snr_per_sample, bootstrap_samples, class_mask
            )
            rmse_mean, rmse_lower, rmse_upper = compute_bootstrap_stats(
                rmse_per_sample, bootstrap_samples, class_mask
            )

            class_results.append({
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

        results_dict[class_type] = pd.DataFrame(class_results)

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

    # Generate bootstrap samples
    print(f"\nGenerating {n_bootstrap} bootstrap samples...")
    np.random.seed(config.get('random_seed', 42))
    y_dummy = np.ones((len(clean_test), 1))
    bootstrap_samples = get_appropriate_bootstrap_samples(y_dummy, n_bootstrap)
    print("  ✓ Bootstrap samples generated")

    # Evaluate each model
    print("\n" + "="*80)
    print("EVALUATING MODELS")
    print("="*80)

    all_results = {
        'diagnostic': [],
        'subdiagnostic': [],
        'superdiagnostic': []
    }

    # Evaluate noisy input baseline
    print("\nEvaluating noisy input baseline...")
    noisy_input_results = evaluate_noisy_input_by_class(
        clean_test, noisy_test, test_labels_dict, bootstrap_samples
    )

    # Add noisy input results first
    for class_type in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
        all_results[class_type].append(noisy_input_results[class_type])

    for model_config in config['models']:
        model_name = model_config['name']

        # Load predictions
        pred_path = os.path.join(exp_folder, 'models', model_name, 'predictions.npy')

        if not os.path.exists(pred_path):
            print(f"\n⚠️  Predictions not found for {model_name}, skipping...")
            print(f"    Expected: {pred_path}")
            continue

        predictions = np.load(pred_path)
        print(f"\n✓ Loaded predictions for {model_name}")
        print(f"  Shape: {predictions.shape}")

        # Evaluate by class
        model_results = evaluate_model_by_class(
            model_name, predictions, clean_test,
            test_labels_dict, bootstrap_samples
        )

        # Accumulate results
        for class_type in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
            all_results[class_type].append(model_results[class_type])

    # Combine and save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results_folder = os.path.join(exp_folder, 'results')
    os.makedirs(results_folder, exist_ok=True)

    for class_type in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
        if all_results[class_type]:
            # Combine all models
            combined_df = pd.concat(all_results[class_type], ignore_index=True)

            # Save to CSV
            output_path = os.path.join(results_folder, f'metrics_by_{class_type}.csv')
            combined_df.to_csv(output_path, index=False)
            print(f"\n✓ Saved: {output_path}")
            print(f"  Rows: {len(combined_df)}")

            # Print summary table
            print(f"\n{class_type.upper()} SUMMARY:")
            print("-" * 80)

            # Group by model and show top 5 classes by sample count
            for model in combined_df['model'].unique():
                model_df = combined_df[combined_df['model'] == model]
                top_classes = model_df.nlargest(5, 'n_samples')

                print(f"\n{model}:")
                print(f"  {'Class':<20} {'N':<8} {'SNR (dB)':<20} {'RMSE':<20}")
                print(f"  {'-'*20} {'-'*8} {'-'*20} {'-'*20}")

                for _, row in top_classes.iterrows():
                    snr_str = f"{row['snr_mean']:.2f} [{row['snr_lower_ci']:.2f}, {row['snr_upper_ci']:.2f}]"
                    rmse_str = f"{row['rmse_mean']:.4f} [{row['rmse_lower_ci']:.4f}, {row['rmse_upper_ci']:.4f}]"
                    print(f"  {row['class']:<20} {row['n_samples']:<8} {snr_str:<20} {rmse_str:<20}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_folder}")
    print("\nGenerated files:")
    print("  - metrics_by_diagnostic.csv")
    print("  - metrics_by_subdiagnostic.csv")
    print("  - metrics_by_superdiagnostic.csv")


if __name__ == '__main__':
    main()
