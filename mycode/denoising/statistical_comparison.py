#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Comparison Script for ECG Denoising Models

This script performs paired t-tests on RMSE and SNR differences between models.
Supports two modes:
1. compare_experiments: Compare the same model across 2+ experiments
2. compare_models: Compare all model pairs within a single experiment

Author: Basile Morel
Date: November 20, 2025
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate
import warnings

# Add paths FIRST before any local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '../../ecg_noise/source'))

from denoising_utils.utils import calculate_snr, calculate_rmse
from ecg_noise_factory.noise import NoiseFactory


def load_experiment_config(exp_folder):
    """
    Load the experiment's base config to get noise config path and sampling frequency.

    Args:
        exp_folder (str): Path to experiment folder

    Returns:
        dict: Configuration with keys 'noise_data_path', 'noise_config_path', and 'sampling_frequency'
    """
    config_path = os.path.join(exp_folder, 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}. Cannot proceed without experiment configuration.")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required keys
    required_keys = ['noise_data_path', 'noise_config_path', 'sampling_frequency']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required keys in config: {missing_keys}. Config path: {config_path}")

    # Construct absolute paths matching evaluate_similarity.py pattern
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    noise_data_path = os.path.join(repo_root, config['noise_data_path'])
    noise_config_path = os.path.join(repo_root, config['noise_config_path'])

    return {
        'noise_data_path': noise_data_path,
        'noise_config_path': noise_config_path,
        'sampling_frequency': config['sampling_frequency']
    }


def load_validation_data(exp_folder):
    """
    Load clean validation data from experiment folder.

    Args:
        exp_folder (str): Path to experiment folder

    Returns:
        np.ndarray: Clean validation data
    """
    data_path = os.path.join(exp_folder, 'data', 'clean_val.npy')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Validation data not found at {data_path}")

    return np.load(data_path)


def load_model_predictions(exp_folder, model_name):
    """
    Load model predictions from experiment folder.

    Args:
        exp_folder (str): Path to experiment folder
        model_name (str): Name of the model

    Returns:
        np.ndarray or None: Model predictions or None if not found
    """
    pred_path = os.path.join(exp_folder, 'models', model_name, 'predictions.npy')

    if not os.path.exists(pred_path):
        warnings.warn(f"Predictions not found at {pred_path}")
        return None

    return np.load(pred_path)


def get_available_models(exp_folder):
    """
    Scan experiment folder and return list of models that have predictions.

    Args:
        exp_folder (str): Path to experiment folder

    Returns:
        list: List of model names with available predictions
    """
    models_dir = os.path.join(exp_folder, 'models')

    if not os.path.exists(models_dir):
        warnings.warn(f"Models directory not found at {models_dir}")
        return []

    available_models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            pred_path = os.path.join(model_path, 'predictions.npy')
            if os.path.exists(pred_path):
                available_models.append(model_name)

    return sorted(available_models)


def compute_metrics_per_sample(clean_val, noisy_val, predictions, exp_folder=None, model_name=None):
    """
    Compute RMSE and SNR metrics for each sample in the validation set.

    Args:
        clean_val (np.ndarray): Clean validation data [n_samples, ...]
        noisy_val (np.ndarray): Noisy validation data [n_samples, ...]
        predictions (np.ndarray): Model predictions [n_samples, ...]
        exp_folder (str): Experiment folder path (for error messages)
        model_name (str): Model name (for error messages)

    Returns:
        tuple: (rmse_noisy, rmse_denoised, snr_input, snr_output) each shape [n_samples]
    """
    # Validate shapes match
    if predictions.shape[0] != clean_val.shape[0]:
        error_msg = f"Shape mismatch: predictions has {predictions.shape[0]} samples, clean_val has {clean_val.shape[0]} samples."
        if exp_folder and model_name:
            error_msg = f"Experiment '{exp_folder}', Model '{model_name}': {error_msg}"
        raise ValueError(error_msg)

    if noisy_val.shape[0] != clean_val.shape[0]:
        error_msg = f"Shape mismatch: noisy_val has {noisy_val.shape[0]} samples, clean_val has {clean_val.shape[0]} samples."
        if exp_folder:
            error_msg = f"Experiment '{exp_folder}': {error_msg}"
        raise ValueError(error_msg)

    n_samples = clean_val.shape[0]

    rmse_noisy = np.zeros(n_samples)
    rmse_denoised = np.zeros(n_samples)
    snr_input = np.zeros(n_samples)
    snr_output = np.zeros(n_samples)

    for i in range(n_samples):
        # Squeeze to remove singleton dimensions for consistent 1D shape
        clean = clean_val[i].squeeze()
        noisy = noisy_val[i].squeeze()
        denoised = predictions[i].squeeze()

        # Validate flattened lengths match
        if clean.size != noisy.size or clean.size != denoised.size:
            error_msg = f"Sample {i}: flattened length mismatch - clean: {clean.size}, noisy: {noisy.size}, denoised: {denoised.size}"
            if exp_folder and model_name:
                error_msg = f"Experiment '{exp_folder}', Model '{model_name}': {error_msg}"
            raise ValueError(error_msg)

        # RMSE metrics
        rmse_noisy[i] = calculate_rmse(clean, noisy)
        rmse_denoised[i] = calculate_rmse(clean, denoised)

        # SNR metrics
        snr_input[i] = calculate_snr(clean, noisy)
        snr_output[i] = calculate_snr(clean, denoised)

    return rmse_noisy, rmse_denoised, snr_input, snr_output


def perform_paired_ttest(metric1, metric2, metric_name):
    """
    Perform paired t-test on two metric arrays.

    Args:
        metric1 (np.ndarray): First metric array
        metric2 (np.ndarray): Second metric array
        metric_name (str): Name of the metric for reporting

    Returns:
        dict: Test results with keys: metric_name, mean_diff, std_diff,
              t_statistic, p_value, df, significant
    """
    # Compute differences
    diff = metric1 - metric2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(metric1, metric2, alternative='two-sided')

    # Determine significance at alpha=0.05
    significant = p_value < 0.05

    return {
        'metric_name': metric_name,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        't_statistic': t_statistic,
        'p_value': p_value,
        'df': len(metric1) - 1,
        'significant': significant
    }


def compare_experiments_mode(experiment_folders, output_dir, noise_config_path=None, sampling_rate=100):
    """
    Compare the same model across multiple experiments.

    Args:
        experiment_folders (list): List of experiment folder paths
        output_dir (str): Output directory for results
        noise_config_path (str): Optional override for noise config
        sampling_rate (int): Sampling rate for noise generation

    Returns:
        pd.DataFrame: Results dataframe
    """
    print("\n" + "="*80)
    print("MODE: COMPARE EXPERIMENTS")
    print("="*80)
    print(f"Comparing {len(experiment_folders)} experiments")
    print(f"Experiments: {experiment_folders}")
    print()

    # Get available models for each experiment
    exp_models = {}
    for exp_folder in experiment_folders:
        models = get_available_models(exp_folder)
        exp_models[exp_folder] = models
        print(f"Experiment '{os.path.basename(exp_folder)}': {len(models)} models")

    # Find common models across all experiments
    common_models = set(exp_models[experiment_folders[0]])
    for exp_folder in experiment_folders[1:]:
        common_models = common_models.intersection(set(exp_models[exp_folder]))

    common_models = sorted(list(common_models))
    print(f"\nCommon models across all experiments: {len(common_models)}")
    if len(common_models) == 0:
        raise ValueError("No common models found across all experiments")
    print(f"Models: {common_models}\n")

    # Results storage
    results = []

    # Process each common model
    for model_name in common_models:
        print(f"\nProcessing model: {model_name}")
        print("-" * 80)

        # Load data and compute metrics for each experiment
        exp_metrics = {}

        for exp_folder in experiment_folders:
            exp_name = os.path.basename(exp_folder)
            print(f"  Loading data from experiment: {exp_name}")

            # Load configuration
            config = load_experiment_config(exp_folder)
            # Apply CLI overrides if provided
            if noise_config_path:
                config['noise_config_path'] = noise_config_path
            if sampling_rate != 100:  # Only override if non-default value provided
                config['sampling_frequency'] = sampling_rate

            # Load validation data
            clean_val = load_validation_data(exp_folder)

            # Load model predictions
            predictions = load_model_predictions(exp_folder, model_name)
            if predictions is None:
                raise ValueError(f"Predictions not found for model '{model_name}' in experiment '{exp_name}'")

            # Validate prediction shape before generating noise
            if predictions.shape[0] != clean_val.shape[0]:
                error_msg = f"Model '{model_name}' in experiment '{exp_name}': predictions shape {predictions.shape} incompatible with clean_val shape {clean_val.shape}"
                raise ValueError(error_msg)

            # Generate noisy validation data using same pattern as evaluate_similarity.py
            noise_factory = NoiseFactory(
                data_path=config['noise_data_path'],
                sampling_rate=config['sampling_frequency'],
                config_path=config['noise_config_path'],
                mode='eval'  # Use evaluation noise samples (no leakage from training)
            )
            noisy_val = noise_factory.add_noise(
                x=clean_val.copy(), batch_axis=0, channel_axis=2, length_axis=1
            )

            # Validate noisy_val shape matches clean_val
            if noisy_val.shape != clean_val.shape:
                raise ValueError(f"Experiment '{exp_name}': noisy_val shape {noisy_val.shape} does not match clean_val shape {clean_val.shape}")

            # Compute per-sample metrics
            rmse_noisy, rmse_denoised, snr_input, snr_output = compute_metrics_per_sample(
                clean_val, noisy_val, predictions, exp_folder=exp_name, model_name=model_name
            )

            exp_metrics[exp_folder] = {
                'rmse_denoised': rmse_denoised,
                'snr_output': snr_output,
                'n_samples': len(rmse_denoised)
            }

            print(f"    Samples: {exp_metrics[exp_folder]['n_samples']}")
            print(f"    Mean RMSE: {np.mean(rmse_denoised):.6f}")
            print(f"    Mean SNR: {np.mean(snr_output):.4f} dB")

        # Validate consistent sample counts across all experiments for this model
        sample_counts = {exp: exp_metrics[exp]['n_samples'] for exp in experiment_folders}
        unique_counts = set(sample_counts.values())
        if len(unique_counts) > 1:
            count_str = ", ".join([f"{os.path.basename(exp)}: {count}" for exp, count in sample_counts.items()])
            warnings.warn(f"Model '{model_name}': inconsistent sample counts across experiments ({count_str}). Skipping this model.")
            continue

        # Compare each pair of experiments
        print(f"\n  Performing pairwise comparisons:")
        for i in range(len(experiment_folders)):
            for j in range(i + 1, len(experiment_folders)):
                exp1 = experiment_folders[i]
                exp2 = experiment_folders[j]
                exp1_name = os.path.basename(exp1)
                exp2_name = os.path.basename(exp2)

                print(f"\n    Comparing: {exp1_name} vs {exp2_name}")

                # Get metrics
                rmse1 = exp_metrics[exp1]['rmse_denoised']
                rmse2 = exp_metrics[exp2]['rmse_denoised']
                snr1 = exp_metrics[exp1]['snr_output']
                snr2 = exp_metrics[exp2]['snr_output']

                # Perform paired t-tests
                rmse_test = perform_paired_ttest(rmse1, rmse2, 'RMSE')
                snr_test = perform_paired_ttest(snr1, snr2, 'SNR')

                # Store results
                results.append({
                    'model': model_name,
                    'exp1': exp1_name,
                    'exp2': exp2_name,
                    'rmse_mean_diff': rmse_test['mean_diff'],
                    'rmse_std_diff': rmse_test['std_diff'],
                    'rmse_t_statistic': rmse_test['t_statistic'],
                    'rmse_p_value': rmse_test['p_value'],
                    'rmse_significant': rmse_test['significant'],
                    'snr_mean_diff': snr_test['mean_diff'],
                    'snr_std_diff': snr_test['std_diff'],
                    'snr_t_statistic': snr_test['t_statistic'],
                    'snr_p_value': snr_test['p_value'],
                    'snr_significant': snr_test['significant'],
                    'n_samples': len(rmse1)
                })

                print(f"      RMSE: mean_diff={rmse_test['mean_diff']:.6f}, p={rmse_test['p_value']:.4f}, sig={rmse_test['significant']}")
                print(f"      SNR:  mean_diff={snr_test['mean_diff']:.4f} dB, p={snr_test['p_value']:.4f}, sig={snr_test['significant']}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'experiment_comparison_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")

    # Print formatted table
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON RESULTS")
    print("="*80)

    # Prepare table for display
    display_df = results_df[['model', 'exp1', 'exp2', 'rmse_mean_diff', 'rmse_p_value',
                               'rmse_significant', 'snr_mean_diff', 'snr_p_value', 'snr_significant']].copy()
    display_df['rmse_mean_diff'] = display_df['rmse_mean_diff'].apply(lambda x: f"{x:.6f}")
    display_df['rmse_p_value'] = display_df['rmse_p_value'].apply(lambda x: f"{x:.4f}")
    display_df['snr_mean_diff'] = display_df['snr_mean_diff'].apply(lambda x: f"{x:.4f}")
    display_df['snr_p_value'] = display_df['snr_p_value'].apply(lambda x: f"{x:.4f}")
    display_df['rmse_significant'] = display_df['rmse_significant'].apply(lambda x: "✓" if x else "✗")
    display_df['snr_significant'] = display_df['snr_significant'].apply(lambda x: "✓" if x else "✗")

    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))

    # Print interpretation notes
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("• p < 0.05 indicates significant difference (marked with ✓)")
    print("• Positive RMSE diff means exp1 has higher RMSE (worse performance)")
    print("• Positive SNR diff means exp1 has higher SNR (better performance)")
    print("• RMSE: Root Mean Square Error (lower is better)")
    print("• SNR: Signal-to-Noise Ratio in dB (higher is better)")
    print("="*80 + "\n")

    return results_df


def compare_models_mode(experiment_folder, output_dir, noise_config_path=None, sampling_rate=100):
    """
    Compare all model pairs within a single experiment.

    Args:
        experiment_folder (str): Path to experiment folder
        output_dir (str): Output directory for results
        noise_config_path (str): Optional override for noise config
        sampling_rate (int): Sampling rate for noise generation

    Returns:
        pd.DataFrame: Results dataframe
    """
    print("\n" + "="*80)
    print("MODE: COMPARE MODELS")
    print("="*80)
    print(f"Experiment: {experiment_folder}")
    print()

    # Get available models
    models = get_available_models(experiment_folder)
    print(f"Available models: {len(models)}")
    if len(models) < 2:
        raise ValueError(f"Need at least 2 models for comparison, found {len(models)}")
    print(f"Models: {models}\n")

    # Load configuration
    config = load_experiment_config(experiment_folder)
    # Apply CLI overrides if provided
    if noise_config_path:
        config['noise_config_path'] = noise_config_path
    if sampling_rate != 100:  # Only override if non-default value provided
        config['sampling_frequency'] = sampling_rate

    # Load validation data once
    print("Loading validation data...")
    clean_val = load_validation_data(experiment_folder)
    print(f"Validation samples: {clean_val.shape[0]}")
    print(f"Validation shape: {clean_val.shape}")

    # Generate noisy validation data once using same pattern as evaluate_similarity.py
    print("Generating noisy validation data...")
    noise_factory = NoiseFactory(
        data_path=config['noise_data_path'],
        sampling_rate=config['sampling_frequency'],
        config_path=config['noise_config_path'],
        mode='eval'  # Use evaluation noise samples (no leakage from training)
    )
    noisy_val = noise_factory.add_noise(
        x=clean_val.copy(), batch_axis=0, channel_axis=2, length_axis=1
    )

    # Validate noisy_val shape matches clean_val
    if noisy_val.shape != clean_val.shape:
        raise ValueError(f"noisy_val shape {noisy_val.shape} does not match clean_val shape {clean_val.shape}")
    print(f"Noisy validation shape: {noisy_val.shape}")

    # Load predictions and compute metrics for each model
    print("\nComputing metrics for each model...")
    model_metrics = {}

    for model_name in models:
        print(f"  Processing model: {model_name}")

        # Load predictions
        predictions = load_model_predictions(experiment_folder, model_name)
        if predictions is None:
            warnings.warn(f"Skipping model '{model_name}' - predictions not found")
            continue

        # Validate prediction shape
        if predictions.shape[0] != clean_val.shape[0]:
            warnings.warn(f"Skipping model '{model_name}' - shape mismatch: predictions {predictions.shape} vs clean_val {clean_val.shape}")
            continue

        # Compute per-sample metrics
        try:
            rmse_noisy, rmse_denoised, snr_input, snr_output = compute_metrics_per_sample(
                clean_val, noisy_val, predictions, exp_folder=experiment_folder, model_name=model_name
            )
        except ValueError as e:
            warnings.warn(f"Skipping model '{model_name}' - metric computation failed: {e}")
            continue

        model_metrics[model_name] = {
            'rmse_denoised': rmse_denoised,
            'snr_output': snr_output
        }

        print(f"    Mean RMSE: {np.mean(rmse_denoised):.6f}")
        print(f"    Mean SNR: {np.mean(snr_output):.4f} dB")

    # Compare each pair of models
    print("\n\nPerforming pairwise model comparisons...")
    results = []

    model_list = sorted(model_metrics.keys())
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            model1 = model_list[i]
            model2 = model_list[j]

            print(f"\n  Comparing: {model1} vs {model2}")

            # Get metrics
            rmse1 = model_metrics[model1]['rmse_denoised']
            rmse2 = model_metrics[model2]['rmse_denoised']
            snr1 = model_metrics[model1]['snr_output']
            snr2 = model_metrics[model2]['snr_output']

            # Perform paired t-tests
            rmse_test = perform_paired_ttest(rmse1, rmse2, 'RMSE')
            snr_test = perform_paired_ttest(snr1, snr2, 'SNR')

            # Store results
            results.append({
                'model1': model1,
                'model2': model2,
                'rmse_mean_diff': rmse_test['mean_diff'],
                'rmse_std_diff': rmse_test['std_diff'],
                'rmse_t_statistic': rmse_test['t_statistic'],
                'rmse_p_value': rmse_test['p_value'],
                'rmse_significant': rmse_test['significant'],
                'snr_mean_diff': snr_test['mean_diff'],
                'snr_std_diff': snr_test['std_diff'],
                'snr_t_statistic': snr_test['t_statistic'],
                'snr_p_value': snr_test['p_value'],
                'snr_significant': snr_test['significant'],
                'n_samples': len(rmse1)
            })

            print(f"    RMSE: mean_diff={rmse_test['mean_diff']:.6f}, p={rmse_test['p_value']:.4f}, sig={rmse_test['significant']}")
            print(f"    SNR:  mean_diff={snr_test['mean_diff']:.4f} dB, p={snr_test['p_value']:.4f}, sig={snr_test['significant']}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'model_comparison_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")

    # Print formatted table
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)

    # Prepare table for display
    display_df = results_df[['model1', 'model2', 'rmse_mean_diff', 'rmse_p_value',
                               'rmse_significant', 'snr_mean_diff', 'snr_p_value', 'snr_significant']].copy()
    display_df['rmse_mean_diff'] = display_df['rmse_mean_diff'].apply(lambda x: f"{x:.6f}")
    display_df['rmse_p_value'] = display_df['rmse_p_value'].apply(lambda x: f"{x:.4f}")
    display_df['snr_mean_diff'] = display_df['snr_mean_diff'].apply(lambda x: f"{x:.4f}")
    display_df['snr_p_value'] = display_df['snr_p_value'].apply(lambda x: f"{x:.4f}")
    display_df['rmse_significant'] = display_df['rmse_significant'].apply(lambda x: "✓" if x else "✗")
    display_df['snr_significant'] = display_df['snr_significant'].apply(lambda x: "✓" if x else "✗")

    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))

    # Print interpretation notes
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("• p < 0.05 indicates significant difference (marked with ✓)")
    print("• Positive RMSE diff means model1 has higher RMSE (worse performance)")
    print("• Positive SNR diff means model1 has higher SNR (better performance)")
    print("• RMSE: Root Mean Square Error (lower is better)")
    print("• SNR: Signal-to-Noise Ratio in dB (higher is better)")
    print("="*80 + "\n")

    return results_df


def main():
    """Main function with argument parsing and mode selection."""
    parser = argparse.ArgumentParser(
        description='Statistical Comparison of ECG Denoising Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare same model across experiments
  python statistical_comparison.py --mode compare_experiments \\
      --experiments exp1/ exp2/ exp3/ --output results/

  # Compare all models within one experiment
  python statistical_comparison.py --mode compare_models \\
      --experiments exp1/ --output results/
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['compare_experiments', 'compare_models'],
        help='Comparison mode: compare_experiments or compare_models'
    )

    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        required=True,
        help='List of experiment folder paths (2+ for compare_experiments, 1 for compare_models)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./statistical_comparison_results',
        help='Output directory for results (default: ./statistical_comparison_results)'
    )

    parser.add_argument(
        '--noise-config',
        type=str,
        default=None,
        help='Optional override for noise config path'
    )

    parser.add_argument(
        '--sampling-rate',
        type=int,
        default=100,
        help='Sampling rate for noise generation (default: 100 Hz)'
    )

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == 'compare_experiments':
        if len(args.experiments) < 2:
            parser.error("compare_experiments mode requires at least 2 experiment folders")

        # Check if all experiment folders exist
        for exp_folder in args.experiments:
            if not os.path.exists(exp_folder):
                parser.error(f"Experiment folder not found: {exp_folder}")

        # Run comparison
        results_df = compare_experiments_mode(
            args.experiments,
            args.output,
            noise_config_path=args.noise_config,
            sampling_rate=args.sampling_rate
        )

    elif args.mode == 'compare_models':
        if len(args.experiments) != 1:
            parser.error("compare_models mode requires exactly 1 experiment folder")

        experiment_folder = args.experiments[0]
        if not os.path.exists(experiment_folder):
            parser.error(f"Experiment folder not found: {experiment_folder}")

        # Run comparison
        results_df = compare_models_mode(
            experiment_folder,
            args.output,
            noise_config_path=args.noise_config,
            sampling_rate=args.sampling_rate
        )

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    if args.mode == 'compare_experiments':
        total_comparisons = len(results_df)
        rmse_sig_count = results_df['rmse_significant'].sum()
        snr_sig_count = results_df['snr_significant'].sum()

        print(f"Total comparisons: {total_comparisons}")
        print(f"Significant RMSE differences: {rmse_sig_count} ({100*rmse_sig_count/total_comparisons:.1f}%)")
        print(f"Significant SNR differences: {snr_sig_count} ({100*snr_sig_count/total_comparisons:.1f}%)")

    elif args.mode == 'compare_models':
        total_comparisons = len(results_df)
        rmse_sig_count = results_df['rmse_significant'].sum()
        snr_sig_count = results_df['snr_significant'].sum()

        print(f"Total model pairs compared: {total_comparisons}")
        print(f"Significant RMSE differences: {rmse_sig_count} ({100*rmse_sig_count/total_comparisons:.1f}%)")
        print(f"Significant SNR differences: {snr_sig_count} ({100*snr_sig_count/total_comparisons:.1f}%)")

    print("="*80 + "\n")

    print("Analysis complete!")


if __name__ == '__main__':
    main()
