"""
Qui Plot: Multi-noise configuration comparison visualization.

This module provides functionality to compare model performance across different
noise configurations with confidence intervals.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import repeat

from .utils import get_model, run_denoise_inference
from ecg_noise_factory.noise import NoiseFactory


def calculate_snr(clean, noisy):
    """Calculate Signal-to-Noise Ratio in dB."""
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum((clean - noisy) ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_rmse(clean, noisy):
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((clean - noisy) ** 2))


def get_appropriate_bootstrap_samples(y_true, n_bootstraping_samples):
    """Generate bootstrap sample indices ensuring all classes are represented.

    Args:
        y_true: Array of true labels or metrics (N,) or (N, classes)
        n_bootstraping_samples: Number of bootstrap samples to generate

    Returns:
        List of bootstrap sample indices
    """
    samples = []
    while True:
        ridxs = np.random.randint(0, len(y_true), len(y_true))
        # If y_true has multiple dimensions, check that all have at least one sample
        if len(y_true.shape) > 1:
            if y_true[ridxs].sum(axis=0).min() != 0:
                samples.append(ridxs)
        else:
            # For 1D arrays, just ensure we have some variation
            samples.append(ridxs)

        if len(samples) == n_bootstraping_samples:
            break
    return samples


def compute_bootstrap_metrics(sample_indices, df, metrics=['rmse_denoised', 'output_snr_db']):
    """Compute metrics for a single bootstrap sample.

    Args:
        sample_indices: Indices for bootstrap sample
        df: DataFrame with sample-level metrics
        metrics: List of metric column names to compute

    Returns:
        Dictionary with mean values for each metric
    """
    sample_df = df.iloc[sample_indices]
    result = {}
    for metric in metrics:
        result[metric] = sample_df[metric].mean()
    return result


def qui_plot(noise_configs, exp_folder, config, clean_test, models, n_bootstrap_samples=100):
    """Create grouped bar chart comparing models across different noise configurations.

    Groups are organized by noise config first, then models within each config.
    Regenerates predictions with each noise configuration.

    Args:
        noise_configs: List of dicts with 'name' and 'path' keys for each noise config
                      Example: [{'name': 'light', 'path': '../noise/config/light.yaml'},
                               {'name': 'default', 'path': '../noise/config/default.yaml'}]
        exp_folder: Experiment folder path
        config: Main config dictionary
        clean_test: Clean test signals
        models: List of model names to evaluate
        n_bootstrap_samples: Number of bootstrap samples for confidence intervals (default: 100)
    """
    # Color mapping for models (Stage1 = light, Stage2 = dark)
    color_map = {
        'fcn': '#aec7e8',         # Light blue (Stage1)
        'drnet_fcn': '#1f77b4',   # Dark blue (Stage2)
        'unet': '#ff9896',        # Light red (Stage1)
        'drnet_unet': '#d62728',  # Dark red (Stage2)
        'imunet': '#98df8a',      # Light green (Stage1)
        'drnet_imunet': '#2ca02c', # Dark green (Stage2)
        'imunet_origin': '#9467bd',    # Purple
        'imunet_mamba_bn': '#ff7f0e',  # Orange
        'imunet_mamba_up': '#17becf',  # Cyan/Teal
        'imunet_mamba_early': '#e377c2', # Magenta/Pink
        'imunet_mamba_late': '#bcbd22'  # Yellow-green
    }

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and
                         config['hardware']['use_cuda'] else 'cpu')

    # Collect results for each noise config
    all_results = []

    print("\n" + "="*80)
    print("QUI PLOT: Evaluating models across noise configurations")
    print("="*80)

    # Progress bar for noise configs
    for noise_config in tqdm(noise_configs, desc="Processing noise configs", position=0):
        config_name = noise_config['name']
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            noise_config['path']
        )

        print(f"\n--- Noise Config: {config_name} ---")
        print(f"Path: {config_path}")

        # Initialize NoiseFactory with this config
        noise_data_path = os.path.join(os.path.dirname(__file__), '../../../ecg_noise/data')
        full_config_path = os.path.join(os.path.dirname(__file__), '..', config_path)

        noise_factory = NoiseFactory(
            data_path=noise_data_path,
            sampling_rate=config['sampling_frequency'],
            config_path=full_config_path,
            mode='eval'  # Use eval mode for fair comparison
        )

        # Generate noisy test data with this noise config
        noisy_test = noise_factory.add_noise(
            x=clean_test, batch_axis=0, channel_axis=2, length_axis=1
        )

        # Evaluate each model
        for model_name in tqdm(models, desc=f"  Evaluating models ({config_name})", position=1, leave=False):
            # Initialize variables
            model = None
            stage1_model = None
            stage1_predictions = None

            # Load model
            model_folder = os.path.join(exp_folder, 'models', model_name)
            model_path = os.path.join(model_folder, 'best_model.pth')

            if not os.path.exists(model_path):
                tqdm.write(f"    Warning: Model not found at {model_path}")
                continue

            # Determine model type
            model_config = next((m for m in config['models'] if m['name'] == model_name), None)
            if not model_config:
                tqdm.write(f"    Warning: Model config not found for {model_name}")
                continue

            model_type = model_config['type']
            is_stage2 = model_type.lower() in ['stage2', 'drnet']

            # For Stage2 models, we need to load the Stage1 model first
            if is_stage2:
                # Get Stage1 model name
                stage1_model_name = model_config.get('stage1_model')
                if not stage1_model_name:
                    tqdm.write(f"    Warning: Stage2 model {model_name} missing stage1_model config")
                    continue

                # Load Stage1 model
                stage1_folder = os.path.join(exp_folder, 'models', stage1_model_name)
                stage1_path = os.path.join(stage1_folder, 'best_model.pth')

                if not os.path.exists(stage1_path):
                    tqdm.write(f"    Warning: Stage1 model not found at {stage1_path}")
                    continue

                # Get Stage1 model config
                stage1_config = next((m for m in config['models'] if m['name'] == stage1_model_name), None)
                if not stage1_config:
                    tqdm.write(f"    Warning: Stage1 model config not found for {stage1_model_name}")
                    continue

                # Load and run Stage1 model
                stage1_model = get_model(stage1_config['type'], input_length=clean_test[0].shape[0], is_stage2=False)
                stage1_model.load_state_dict(torch.load(stage1_path, map_location=device))
                stage1_model = stage1_model.to(device)
                stage1_model.eval()

                # Detect if Stage1 is MECGE model
                is_stage1_mecge = hasattr(stage1_model, 'denoising')
                if is_stage1_mecge:
                    tqdm.write(f"    Using MECGE denoising method for Stage1 ({stage1_config['type']})")
                else:
                    tqdm.write(f"    Using standard forward pass for Stage1 ({stage1_config['type']})")

                # Generate Stage1 predictions
                stage1_predictions = []
                with torch.no_grad():
                    for i in tqdm(range(len(clean_test)), desc=f"    Stage1 predictions", position=2, leave=False):
                        noisy_sample = noisy_test[i]
                        noisy_sample = torch.FloatTensor(noisy_sample).permute(1, 0).unsqueeze(0).unsqueeze(0)
                        noisy_sample = noisy_sample.to(device)

                        # Use helper for inference (handles MECGE vs standard)
                        stage1_pred = run_denoise_inference(stage1_model, noisy_sample, is_stage2=False)
                        stage1_predictions.append(stage1_pred)

                stage1_predictions = np.array(stage1_predictions)

                # Clean up Stage1 model
                del stage1_model
                torch.cuda.empty_cache()

            # Load Stage2 model
            model = get_model(model_type, input_length=clean_test[0].shape[0], is_stage2=is_stage2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()

            # Detect if main model is MECGE
            is_mecge = hasattr(model, 'denoising')

            # Guard: Stage2 MECGE is unsupported (would pass invalid shape B,2,1,T)
            if is_stage2 and is_mecge:
                tqdm.write(f"    WARNING: Stage2 MECGE is unsupported, using standard forward pass for {model_name}")
            elif is_mecge:
                tqdm.write(f"    Using MECGE denoising method for {model_name}")
            else:
                tqdm.write(f"    Using standard forward pass for {model_name}")

            # Generate predictions
            predictions = []
            with torch.no_grad():
                desc_text = f"    {model_name} predictions"
                for i in tqdm(range(len(clean_test)), desc=desc_text, position=2, leave=False):
                    # Convert to correct shape: (time, channels) -> (1, 1, channels, time)
                    noisy_sample = noisy_test[i]  # Shape: (time, channels)
                    noisy_sample = torch.FloatTensor(noisy_sample).permute(1, 0).unsqueeze(0).unsqueeze(0)  # (1, 1, channels, time)

                    if is_stage2:
                        # For Stage2: concatenate noisy + stage1_pred to create 2-channel input
                        stage1_sample = stage1_predictions[i]
                        stage1_sample = torch.FloatTensor(stage1_sample).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, time)
                        model_input = torch.cat([noisy_sample, stage1_sample], dim=1)  # (1, 2, 1, time)
                    else:
                        # For Stage1: single channel input
                        model_input = noisy_sample

                    model_input = model_input.to(device)

                    # Use helper for inference (handles MECGE vs standard with Stage2 guard)
                    pred = run_denoise_inference(model, model_input, is_stage2=is_stage2)
                    predictions.append(pred)

            predictions = np.array(predictions)

            # Calculate metrics for each sample
            for i in range(len(clean_test)):
                clean = clean_test[i].squeeze()
                noisy = noisy_test[i].squeeze()
                denoised = predictions[i]

                # Calculate SNR
                input_snr = calculate_snr(clean, noisy)
                output_snr = calculate_snr(clean, denoised)
                snr_improvement = output_snr - input_snr

                # Calculate RMSE
                rmse_noisy = calculate_rmse(clean, noisy)
                rmse_denoised = calculate_rmse(clean, denoised)

                all_results.append({
                    'noise_config': config_name,
                    'model': model_name,
                    'sample_idx': i,
                    'snr_improvement_db': snr_improvement,
                    'rmse_denoised': rmse_denoised,
                    'input_snr_db': input_snr,
                    'output_snr_db': output_snr
                })

            # Clean up model
            if model is not None:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if not all_results:
        print("Warning: No results collected for qui_plot")
        return

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Generate bootstrap samples and compute confidence intervals
    print("\nComputing bootstrap confidence intervals...")
    summary_stats = []

    for noise_config in noise_configs:
        config_name = noise_config['name']
        config_df = results_df[results_df['noise_config'] == config_name]

        for model_name in models:
            model_df = config_df[config_df['model'] == model_name]

            if not model_df.empty:
                # Create simple array for bootstrap sampling (just need the indices)
                y_dummy = np.ones((len(model_df), 1))
                bootstrap_samples = get_appropriate_bootstrap_samples(y_dummy, n_bootstrap_samples)

                # Compute metrics for each bootstrap sample
                bootstrap_results = []
                for sample_indices in bootstrap_samples:
                    metrics = compute_bootstrap_metrics(
                        sample_indices,
                        model_df.reset_index(drop=True),
                        metrics=['rmse_denoised', 'output_snr_db']
                    )
                    bootstrap_results.append(metrics)

                # Convert to DataFrame for easy quantile computation
                bootstrap_df = pd.DataFrame(bootstrap_results)

                # Point estimate (using all data)
                point_rmse = model_df['rmse_denoised'].mean()
                point_snr = model_df['output_snr_db'].mean()

                # Bootstrap mean and quantiles
                mean_rmse = bootstrap_df['rmse_denoised'].mean()
                lower_rmse = bootstrap_df['rmse_denoised'].quantile(0.05)
                upper_rmse = bootstrap_df['rmse_denoised'].quantile(0.95)

                mean_snr = bootstrap_df['output_snr_db'].mean()
                lower_snr = bootstrap_df['output_snr_db'].quantile(0.05)
                upper_snr = bootstrap_df['output_snr_db'].quantile(0.95)

                summary_stats.append({
                    'noise_config': config_name,
                    'model': model_name,
                    'point_rmse': point_rmse,
                    'mean_rmse': mean_rmse,
                    'lower_rmse': lower_rmse,
                    'upper_rmse': upper_rmse,
                    'point_output_snr': point_snr,
                    'mean_output_snr': mean_snr,
                    'lower_output_snr': lower_snr,
                    'upper_output_snr': upper_snr
                })

    summary_df = pd.DataFrame(summary_stats)

    # Create figure - Grouped by NOISE CONFIG first
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Performance Across Noise Configurations',
                fontsize=16, fontweight='bold', y=0.98)

    # Group models by base (fcn, unet, imunet) and sort
    def get_model_order(model_name):
        """Return (base_name, is_stage2) for sorting"""
        if 'drnet_' in model_name:
            base = model_name.replace('drnet_', '')
            return (base, 1)  # Stage2 comes after Stage1
        else:
            return (model_name, 0)

    sorted_models = sorted(models, key=get_model_order)

    # Setup for grouped bar chart
    n_models = len(sorted_models)
    n_configs = len(noise_configs)
    group_width = 0.8
    bar_width = group_width / n_models

    config_positions = []
    config_labels = []

    # Plot 1: RMSE
    ax1 = axes[0]
    x_pos = 0
    max_rmse_y = 0
    min_rmse_y = float('inf')

    for config_idx, noise_config in enumerate(noise_configs):
        config_name = noise_config['name']
        config_df = summary_df[summary_df['noise_config'] == config_name]

        for model_idx, model_name in enumerate(sorted_models):
            model_df = config_df[config_df['model'] == model_name]

            if not model_df.empty:
                point_rmse = model_df['point_rmse'].values[0]
                lower_rmse = model_df['lower_rmse'].values[0]
                upper_rmse = model_df['upper_rmse'].values[0]

                # Compute error bars (asymmetric)
                lower_err = point_rmse - lower_rmse
                upper_err = upper_rmse - point_rmse

                color = color_map.get(model_name, '#cccccc')

                bar_pos = x_pos + model_idx * bar_width
                ax1.bar(bar_pos, point_rmse, bar_width,
                       yerr=[[lower_err], [upper_err]], capsize=3,
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
                       label=model_name if config_idx == 0 else '')

                # Add text label on top of bar
                ax1.text(bar_pos, point_rmse + upper_err + 0.001, f'{point_rmse:.3f}',
                        ha='center', va='bottom', fontsize=8, color='black',
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
    ax1.set_ylabel('Mean RMSE (Denoised)', fontsize=12, fontweight='bold')
    ax1.set_title('RMSE Comparison (90% Bootstrap CI)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(config_positions)
    ax1.set_xticklabels(config_labels, fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=min_rmse_y * 0.95, top=max_rmse_y * 1.15)
    ax1.legend(loc='lower right', fontsize=9)

    # Plot 2: Output SNR (changed from SNR Improvement)
    ax2 = axes[1]
    x_pos = 0
    max_snr_y = 0
    min_snr_y = float('inf')

    for config_idx, noise_config in enumerate(noise_configs):
        config_name = noise_config['name']
        config_df = summary_df[summary_df['noise_config'] == config_name]

        for model_idx, model_name in enumerate(sorted_models):
            model_df = config_df[config_df['model'] == model_name]

            if not model_df.empty:
                point_snr = model_df['point_output_snr'].values[0]
                lower_snr = model_df['lower_output_snr'].values[0]
                upper_snr = model_df['upper_output_snr'].values[0]

                # Compute error bars (asymmetric)
                lower_err = point_snr - lower_snr
                upper_err = upper_snr - point_snr

                color = color_map.get(model_name, '#cccccc')

                bar_pos = x_pos + model_idx * bar_width
                ax2.bar(bar_pos, point_snr, bar_width,
                       yerr=[[lower_err], [upper_err]], capsize=3,
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
                       label=model_name if config_idx == 0 else '')

                # Add text label on top of bar
                ax2.text(bar_pos, point_snr + upper_err, f'{point_snr:.2f}',
                        ha='center', va='bottom', fontsize=8, color='black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='none', alpha=0.7))

                # Track maximum and minimum y values
                max_snr_y = max(max_snr_y, point_snr + upper_err)
                min_snr_y = min(min_snr_y, point_snr - lower_err)

        x_pos += group_width + 0.3

    ax2.set_xlabel('Noise Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Output SNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_title('Output SNR Comparison (90% Bootstrap CI)', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(config_positions)
    ax2.set_xticklabels(config_labels, fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=min_snr_y * 0.95, top=max_snr_y * 1.15)
    ax2.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    # Save
    plot_path = os.path.join(exp_folder, 'results', 'qui_plot_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved qui_plot to: {plot_path}")

    # Also save summary stats
    summary_path = os.path.join(exp_folder, 'results', 'qui_plot_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved qui_plot summary to: {summary_path}")
