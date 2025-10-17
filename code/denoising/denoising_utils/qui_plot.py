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

from .utils import get_model
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


def qui_plot(noise_configs, exp_folder, config, clean_test, models):
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
    """
    # Color mapping for models (Stage1 = light, Stage2 = dark)
    color_map = {
        'fcn': '#aec7e8',         # Light blue (Stage1)
        'drnet_fcn': '#1f77b4',   # Dark blue (Stage2)
        'unet': '#ff9896',        # Light red (Stage1)
        'drnet_unet': '#d62728',  # Dark red (Stage2)
        'imunet': '#98df8a',      # Light green (Stage1)
        'drnet_imunet': '#2ca02c' # Dark green (Stage2)
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

                # Generate Stage1 predictions
                stage1_predictions = []
                with torch.no_grad():
                    for i in tqdm(range(len(clean_test)), desc=f"    Stage1 predictions", position=2, leave=False):
                        noisy_sample = noisy_test[i]
                        noisy_sample = torch.FloatTensor(noisy_sample).permute(1, 0).unsqueeze(0).unsqueeze(0)
                        noisy_sample = noisy_sample.to(device)

                        stage1_pred = stage1_model(noisy_sample)
                        stage1_pred = stage1_pred.squeeze().cpu().numpy()
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
                    pred = model(model_input)
                    pred = pred.squeeze().cpu().numpy()  # Remove batch and channel dims
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

    # Compute summary statistics (mean and std for confidence intervals)
    summary_stats = []
    for noise_config in noise_configs:
        config_name = noise_config['name']
        config_df = results_df[results_df['noise_config'] == config_name]

        for model_name in models:
            model_df = config_df[config_df['model'] == model_name]

            if not model_df.empty:
                summary_stats.append({
                    'noise_config': config_name,
                    'model': model_name,
                    'mean_rmse': model_df['rmse_denoised'].mean(),
                    'std_rmse': model_df['rmse_denoised'].std(),
                    'mean_output_snr': model_df['output_snr_db'].mean(),
                    'std_output_snr': model_df['output_snr_db'].std()
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

    for config_idx, noise_config in enumerate(noise_configs):
        config_name = noise_config['name']
        config_df = summary_df[summary_df['noise_config'] == config_name]

        for model_idx, model_name in enumerate(sorted_models):
            model_df = config_df[config_df['model'] == model_name]

            if not model_df.empty:
                mean_rmse = model_df['mean_rmse'].values[0]
                std_rmse = model_df['std_rmse'].values[0]
                color = color_map.get(model_name, '#cccccc')

                bar_pos = x_pos + model_idx * bar_width
                ax1.bar(bar_pos, mean_rmse, bar_width,
                       yerr=std_rmse, capsize=3,
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
                       label=model_name if config_idx == 0 else '')

        group_center = x_pos + (n_models - 1) * bar_width / 2
        config_positions.append(group_center)
        config_labels.append(config_name.upper())
        x_pos += group_width + 0.3

    ax1.set_xlabel('Noise Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean RMSE (Denoised)', fontsize=12, fontweight='bold')
    ax1.set_title('RMSE Comparison', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(config_positions)
    ax1.set_xticklabels(config_labels, fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='upper right', fontsize=9)

    # Plot 2: Output SNR (changed from SNR Improvement)
    ax2 = axes[1]
    x_pos = 0

    for config_idx, noise_config in enumerate(noise_configs):
        config_name = noise_config['name']
        config_df = summary_df[summary_df['noise_config'] == config_name]

        for model_idx, model_name in enumerate(sorted_models):
            model_df = config_df[config_df['model'] == model_name]

            if not model_df.empty:
                mean_snr = model_df['mean_output_snr'].values[0]
                std_snr = model_df['std_output_snr'].values[0]
                color = color_map.get(model_name, '#cccccc')

                bar_pos = x_pos + model_idx * bar_width
                ax2.bar(bar_pos, mean_snr, bar_width,
                       yerr=std_snr, capsize=3,
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
                       label=model_name if config_idx == 0 else '')

        x_pos += group_width + 0.3

    ax2.set_xlabel('Noise Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Output SNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_title('Output SNR Comparison', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(config_positions)
    ax2.set_xticklabels(config_labels, fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper right', fontsize=9)

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
