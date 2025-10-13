"""
RMSE Analysis: Visualize examples with high and low RMSE to understand variance.

This module helps analyze why RMSE has higher standard deviation compared to SNR
by visualizing examples at the extremes of the RMSE distribution.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import calculate_snr, calculate_rmse


def analyze_rmse_variance(noise_configs, exp_folder, config, clean_val, models):
    """Analyze RMSE variance by plotting high and low RMSE examples.

    For each model and noise config, shows:
    - 3 examples with highest RMSE (mean + stddev)
    - 3 examples with lowest RMSE (mean - stddev)

    Args:
        noise_configs: List of dicts with 'name' and 'path' keys
        exp_folder: Experiment folder path
        config: Main config dictionary
        clean_val: Clean validation signals
        models: List of model names to evaluate
    """
    import torch
    from .utils import get_model
    from ecg_noise_factory.noise import NoiseFactory

    print("\n" + "="*80)
    print("RMSE VARIANCE ANALYSIS")
    print("="*80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and
                         config['hardware']['use_cuda'] else 'cpu')

    # Load the qui_plot_summary for reference
    summary_path = os.path.join(exp_folder, 'results', 'qui_plot_summary.csv')
    if not os.path.exists(summary_path):
        print(f"Warning: qui_plot_summary.csv not found at {summary_path}")
        print("Please run qui_plot first to generate the summary.")
        return

    summary_df = pd.read_csv(summary_path)
    print(f"\nLoaded summary from: {summary_path}")
    print(f"Models: {summary_df['model'].unique()}")
    print(f"Noise configs: {summary_df['noise_config'].unique()}")

    # For each noise config and model, find high/low RMSE examples
    for noise_config in tqdm(noise_configs, desc="Processing noise configs"):
        config_name = noise_config['name']
        config_path = noise_config['path']

        print(f"\n--- Analyzing Noise Config: {config_name} ---")

        # Initialize NoiseFactory
        noise_data_path = os.path.join(os.path.dirname(__file__), '../../../ecg_noise/data')
        full_config_path = os.path.join(os.path.dirname(__file__), '..', config_path)

        noise_factory = NoiseFactory(
            data_path=noise_data_path,
            sampling_rate=config['sampling_frequency'],
            config_path=full_config_path,
            mode='eval'
        )

        # Generate noisy validation data
        noisy_val = noise_factory.add_noise(
            x=clean_val, batch_axis=0, channel_axis=2, length_axis=1
        )

        for model_name in tqdm(models, desc=f"  Analyzing models ({config_name})", leave=False):
            # Get model summary stats
            model_summary = summary_df[
                (summary_df['noise_config'] == config_name) &
                (summary_df['model'] == model_name)
            ]

            if model_summary.empty:
                tqdm.write(f"    Warning: No summary found for {model_name} in {config_name}")
                continue

            mean_rmse = model_summary['mean_rmse'].values[0]
            std_rmse = model_summary['std_rmse'].values[0]

            print(f"\n  Model: {model_name}")
            print(f"    Mean RMSE: {mean_rmse:.4f}, Std RMSE: {std_rmse:.4f}")

            # Load model and generate predictions
            model_folder = os.path.join(exp_folder, 'models', model_name)
            model_path = os.path.join(model_folder, 'best_model.pth')

            if not os.path.exists(model_path):
                tqdm.write(f"    Warning: Model not found at {model_path}")
                continue

            model_config = next((m for m in config['models'] if m['name'] == model_name), None)
            if not model_config:
                continue

            model_type = model_config['type']
            is_stage2 = model_type.lower() in ['stage2', 'drnet']

            # Handle Stage2 models
            stage1_predictions = None
            if is_stage2:
                stage1_model_name = model_config.get('stage1_model')
                if not stage1_model_name:
                    continue

                stage1_folder = os.path.join(exp_folder, 'models', stage1_model_name)
                stage1_path = os.path.join(stage1_folder, 'best_model.pth')

                if not os.path.exists(stage1_path):
                    continue

                stage1_config = next((m for m in config['models'] if m['name'] == stage1_model_name), None)
                if not stage1_config:
                    continue

                # Load and run Stage1 model
                stage1_model = get_model(stage1_config['type'], input_length=clean_val[0].shape[0], is_stage2=False)
                stage1_model.load_state_dict(torch.load(stage1_path, map_location=device))
                stage1_model = stage1_model.to(device)
                stage1_model.eval()

                stage1_predictions = []
                with torch.no_grad():
                    for i in range(len(clean_val)):
                        noisy_sample = noisy_val[i]
                        noisy_sample = torch.FloatTensor(noisy_sample).permute(1, 0).unsqueeze(0).unsqueeze(0)
                        noisy_sample = noisy_sample.to(device)

                        stage1_pred = stage1_model(noisy_sample)
                        stage1_pred = stage1_pred.squeeze().cpu().numpy()
                        stage1_predictions.append(stage1_pred)

                stage1_predictions = np.array(stage1_predictions)
                del stage1_model
                torch.cuda.empty_cache()

            # Load main model
            model = get_model(model_type, input_length=clean_val[0].shape[0], is_stage2=is_stage2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()

            # Generate predictions and calculate RMSE for each sample
            predictions = []
            rmse_values = []
            snr_values = []

            with torch.no_grad():
                for i in range(len(clean_val)):
                    noisy_sample = noisy_val[i]
                    noisy_sample = torch.FloatTensor(noisy_sample).permute(1, 0).unsqueeze(0).unsqueeze(0)

                    if is_stage2:
                        stage1_sample = stage1_predictions[i]
                        stage1_sample = torch.FloatTensor(stage1_sample).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                        model_input = torch.cat([noisy_sample, stage1_sample], dim=1)
                    else:
                        model_input = noisy_sample

                    model_input = model_input.to(device)
                    pred = model(model_input)
                    pred = pred.squeeze().cpu().numpy()
                    predictions.append(pred)

                    # Calculate metrics
                    clean = clean_val[i].squeeze()
                    denoised = pred
                    rmse = calculate_rmse(clean, denoised)
                    snr = calculate_snr(clean, denoised)

                    rmse_values.append(rmse)
                    snr_values.append(snr)

            predictions = np.array(predictions)
            rmse_values = np.array(rmse_values)
            snr_values = np.array(snr_values)

            # Clean up model
            del model
            torch.cuda.empty_cache()

            # Find high and low RMSE examples
            high_threshold = mean_rmse + std_rmse
            low_threshold = mean_rmse - std_rmse

            high_rmse_indices = np.where(rmse_values >= high_threshold)[0]
            low_rmse_indices = np.where(rmse_values <= low_threshold)[0]

            # Sort and take top 3
            if len(high_rmse_indices) > 0:
                high_rmse_indices = high_rmse_indices[np.argsort(rmse_values[high_rmse_indices])[-3:]]
            if len(low_rmse_indices) > 0:
                low_rmse_indices = low_rmse_indices[np.argsort(rmse_values[low_rmse_indices])[:3]]

            print(f"    High RMSE examples (>= {high_threshold:.4f}): {len(high_rmse_indices)}")
            print(f"    Low RMSE examples (<= {low_threshold:.4f}): {len(low_rmse_indices)}")

            if len(high_rmse_indices) == 0 and len(low_rmse_indices) == 0:
                print(f"    No extreme examples found, skipping plot")
                continue

            # Create plot
            n_high = min(3, len(high_rmse_indices))
            n_low = min(3, len(low_rmse_indices))
            n_total = n_high + n_low

            if n_total == 0:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'RMSE Analysis: {model_name} - {config_name.upper()} noise\n'
                        f'Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}',
                        fontsize=16, fontweight='bold')

            # Plot high RMSE examples (top row)
            for i in range(3):
                ax = axes[0, i]
                if i < n_high:
                    idx = high_rmse_indices[i]
                    clean = clean_val[idx].squeeze()
                    noisy = noisy_val[idx].squeeze()
                    denoised = predictions[idx]

                    rmse = rmse_values[idx]
                    snr = snr_values[idx]

                    # Plot signals
                    ax.plot(clean, 'g-', linewidth=1.5, label='Clean', alpha=0.8)
                    ax.plot(noisy, color='grey', linewidth=1, label='Noisy', alpha=0.5)
                    ax.plot(denoised, 'b-', linewidth=1.5, label='Denoised', alpha=0.9)

                    ax.set_title(f'HIGH RMSE Example {i+1}\n'
                                f'RMSE: {rmse:.4f} | SNR: {snr:.2f} dB',
                                fontsize=11, fontweight='bold', color='red')
                    ax.legend(loc='upper right', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    if i == 0:
                        ax.set_ylabel('Amplitude', fontsize=10)
                else:
                    ax.axis('off')

            # Plot low RMSE examples (bottom row)
            for i in range(3):
                ax = axes[1, i]
                if i < n_low:
                    idx = low_rmse_indices[i]
                    clean = clean_val[idx].squeeze()
                    noisy = noisy_val[idx].squeeze()
                    denoised = predictions[idx]

                    rmse = rmse_values[idx]
                    snr = snr_values[idx]

                    # Plot signals
                    ax.plot(clean, 'g-', linewidth=1.5, label='Clean', alpha=0.8)
                    ax.plot(noisy, color='grey', linewidth=1, label='Noisy', alpha=0.5)
                    ax.plot(denoised, 'b-', linewidth=1.5, label='Denoised', alpha=0.9)

                    ax.set_title(f'LOW RMSE Example {i+1}\n'
                                f'RMSE: {rmse:.4f} | SNR: {snr:.2f} dB',
                                fontsize=11, fontweight='bold', color='green')
                    ax.legend(loc='upper right', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel('Sample', fontsize=10)
                    if i == 0:
                        ax.set_ylabel('Amplitude', fontsize=10)
                else:
                    ax.axis('off')

            plt.tight_layout()

            # Save plot
            plot_filename = f'rmse_analysis_{model_name}_{config_name}.png'
            plot_path = os.path.join(exp_folder, 'results', plot_filename)
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()

            print(f"    ✓ Saved analysis to: {plot_filename}")

    print("\n" + "="*80)
    print("✓ RMSE variance analysis complete!")
    print("="*80)
