"""
Evaluate denoising experiment results and generate report.
"""
import os
import sys

# Add paths FIRST before any local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '../../ecg_noise/source'))

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from denoising_utils.utils import calculate_snr, calculate_rmse
from denoising_utils.qui_plot import qui_plot
from ecg_noise_factory.noise import NoiseFactory


def load_config(config_path='config.yaml'):
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_model(model_name, exp_folder, clean_test, noisy_test):
    """Evaluate a single model."""
    model_folder = os.path.join(exp_folder, 'models', model_name)
    pred_path = os.path.join(model_folder, 'predictions.npy')

    if not os.path.exists(pred_path):
        print(f"Warning: Predictions not found for {model_name}")
        return None

    predictions = np.load(pred_path)

    # Calculate metrics for each sample
    results = []
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
        rmse_improvement = rmse_noisy - rmse_denoised

        results.append({
            'model': model_name,
            'sample_idx': i,
            'input_snr_db': input_snr,
            'output_snr_db': output_snr,
            'snr_improvement_db': snr_improvement,
            'rmse_noisy': rmse_noisy,
            'rmse_denoised': rmse_denoised,
            'rmse_improvement': rmse_improvement,
            'rmse_improvement_pct': (rmse_improvement / rmse_noisy) * 100 if rmse_noisy > 0 else 0
        })

    return pd.DataFrame(results)


def compute_summary(results_df):
    """Compute summary statistics."""
    summary = []

    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]

        summary.append({
            'model': model,
            'mean_snr_improvement_db': model_df['snr_improvement_db'].mean(),
            'std_snr_improvement_db': model_df['snr_improvement_db'].std(),
            'median_snr_improvement_db': model_df['snr_improvement_db'].median(),
            'mean_output_snr_db': model_df['output_snr_db'].mean(),
            'mean_rmse_noisy': model_df['rmse_noisy'].mean(),
            'mean_rmse_denoised': model_df['rmse_denoised'].mean(),
            'mean_rmse_improvement': model_df['rmse_improvement'].mean(),
            'rmse_improvement_pct': model_df['rmse_improvement_pct'].mean(),
            'n_samples': len(model_df)
        })

    return pd.DataFrame(summary)


def plot_comparison(summary_df, results_df, exp_folder):
    """Create comparison plots with Stage1/Stage2 grouping."""
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle('ECG Denoising Results', fontsize=16, fontweight='bold', y=0.98)

    models = summary_df['model'].unique()

    # Determine Stage1 vs Stage2 and assign colors
    colors = []
    stage_types = []
    for model in models:
        if 'drnet' in model.lower() or 'stage2' in model.lower():
            stage_types.append('stage2')
            # Extract base model name (e.g., 'fcn' from 'drnet_fcn')
            if '_' in model:
                base = model.split('_')[-1]
            else:
                base = model
            # Assign darker shade for stage2
            if 'fcn' in base.lower():
                colors.append('#1f77b4')  # Darker blue for fcn stage2
            elif 'unet' in base.lower():
                colors.append('#d62728')  # Darker red for unet stage2
            elif 'imunet' in base.lower():
                colors.append('#2ca02c')  # Darker green for imunet stage2
            else:
                colors.append('#9467bd')  # Darker purple for other stage2
        else:
            stage_types.append('stage1')
            # Assign lighter shade for stage1
            if 'fcn' in model.lower():
                colors.append('#aec7e8')  # Light blue for fcn stage1
            elif 'unet' in model.lower():
                colors.append('#ff9896')  # Light red for unet stage1
            elif 'imunet' in model.lower():
                colors.append('#98df8a')  # Light green for imunet stage1
            else:
                colors.append('#c5b0d5')  # Light purple for other stage1

    x = np.arange(len(models))

    # Plot 1: SNR Improvement
    ax1 = fig.add_subplot(gs[0, 0])
    means = summary_df['mean_snr_improvement_db'].values
    stds = summary_df['std_snr_improvement_db'].values
    bars = ax1.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('SNR Improvement (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('SNR Improvement by Model', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m + s + 0.5, f'{m:.2f}±{s:.2f}', ha='center', fontsize=8)

    # Plot 2: RMSE Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.35
    rmse_noisy = summary_df['mean_rmse_noisy'].values
    rmse_denoised = summary_df['mean_rmse_denoised'].values
    ax2.bar(x - width/2, rmse_noisy, width, label='Noisy', color='#e74c3c',
           alpha=0.8, edgecolor='black')
    ax2.bar(x + width/2, rmse_denoised, width, label='Denoised', color=colors,
           alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('RMSE: Noisy vs Denoised', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: RMSE Improvement Percentage
    ax3 = fig.add_subplot(gs[1, 0])
    improvement_pct = summary_df['rmse_improvement_pct'].values
    ax3.bar(x, improvement_pct, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax3.set_ylabel('RMSE Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Relative RMSE Improvement', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Good (>50%)')
    ax3.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Average (>30%)')
    ax3.legend(fontsize=9)

    # Plot 4: Output SNR Distribution (boxplot)
    ax4 = fig.add_subplot(gs[1, 1])
    data_to_plot = [results_df[results_df['model'] == m]['output_snr_db'].values
                   for m in models]
    bp = ax4.boxplot(data_to_plot, labels=models, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Output SNR (dB)', fontsize=12, fontweight='bold')
    ax4.set_title('Output SNR Distribution', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    plot_path = os.path.join(exp_folder, 'results', 'comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plots to: {plot_path}")


def plot_examples(exp_folder, clean_test, noisy_test, models, n_examples=3):
    """Plot example denoising results.

    Layout:
    - Column 1: Ground truth (green) + Noisy (red with opacity) overlaid
    - Columns 2+: Each model's prediction with ground truth (green background) and noisy (grey 50% opacity)
    """
    fig, axes = plt.subplots(n_examples, len(models) + 1,
                            figsize=(4*(len(models)+1), 3*n_examples))

    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_examples):
        # Column 1: Ground truth + Noisy overlay
        axes[i, 0].plot(clean_test[i].squeeze(), 'g-', linewidth=1.5, label='Ground Truth', alpha=0.8)
        axes[i, 0].plot(noisy_test[i].squeeze(), 'r-', linewidth=1, label='Noisy', alpha=0.6)
        axes[i, 0].set_title('Ground Truth + Noisy' if i == 0 else '')
        axes[i, 0].set_ylabel(f'Sample {i+1}', fontsize=10, fontweight='bold')
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].legend(loc='upper right', fontsize=8)

        # Columns 2+: Denoised signals with ground truth and noisy in background
        for j, model in enumerate(models):
            pred_path = os.path.join(exp_folder, 'models', model, 'predictions.npy')
            if os.path.exists(pred_path):
                predictions = np.load(pred_path)

                # Background: Ground truth (green) and noisy (grey 50% opacity)
                axes[i, j+1].plot(clean_test[i].squeeze(), 'g-', linewidth=1, alpha=0.5, label='Ground Truth')
                axes[i, j+1].plot(noisy_test[i].squeeze(), color='grey', linewidth=0.8, alpha=0.5, label='Noisy')

                # Foreground: Prediction (blue, prominent)
                axes[i, j+1].plot(predictions[i], 'b-', linewidth=1.5, label='Prediction', alpha=0.9)

                axes[i, j+1].set_title(model if i == 0 else '')
                axes[i, j+1].grid(True, alpha=0.3)

                if i == 0:
                    axes[i, j+1].legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    plot_path = os.path.join(exp_folder, 'results', 'example_denoising.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved examples to: {plot_path}")


def generate_pdf_report(summary_df, results_df, exp_folder, config):
    """Generate comprehensive PDF report."""
    pdf_path = os.path.join(exp_folder, 'results', 'denoising_report.pdf')

    with PdfPages(pdf_path) as pdf:
        # Page 1: Summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('ECG Denoising Report', fontsize=20, fontweight='bold')

        summary_text = f"""
EXPERIMENT: {config['experiment_name']}
Sampling Frequency: {config['sampling_frequency']} Hz
Models Evaluated: {len(summary_df)}

MODEL PERFORMANCE SUMMARY:
"""
        for _, row in summary_df.iterrows():
            summary_text += f"\n{row['model'].upper()}"
            summary_text += f"\n  SNR Improvement: {row['mean_snr_improvement_db']:.2f} ± {row['std_snr_improvement_db']:.2f} dB"
            summary_text += f"\n  RMSE Reduction: {row['rmse_improvement_pct']:.1f}%"
            summary_text += f"\n  Output SNR: {row['mean_output_snr_db']:.2f} dB"
            summary_text += "\n"

        plt.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=fig.transFigure)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Comparison plots
        plot_comparison(summary_df, results_df, exp_folder)
        img = plt.imread(os.path.join(exp_folder, 'results', 'comparison_plots.png'))
        fig = plt.figure(figsize=(11, 8.5))
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"✓ Generated PDF report: {pdf_path}")


def main():
    """Main evaluation function."""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate denoising results')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--report', action='store_true', help='Generate PDF report')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    exp_folder = os.path.join(config['outputfolder'], config['experiment_name'])

    print("\n" + "="*80)
    print("EVALUATING DENOISING RESULTS")
    print("="*80)

    # Load clean test data
    clean_test = np.load(os.path.join(exp_folder, 'data', 'clean_test.npy'))

    # Initialize NoiseFactory with 'eval' mode to avoid data leakage
    print("\nInitializing NoiseFactory with 'eval' mode (no data leakage)...")
    noise_data_path = os.path.join(os.path.dirname(__file__), '../../ecg_noise/data')
    noise_config_path = os.path.join(os.path.dirname(__file__), config['noise_config_path'])

    noise_factory = NoiseFactory(
        data_path=noise_data_path,
        sampling_rate=config['sampling_frequency'],
        config_path=noise_config_path,
        mode='eval'  # Use evaluation noise samples (no leakage from training)
    )

    # Generate noisy test data with eval noise
    print("Generating noisy test data with 'eval' mode noise samples...")
    noisy_test = noise_factory.add_noise(
        x=clean_test, batch_axis=0, channel_axis=2, length_axis=1
    )

    # Save eval noisy test data
    np.save(os.path.join(exp_folder, 'data', 'noisy_test_eval.npy'), noisy_test)

    print(f"\nTest samples: {len(clean_test)}")    # Evaluate each model
    all_results = []
    model_names = [m['name'] for m in config['models']]

    for model_name in model_names:
        print(f"\nEvaluating {model_name}...")
        results = evaluate_model(model_name, exp_folder, clean_test, noisy_test)
        if results is not None:
            all_results.append(results)

    if not all_results:
        print("No results found!")
        return

    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)

    # Compute summary
    summary_df = compute_summary(results_df)

    # Save results
    results_path = os.path.join(exp_folder, 'results', 'detailed_results.csv')
    results_df.to_csv(results_path, index=False)

    summary_path = os.path.join(exp_folder, 'results', 'summary.csv')
    summary_df.to_csv(summary_path, index=False)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    # Generate plots
    if config['output']['generate_plots']:
        print("\nGenerating plots...")
        plot_comparison(summary_df, results_df, exp_folder)

        if config['output']['n_example_plots'] > 0:
            plot_examples(exp_folder, clean_test, noisy_test, model_names,
                         n_examples=config['output']['n_example_plots'])

    # Generate qui_plot if enabled (multi-noise config comparison)
    if config.get('evaluation', {}).get('qui_plot', {}).get('enabled', False):
        print("\nGenerating qui_plot (multi-noise comparison)...")
        noise_configs = config['evaluation']['qui_plot']['noise_configs']
        qui_plot(noise_configs, exp_folder, config, clean_test, model_names)

    # Generate PDF report
    if args.report:
        print("\nGenerating PDF report...")
        generate_pdf_report(summary_df, results_df, exp_folder, config)

    print("\n✓ Evaluation complete!")
    print(f"\nResults saved to: {exp_folder}/results/")


if __name__ == '__main__':
    main()
