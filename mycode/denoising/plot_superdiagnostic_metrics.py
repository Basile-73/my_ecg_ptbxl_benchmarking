#!/usr/bin/env python3
"""
Script to plot SNR and RMSE metrics by superdiagnostic class from CSV file.

This script generates grouped bar plots showing denoising model performance
across different superdiagnostic classes with bootstrap confidence intervals.

Usage:
    python plot_superdiagnostic_metrics.py <path_to_metrics_by_superdiagnostic.csv>

Example:
    python plot_superdiagnostic_metrics.py output/test_mamba_models_experiment/results/metrics_by_superdiagnostic.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


# Color map for models (12 models as specified)
color_map = {
    'fcn': '#6baed6',
    'unet': '#fc8d62',
    'imunet': '#66c2a5',
    'drnet_fcn': '#8c564b',
    'drnet_unet': '#e377c2',
    'drnet_imunet': '#7f7f7f',
    'imunet_origin': '#bcbd22',
    'imunet_mamba_bn': '#ff7f0e',
    'imunet_mamba_up': '#17becf',
    'imunet_mamba_early': '#e377c2',
    'imunet_mamba_late': '#bcbd22',
    'mecge_phase': '#c5b0d5',
    # Add alias keys for model names with source tags
    'fcn (Qui)': '#6baed6',
    'unet (Qui)': '#fc8d62',
    'imunet (Qui)': '#66c2a5',
    'mecge_phase (Hung)': '#c5b0d5'
}


def create_metric_plot(df, metric_name, output_dir, color_map):
    """
    Create a grouped bar plot for a specific metric.

    Args:
        df (pd.DataFrame): DataFrame with metrics data
        metric_name (str): 'snr' or 'rmse'
        output_dir (str): Directory to save output files
        color_map (dict): Dictionary mapping model names to colors
    """
    # Extract unique classes and models, then sort
    classes = sorted(df['class'].unique())
    models = sorted(df['model'].unique())

    # Create figure with dynamic sizing (Comment 4: account for both classes and models)
    width_factor = max(14, len(classes) * (0.8 + 0.15 * max(0, len(models) - 4)))
    fig, ax = plt.subplots(figsize=(width_factor, 6))

    # Calculate bar positioning
    x = np.arange(len(classes))
    width = 0.8 / len(models)

    # Adjust font size based on number of models
    fontsize_annotation = 7 if len(models) > 8 else 8
    if len(models) > 10:
        fontsize_annotation = 6

    # For each model, plot bars
    for i, model in enumerate(models):
        # Filter DataFrame for current model
        model_data = df[df['model'] == model]

        # Extract mean values for all classes
        mean_values = []
        lower_ci_values = []
        upper_ci_values = []
        sample_counts = []

        for class_name in classes:
            class_data = model_data[model_data['class'] == class_name]
            if len(class_data) > 0:
                mean_values.append(class_data.iloc[0][f'{metric_name}_mean'])
                lower_ci_values.append(class_data.iloc[0][f'{metric_name}_lower_ci'])
                upper_ci_values.append(class_data.iloc[0][f'{metric_name}_upper_ci'])
                sample_counts.append(class_data.iloc[0]['n_samples'])
            else:
                # Handle missing data
                mean_values.append(np.nan)
                lower_ci_values.append(np.nan)
                upper_ci_values.append(np.nan)
                sample_counts.append(0)

        mean_values = np.array(mean_values)
        lower_ci_values = np.array(lower_ci_values)
        upper_ci_values = np.array(upper_ci_values)

        # Calculate asymmetric error bars (Comment 1: prevent negative error extents)
        valid = (~np.isnan(mean_values)) & (~np.isnan(lower_ci_values)) & (~np.isnan(upper_ci_values))
        yerr_lower = np.zeros_like(mean_values)
        yerr_upper = np.zeros_like(mean_values)
        yerr_lower[valid] = np.maximum(mean_values[valid] - lower_ci_values[valid], 0)
        yerr_upper[valid] = np.maximum(upper_ci_values[valid] - mean_values[valid], 0)

        # Plot bars with error bars
        bars = ax.bar(
            x + i * width,
            mean_values,
            width,
            yerr=[yerr_lower, yerr_upper],
            capsize=3,
            color=color_map.get(model, '#808080'),
            label=model,
            error_kw={'elinewidth': 1, 'capthick': 1}
        )

        # Annotate mean values on top of each bar
        for j, (bar, value, err_upper) in enumerate(zip(bars, mean_values, yerr_upper)):
            if not np.isnan(value):
                # Position text slightly above the error bar (Comment 3: use fixed offset)
                offset = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                y_pos = bar.get_height() + err_upper + offset

                # Format based on metric type
                if metric_name == 'snr':
                    text = f'{value:.2f}'
                else:  # rmse
                    text = f'{value:.4f}'

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    text,
                    ha='center',
                    va='bottom',
                    fontsize=fontsize_annotation
                )

    # Comment 3: Add vertical headroom to prevent clipping of annotations
    ax.margins(y=0.1)

    # Create x-axis labels with sample counts
    # Use first occurrence of n_samples for each class (should be consistent across models)
    labels = []
    for class_name in classes:
        class_data = df[df['class'] == class_name]
        if len(class_data) > 0:
            n_samples = int(class_data.iloc[0]['n_samples'])
            labels.append(f'{class_name}\n(n={n_samples})')
        else:
            labels.append(f'{class_name}\n(n=0)')

    # Format plot
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Super Diagnostic Class', fontweight='bold', fontsize=12)

    if metric_name == 'snr':
        ax.set_ylabel('SNR (dB)', fontweight='bold', fontsize=12)
        ax.set_title('SNR by Super Diagnostic Class', fontweight='bold', fontsize=14)
    else:  # rmse
        ax.set_ylabel('RMSE', fontweight='bold', fontsize=12)
        ax.set_title('RMSE by Super Diagnostic Class', fontweight='bold', fontsize=14)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Save figures
    plt.tight_layout()

    # Save PNG
    png_path = os.path.join(output_dir, f'superdiagnostic_{metric_name}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {png_path}")

    # Save PDF
    pdf_path = os.path.join(output_dir, f'superdiagnostic_{metric_name}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Plot saved to: {pdf_path}")

    plt.close()


def plot_superdiagnostic_metrics(csv_path):
    """
    Load CSV file and generate plots for SNR and RMSE metrics.

    Args:
        csv_path (str): Path to the metrics_by_superdiagnostic.csv file
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Validate required columns exist
    required_columns = [
        'model', 'class', 'n_samples',
        'snr_mean', 'snr_lower_ci', 'snr_upper_ci',
        'rmse_mean', 'rmse_lower_ci', 'rmse_upper_ci'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Print summary statistics
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(f"CSV file: {csv_path}")
    print(f"Number of models: {df['model'].nunique()}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Number of classes: {df['class'].nunique()}")
    print(f"Classes: {sorted(df['class'].unique())}")
    print(f"DataFrame shape: {df.shape}")
    print("="*80 + "\n")

    # Determine output directory
    output_dir = os.path.dirname(csv_path)
    if not output_dir:
        output_dir = '.'

    # Create SNR plot
    print("\nCreating SNR plot...")
    create_metric_plot(df, 'snr', output_dir, color_map)

    # Create RMSE plot
    print("\nCreating RMSE plot...")
    create_metric_plot(df, 'rmse', output_dir, color_map)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Plot metrics by superdiagnostic class from CSV file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python plot_superdiagnostic_metrics.py output/test_mamba_models_experiment/results/metrics_by_superdiagnostic.csv
        """
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to metrics_by_superdiagnostic.csv file'
    )

    args = parser.parse_args()

    # Validate file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found: {args.csv_path}")
        return 1

    # Validate file extension
    if not args.csv_path.endswith('.csv'):
        print(f"Warning: The file does not have a .csv extension: {args.csv_path}")

    # Generate plots
    try:
        plot_superdiagnostic_metrics(args.csv_path)
        print("\nDone!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
