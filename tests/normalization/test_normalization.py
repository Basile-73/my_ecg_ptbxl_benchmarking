#!/usr/bin/env python3
"""
General-purpose normalization testing script for numpy arrays.

This script checks normalization of .npy files according to a configuration file
and generates reports and visualizations.
"""

import os
import sys
import yaml
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse

from stats_utils import (
    analyze_normalization,
    format_stats_report,
    get_extreme_indices
)
from plot_utils import (
    plot_distribution_histograms,
    plot_extreme_examples,
    plot_extreme_examples_multi_metric,
    create_summary_plot
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate that the configuration has all required fields."""
    required_fields = ['test_name', 'files', 'normalization_type']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    # Validate normalization type
    valid_types = ['standard', 'minmax', 'robust']
    if config['normalization_type'] not in valid_types:
        raise ValueError(f"Invalid normalization type. Must be one of: {valid_types}")

    # Validate files list
    if not isinstance(config['files'], list) or len(config['files']) == 0:
        raise ValueError("'files' must be a non-empty list")


def check_dimensions(data: np.ndarray, expected_n_dims: Optional[int],
                     file_path: str) -> bool:
    """
    Check if data has the expected number of dimensions.

    Args:
        data: Loaded numpy array
        expected_n_dims: Expected number of dimensions (None to skip check)
        file_path: Path to the file (for error messages)

    Returns:
        True if dimensions match or no check required, False otherwise
    """
    if expected_n_dims is None:
        return True

    if len(data.shape) != expected_n_dims:
        print(f"ERROR: Dimension mismatch in {file_path}")
        print(f"  Expected: {expected_n_dims} dimensions")
        print(f"  Got: {len(data.shape)} dimensions")
        return False

    return True
def process_file(file_path: str, config: Dict[str, Any], output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Process a single .npy file and generate analysis.

    Args:
        file_path: Path to the .npy file
        config: Configuration dictionary
        output_dir: Output directory for this file's results

    Returns:
        Statistics dictionary or None if processing failed
    """
    print(f"\nProcessing: {file_path}")

    # Load data
    try:
        data = np.load(file_path)
        print(f"  Loaded array with shape: {data.shape}")
    except Exception as e:
        print(f"  ERROR: Failed to load file: {e}")
        return None

    # Check dimensions
    expected_n_dims = config.get('expected_n_dimensions', None)
    if not check_dimensions(data, expected_n_dims, file_path):
        return None

    # Get normalization parameters
    norm_type = config['normalization_type']
    axis = config.get('normalization_axis', None)
    n_examples = config.get('n_extreme_examples', 5)

    # Analyze normalization
    print(f"  Analyzing {norm_type} normalization along axis {axis}...")
    stats = analyze_normalization(data, norm_type, axis)

    # Generate report
    report = format_stats_report(stats, norm_type, axis)

    # Save report to file
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    report_path = os.path.join(output_dir, f"{file_name}_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Normalization Analysis Report\n")
        f.write(f"File: {file_path}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n{report}\n")
    print(f"  Saved report to {report_path}")

    # Plot distributions
    dist_plot_path = os.path.join(output_dir, f"{file_name}_distributions.png")
    plot_distribution_histograms(stats, norm_type, dist_plot_path, axis)

    # Plot extreme examples for all relevant metrics
    plot_extreme_examples_multi_metric(data, stats, norm_type, n_examples,
                                       output_dir, file_name, axis)

    return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Test normalization of numpy arrays stored in .npy files'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='test_config.yaml',
        help='Path to configuration file (default: test_config.yaml)'
    )
    args = parser.parse_args()

    # Load and validate configuration
    print("=" * 80)
    print("Normalization Testing Script")
    print("=" * 80)

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)

    print(f"\nLoading configuration from: {config_path}")
    config = load_config(config_path)
    validate_config(config)

    # Create output directory
    test_name = config['test_name']
    base_output_dir = os.path.join(os.path.dirname(__file__), 'output', test_name)
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Output directory: {base_output_dir}")

    # Process each file
    all_stats = []
    processed_files = []

    for file_path in config['files']:
        if not os.path.exists(file_path):
            print(f"\nWARNING: File not found, skipping: {file_path}")
            continue

        # Create subdirectory for this file
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_output_dir = os.path.join(base_output_dir, file_name)
        os.makedirs(file_output_dir, exist_ok=True)

        # Process file
        stats = process_file(file_path, config, file_output_dir)
        if stats is not None:
            all_stats.append(stats)
            processed_files.append(file_path)

    # Generate summary if multiple files were processed
    if len(all_stats) > 1:
        print("\nGenerating summary comparison plot...")
        summary_path = os.path.join(base_output_dir, "summary_comparison.png")
        create_summary_plot(all_stats, processed_files,
                          config['normalization_type'], summary_path)

    # Generate overall summary report
    summary_report_path = os.path.join(base_output_dir, "summary_report.txt")
    with open(summary_report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NORMALIZATION TESTING SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Name: {test_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Normalization Type: {config['normalization_type']}\n")
        f.write(f"Axis: {config.get('normalization_axis', 'Global')}\n")
        f.write(f"\nFiles Processed: {len(processed_files)} / {len(config['files'])}\n")
        f.write("\n" + "=" * 80 + "\n")
        for file_path in processed_files:
            f.write(f"  - {file_path}\n")
        f.write("=" * 80 + "\n")

    print("\n" + "=" * 80)
    print(f"Testing complete! Results saved to: {base_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
