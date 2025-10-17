# Normalization Testing Tool

A general-purpose Python toolkit for testing and validating normalization of numpy arrays stored as `.npy` files.

## Overview

This tool helps verify that your data has been properly normalized according to different normalization schemes (standard, min-max, or robust). It provides:

1. **Statistical Analysis**: Computes relevant statistics and checks deviations from expected values
2. **Visualization**: Generates distribution plots and highlights extreme cases
3. **Reporting**: Creates comprehensive text reports with all findings

## Files

- `test_normalization.py`: Main script that orchestrates the testing process
- `stats_utils.py`: Statistical computation functions for different normalization types
- `plot_utils.py`: Visualization functions for creating plots and charts
- `test_config.yaml`: Configuration file template (customize for your tests)
- `README_test.md`: This documentation file

## Installation

No special installation required beyond standard scientific Python packages:

```bash
pip install numpy matplotlib pyyaml
```

## Configuration

Create or modify `test_config.yaml` to specify your test parameters:

```yaml
# Name of the test (used for output directory)
test_name: "my_normalization_test"

# List of .npy files to analyze
files:
  - "path/to/file1.npy"
  - "path/to/file2.npy"

# Expected number of dimensions (e.g., 3 for 3D arrays)
# Set to null to skip dimension checking
expected_n_dimensions: 3

# Axis along which to check normalization
# Can be a single integer, null for global normalization
normalization_axis: 0

# Type of normalization to check
# Options: "standard", "minmax", "robust"
normalization_type: "standard"

# Number of extreme examples to plot
n_extreme_examples: 5
```

### Configuration Parameters

- **test_name**: Name for this test run (creates a subdirectory in `outputs/`)
- **files**: List of paths to `.npy` files to analyze
- **expected_n_dimensions**: Expected number of dimensions (validation step)
  - Set to an integer like `3` for 3D arrays
  - Use `null` to skip dimension checking
- **normalization_axis**: Axis along which normalization was applied
  - `0`: Along first dimension (e.g., across samples)
  - `1`: Along second dimension
  - `null`: Global normalization (entire array)
- **normalization_type**: Type of normalization to verify
  - `"standard"`: Zero mean, unit variance (z-score normalization)
  - `"minmax"`: Scaled to [0, 1] range
  - `"robust"`: Median centering with IQR scaling
- **n_extreme_examples**: Number of most extreme examples to plot

## Usage

### Basic Usage

Run with the default configuration file (`test_config.yaml`):

```bash
python test_normalization.py
```

### Custom Configuration

Specify a different configuration file:

```bash
python test_normalization.py --config my_custom_config.yaml
```

### Example Workflow

1. Create a configuration file for your test:
```yaml
test_name: "ecg_data_check"
files:
  - "data/train_normalized.npy"
  - "data/test_normalized.npy"
expected_n_dimensions: 3
normalization_axis: 0
normalization_type: "standard"
n_extreme_examples: 10
```

2. Run the test:
```bash
python test_normalization.py --config ecg_config.yaml
```

3. Check results in `outputs/ecg_data_check/`

## Output Structure

Results are saved in `outputs/<test_name>/` with the following structure:

```
outputs/
└── test_name/
    ├── summary_report.txt              # Overall summary
    ├── summary_comparison.png          # Comparison across files (if multiple)
    ├── file1/
    │   ├── file1_report.txt           # Detailed statistics
    │   ├── file1_distributions.png     # Distribution histograms
    │   └── file1_extreme_examples.png  # Most extreme cases
    └── file2/
        ├── file2_report.txt
        ├── file2_distributions.png
        └── file2_extreme_examples.png
```

## Understanding Results

### Standard Normalization (Z-score)

**Expected**: Mean ≈ 0, Standard Deviation ≈ 1

The tool checks:
- Distribution of means across the specified axis
- Distribution of standard deviations
- Extreme examples with highest/lowest standard deviations

**Good normalization**: Means centered around 0, std dev centered around 1

### Min-Max Normalization

**Expected**: Min ≈ 0, Max ≈ 1

The tool checks:
- Distribution of minimum values
- Distribution of maximum values
- Extreme examples with highest/lowest maximum values

**Good normalization**: Min values near 0, max values near 1

### Robust Normalization

**Expected**: Median ≈ 0, IQR ≈ 1

The tool checks:
- Distribution of medians
- Distribution of interquartile ranges (IQR)
- Q25 and Q75 distributions
- Extreme examples with highest/lowest IQR

**Good normalization**: Medians centered around 0, IQR centered around 1

## Interpretation Tips

1. **Check the distribution plots**: Tight distributions around expected values indicate good normalization
2. **Review extreme examples**: These highlight potential issues or outliers in your data
3. **Compare across files**: If normalizing train/test sets, ensure consistency
4. **Look for systematic deviations**: Consistent bias might indicate preprocessing errors

## Examples

### Example 1: Check ECG signal normalization
```yaml
test_name: "ecg_standard_norm"
files:
  - "data/ptbxl/normalized_train.npy"
  - "data/ptbxl/normalized_test.npy"
expected_n_dimensions: 3
normalization_axis: 0
normalization_type: "standard"
n_extreme_examples: 5
```

### Example 2: Verify channel-wise normalization
```yaml
test_name: "channel_wise_norm"
files:
  - "processed_signals.npy"
expected_n_dimensions: null
normalization_axis: 2
normalization_type: "minmax"
n_extreme_examples: 10
```

### Example 3: Global robust scaling
```yaml
test_name: "global_robust"
files:
  - "features.npy"
expected_n_dimensions: 2
normalization_axis: null
normalization_type: "robust"
n_extreme_examples: 3
```

## Troubleshooting

**Problem**: "Dimension mismatch" error
- **Solution**: Check that `expected_n_dimensions` matches the number of dimensions in your data, or set to `null`

**Problem**: All statistics show large deviations
- **Solution**: Verify that normalization was actually applied to the data

**Problem**: No plots generated
- **Solution**: Ensure matplotlib is installed and you have write permissions to the output directory

**Problem**: Memory error with large files
- **Solution**: Process files individually or reduce `n_extreme_examples`

## Extension Ideas

The modular design makes it easy to extend:

- Add new normalization types in `stats_utils.py`
- Create custom visualizations in `plot_utils.py`
- Implement additional validation checks
- Add support for specific data formats

## License

This tool is part of the ECG PTB-XL benchmarking project.
