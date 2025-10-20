# PTB-XL Exploratory Data Analysis (EDA)

This directory contains scripts and outputs for comprehensive exploratory data analysis of the PTB-XL ECG dataset.

## Overview

The EDA pipeline analyzes the PTB-XL dataset to provide insights into:
- Signal statistics (mean, standard deviation, peaks)
- Lead inversion detection using robust QRS polarity analysis
- Label distributions across diagnostic hierarchies
- Superdiagnostic class-specific statistics

## Files

### Scripts
- **`ptbxl_eda.py`**: Main EDA script that performs all analyses and generates visualizations
- **`test_flipped_detection.py`**: Testing script for the flipped lead detection algorithm

### Output Structure

```
output/
├── data/                          # CSV files with statistics
│   ├── signal_statistics.csv     # Per-lead signal statistics
│   ├── distribution_diagnostic.csv
│   ├── distribution_subdiagnostic.csv
│   ├── distribution_superdiagnostic.csv
│   ├── stats_*.csv               # Statistics per superdiagnostic class
│   └── eda_summary_report.txt    # Comprehensive text report
│
└── plots/                         # Visualizations
    ├── signal_statistics.png      # Boxplots of signal means and stds
    ├── sample_ecgs/               # Sample ECG visualizations
    ├── flipped_examples/          # Examples of inverted leads with R-peaks
    └── by_superclass/             # Class-specific statistics
        ├── peak_statistics_*.png
        └── flipped_records_*.png
```

## Key Features

### 1. Robust Lead Inversion Detection

The script implements a sophisticated algorithm to detect inverted ECG leads:

**Methodology:**
1. **Preprocessing**: High-pass filter (0.5 Hz) to remove baseline wander, z-score normalization
2. **QRS Detection**: Uses biosppy library for robust R-peak detection
3. **Polarity Analysis**: Analyzes QRS complex polarity by:
   - Measuring R-wave amplitude at each detected peak
   - Computing signed area (integral) of QRS windows (±40ms around R-peak)
   - Tracking negative QRS ratio across all beats
4. **Decision Rule**: Lead classified as "inverted" if >50% of QRS complexes are negative

**Metrics Computed:**
- `is_inverted`: Boolean classification
- `negative_qrs_ratio`: Percentage of negative QRS complexes
- `mean_qrs_area`: Average signed QRS area
- `mean_r_amplitude`: Average R-wave amplitude
- `r_peaks`: Detected R-peak locations

### 2. Signal Statistics

Per-lead analysis includes:
- Mean and standard deviation of signal values
- Peak counts and prominence
- Percentage of inverted records
- All statistics computed across diagnostic superclasses

### 3. Label Analysis

Distribution analysis at three diagnostic levels:
- **Diagnostic**: 44 unique labels (most granular)
- **Subdiagnostic**: 23 unique labels
- **Superdiagnostic**: 5 unique labels (NORM, MI, STTC, CD, HYP)

## Usage

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scipy tqdm biosppy peakutils
```

### Running the Analysis

```bash
python ptbxl_eda.py
```

This will:
1. Load the PTB-XL dataset from `../data/ptbxl/`
2. Compute signal statistics for all 12 leads
3. Detect inverted leads using QRS polarity analysis
4. Analyze label distributions
5. Generate visualizations and CSV reports
6. Save all outputs to `output/` directory

**Note**: The analysis is computationally intensive (~22K records × 12 leads with biosppy QRS detection). Expect runtime of 30-60 minutes depending on hardware.

### Testing Flipped Detection

To test the flipped detection on a small sample:

```bash
python test_flipped_detection.py
```

## Key Findings

Based on the PTB-XL dataset (21,837 records, 100 Hz sampling):

### Signal Statistics
- Overall mean signal value: ~0.0 (centered)
- Mean number of peaks per lead: ~68.5
- Signal variability differs significantly across leads (precordial leads V2-V4 show higher amplitude)

### Lead Inversion
- Inverted leads are relatively common in the dataset
- Precordial leads (V2-V6) show higher inversion rates
- The robust QRS-based detection provides more accurate classification than simple peak-based methods

### Label Distribution
- **Superdiagnostic level**:
  - NORM: 43.63% (9,528 records)
  - MI (Myocardial Infarction): 25.12% (5,486 records)
  - STTC (ST/T Changes): 24.04% (5,250 records)
  - CD (Conduction Disturbance): 22.47% (4,907 records)
  - HYP (Hypertrophy): 12.16% (2,655 records)

- Multi-label characteristics:
  - Mean labels per record: 1.27-1.41 (depending on hierarchy level)
  - Only 1.86% of records have no diagnostic labels

## Visualizations

### Signal Statistics Plot
Two boxplots showing:
1. Distribution of signal means (mean and std) across all 12 leads
2. Distribution of signal standard deviations (mean and std) across all 12 leads

### Flipped Examples
Sample ECG records with inverted leads:
- All 12 leads displayed
- Inverted leads highlighted in red
- R-peaks marked with red dots
- Annotation showing inversion status and % negative QRS

### Superclass-Specific Plots
For each superdiagnostic class (NORM, MI, STTC, CD, HYP):
- Peak statistics per lead
- Flipped records percentage per lead

## Algorithm Details

### QRS Detection (biosppy)
- Hamilton segmenter algorithm
- Adaptive thresholding for R-peak detection
- Robust to noise and baseline variations

### Polarity Analysis
- QRS window: ±40ms around R-peak (±4 samples at 100 Hz)
- Signed area computed using trapezoidal integration
- Per-beat classification: negative if R-amplitude < 0 OR QRS area < 0

### Fallback Mechanism
If biosppy fails, the algorithm falls back to simple peak detection using scipy's `find_peaks`.

## References

- PTB-XL Database: https://physionet.org/content/ptbxl/
- biosppy Library: https://biosppy.readthedocs.io/
- Hamilton, P. S., & Tompkins, W. J. (1986). Quantitative investigation of QRS detection rules using the MIT/BIH arrhythmia database. IEEE transactions on biomedical engineering, (12), 1157-1165.

## Citation

If you use this EDA pipeline, please cite the PTB-XL dataset:

```
Wagner, P., Strodthoff, N., Bousseljot, R. D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020).
PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 7(1), 154.
```

## Contact

For questions or issues, please refer to the main repository README.
