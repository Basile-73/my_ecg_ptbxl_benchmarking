"""
Exploratory Data Analysis (EDA) for PTB-XL Dataset

This script performs comprehensive EDA on the PTB-XL ECG dataset including:
- Signal statistics (mean, std, peaks)
- Flipped records detection
- Label distributions across diagnostic levels
- Visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, butter, filtfilt
from tqdm import tqdm
import ast
import warnings
warnings.filterwarnings('ignore')

# Import biosppy for ECG processing
try:
    from biosppy.signals import ecg
    BIOSPPY_AVAILABLE = True
except ImportError:
    BIOSPPY_AVAILABLE = False
    print("Warning: biosppy not available. Using fallback method for flipped detection.")

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
from utils import utils

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Constants
SAMPLING_RATE = 100  # Hz
DATA_FOLDER = '/local/home/bamorel/my_ecg_ptbxl_benchmarking/data/ptbxl/'
OUTPUT_FOLDER = '/local/home/bamorel/my_ecg_ptbxl_benchmarking/eda/output/'
LEAD_NAMES = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def preprocess_signal(signal, sampling_rate=100):
    """
    Step 1: Preprocess signal - remove baseline wander and normalize.

    Args:
        signal: 1D ECG signal
        sampling_rate: Sampling rate in Hz

    Returns:
        Preprocessed signal
    """
    # High-pass filter to remove baseline wander (0.5 Hz cutoff)
    nyq = 0.5 * sampling_rate
    cutoff = 0.5 / nyq
    if cutoff >= 1.0:
        cutoff = 0.99

    try:
        b, a = butter(2, cutoff, btype='high')
        filtered = filtfilt(b, a, signal)
    except Exception:
        filtered = signal.copy()

    # Z-score normalization
    mean = np.mean(filtered)
    std = np.std(filtered)
    if std > 1e-6:
        normalized = (filtered - mean) / std
    else:
        normalized = filtered - mean

    return normalized


def detect_qrs_complexes(signal, sampling_rate=100):
    """
    Step 2: Detect QRS complexes using biosppy.

    Args:
        signal: Preprocessed 1D ECG signal
        sampling_rate: Sampling rate in Hz

    Returns:
        r_peaks: Array of R-peak indices
    """
    if not BIOSPPY_AVAILABLE:
        # Fallback: simple peak detection
        try:
            peaks_pos, _ = find_peaks(signal, distance=sampling_rate // 3, prominence=0.3)
            return peaks_pos
        except Exception:
            return np.array([])

    try:
        # Use biosppy ECG processing
        out = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)
        r_peaks = out['rpeaks']
        return r_peaks
    except Exception:
        # Fallback to simple peak detection
        try:
            peaks_pos, _ = find_peaks(signal, distance=sampling_rate // 3, prominence=0.3)
            return peaks_pos
        except Exception:
            return np.array([])


def analyze_qrs_polarity(signal, r_peaks, sampling_rate=100):
    """
    Step 3: Measure QRS polarity per lead.

    Args:
        signal: Preprocessed 1D ECG signal
        r_peaks: Array of R-peak indices
        sampling_rate: Sampling rate in Hz

    Returns:
        Dictionary with polarity metrics
    """
    if len(r_peaks) == 0:
        return {
            'is_inverted': False,
            'negative_qrs_ratio': 0.0,
            'mean_qrs_area': 0.0,
            'mean_r_amplitude': 0.0,
            'r_peaks': [],
            'qrs_polarities': []
        }

    # Define QRS window (±40ms around R-peak for 100Hz = ±4 samples)
    qrs_half_window = int(0.04 * sampling_rate)  # 40ms

    qrs_areas = []
    r_amplitudes = []
    qrs_polarities = []

    for r_peak in r_peaks:
        # Define QRS window
        qrs_start = max(0, r_peak - qrs_half_window)
        qrs_end = min(len(signal), r_peak + qrs_half_window)

        # Extract QRS window
        qrs_window = signal[qrs_start:qrs_end]

        # Compute signed area (integral)
        qrs_area = np.trapz(qrs_window)
        qrs_areas.append(qrs_area)

        # R-wave amplitude (value at R-peak)
        r_amplitude = signal[r_peak]
        r_amplitudes.append(r_amplitude)

        # Determine polarity: negative if R-amplitude < 0 or area < 0
        is_negative = (r_amplitude < 0) or (qrs_area < 0)
        qrs_polarities.append(is_negative)

    # Calculate metrics
    negative_qrs_ratio = np.mean(qrs_polarities) if len(qrs_polarities) > 0 else 0.0
    mean_qrs_area = np.mean(qrs_areas) if len(qrs_areas) > 0 else 0.0
    mean_r_amplitude = np.mean(r_amplitudes) if len(r_amplitudes) > 0 else 0.0

    # Step 4: Decision rule - if majority of QRS complexes are negative, lead is inverted
    is_inverted = negative_qrs_ratio > 0.5

    return {
        'is_inverted': is_inverted,
        'negative_qrs_ratio': negative_qrs_ratio,
        'mean_qrs_area': mean_qrs_area,
        'mean_r_amplitude': mean_r_amplitude,
        'r_peaks': r_peaks,
        'qrs_polarities': qrs_polarities
    }


def detect_flipped_records(signal, sampling_rate=100):
    """
    Detect if a signal lead is flipped (inverted) using robust QRS polarity analysis.

    Steps:
    1. Preprocess: Remove baseline wander and normalize
    2. Detect QRS complexes using biosppy
    3. Measure QRS polarity (R-wave amplitude and QRS area)
    4. Decision: If majority of QRS are negative, lead is inverted

    Args:
        signal: 1D ECG signal
        sampling_rate: Sampling rate in Hz

    Returns:
        Dictionary with inversion analysis results
    """
    try:
        # Step 1: Preprocess
        processed_signal = preprocess_signal(signal, sampling_rate)

        # Step 2: Detect QRS complexes
        r_peaks = detect_qrs_complexes(processed_signal, sampling_rate)

        # Step 3 & 4: Analyze polarity and make decision
        polarity_results = analyze_qrs_polarity(processed_signal, r_peaks, sampling_rate)

        return polarity_results
    except Exception as e:
        # Fallback
        return {
            'is_inverted': False,
            'negative_qrs_ratio': 0.0,
            'mean_qrs_area': 0.0,
            'mean_r_amplitude': 0.0,
            'r_peaks': [],
            'qrs_polarities': []
        }


def detect_flipped_records_simple(signal, sampling_rate=100):
    """
    Simple version that returns just a boolean for backward compatibility.
    """
    result = detect_flipped_records(signal, sampling_rate)
    return result['is_inverted']


def compute_signal_statistics(data, sampling_rate=100):
    """
    Compute comprehensive statistics for ECG signals.

    Args:
        data: Array of shape (n_samples, time_steps, n_leads)
        sampling_rate: Sampling rate in Hz

    Returns:
        DataFrame with statistics per lead, and lists of flipped record indices
    """
    n_samples, time_steps, n_leads = data.shape

    stats_per_lead = {
        'lead': [],
        'mean_signal_mean': [],
        'std_signal_mean': [],
        'mean_signal_std': [],
        'std_signal_std': [],
        'mean_num_peaks': [],
        'std_num_peaks': [],
        'mean_peak_prominence': [],
        'std_peak_prominence': [],
        'num_flipped_records': [],
        'pct_flipped_records': []
    }

    # Store indices of flipped records per lead
    flipped_indices_per_lead = {}

    print("Computing signal statistics per lead...")
    for lead_idx in tqdm(range(n_leads)):
        lead_name = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f'Lead_{lead_idx}'

        signal_means = []
        signal_stds = []
        num_peaks_list = []
        peak_prominences = []
        flipped_count = 0
        flipped_indices = []

        for sample_idx in range(n_samples):
            signal = data[sample_idx, :, lead_idx]

            # Basic statistics
            signal_means.append(np.mean(signal))
            signal_stds.append(np.std(signal))

            # Peak detection
            try:
                peaks_up, props_up = find_peaks(signal, distance=sampling_rate // 5)
                peaks_down, props_down = find_peaks(-signal, distance=sampling_rate // 5)
                total_peaks = len(peaks_up) + len(peaks_down)
                num_peaks_list.append(total_peaks)

                # Peak prominence
                if 'prominences' in props_up and len(props_up['prominences']) > 0:
                    peak_prominences.extend(props_up['prominences'])
            except Exception:
                num_peaks_list.append(0)

            # Flipped detection using new robust method
            flip_result = detect_flipped_records(signal, sampling_rate)
            if flip_result['is_inverted']:
                flipped_count += 1
                flipped_indices.append(sample_idx)

        # Store flipped indices
        flipped_indices_per_lead[lead_name] = flipped_indices

        # Aggregate statistics
        stats_per_lead['lead'].append(lead_name)
        stats_per_lead['mean_signal_mean'].append(np.mean(signal_means))
        stats_per_lead['std_signal_mean'].append(np.std(signal_means))
        stats_per_lead['mean_signal_std'].append(np.mean(signal_stds))
        stats_per_lead['std_signal_std'].append(np.std(signal_stds))
        stats_per_lead['mean_num_peaks'].append(np.mean(num_peaks_list))
        stats_per_lead['std_num_peaks'].append(np.std(num_peaks_list))
        stats_per_lead['mean_peak_prominence'].append(
            np.mean(peak_prominences) if len(peak_prominences) > 0 else 0
        )
        stats_per_lead['std_peak_prominence'].append(
            np.std(peak_prominences) if len(peak_prominences) > 0 else 0
        )
        stats_per_lead['num_flipped_records'].append(flipped_count)
        stats_per_lead['pct_flipped_records'].append(100 * flipped_count / n_samples)

    return pd.DataFrame(stats_per_lead), flipped_indices_per_lead


def analyze_label_distributions(labels_df, datafolder):
    """
    Analyze label distributions across different diagnostic levels.

    Returns:
        Dictionary with distribution statistics
    """
    print("\nAnalyzing label distributions...")

    distributions = {}

    # Diagnostic level
    labels_diag = utils.compute_label_aggregations(labels_df.copy(), datafolder, 'diagnostic')
    if 'diagnostic' in labels_diag.columns:
        diag_counts = pd.Series([item for sublist in labels_diag.diagnostic for item in sublist]).value_counts()
        distributions['diagnostic'] = diag_counts

    # Subdiagnostic level
    labels_subdiag = utils.compute_label_aggregations(labels_df.copy(), datafolder, 'subdiagnostic')
    if 'subdiagnostic' in labels_subdiag.columns:
        subdiag_counts = pd.Series([item for sublist in labels_subdiag.subdiagnostic for item in sublist]).value_counts()
        distributions['subdiagnostic'] = subdiag_counts

    # Superdiagnostic level
    labels_superdiag = utils.compute_label_aggregations(labels_df.copy(), datafolder, 'superdiagnostic')
    if 'superdiagnostic' in labels_superdiag.columns:
        superdiag_counts = pd.Series([item for sublist in labels_superdiag.superdiagnostic for item in sublist]).value_counts()
        distributions['superdiagnostic'] = superdiag_counts

    return distributions, labels_diag, labels_subdiag, labels_superdiag


def plot_signal_statistics(stats_df, output_folder):
    """
    Create visualizations for signal statistics using boxplots.
    """
    print("\nCreating signal statistics visualizations...")

    plots_folder = os.path.join(output_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)

    # Create figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Boxplot of signal means per lead
    mean_data = [stats_df['mean_signal_mean'].values, stats_df['std_signal_mean'].values]
    bp1 = axes[0].boxplot(mean_data, labels=['Mean', 'Std'], patch_artist=True,
                          widths=0.6, showmeans=True,
                          boxprops=dict(facecolor='steelblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
    axes[0].set_title('Distribution of Signal Means Across Leads', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xlabel('Statistic Type', fontsize=12)

    # Subplot 2: Boxplot of signal stds per lead
    std_data = [stats_df['mean_signal_std'].values, stats_df['std_signal_std'].values]
    bp2 = axes[1].boxplot(std_data, labels=['Mean', 'Std'], patch_artist=True,
                          widths=0.6, showmeans=True,
                          boxprops=dict(facecolor='coral', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
    axes[1].set_title('Distribution of Signal Stds Across Leads', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xlabel('Statistic Type', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'signal_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def compute_statistics_by_superclass(data, labels_superdiag, sampling_rate=100):
    """
    Compute peak and flipped statistics grouped by superdiagnostic class.
    """
    print("\nComputing statistics by superdiagnostic class...")

    superclass_stats = {}

    # Get unique superclasses
    all_superclasses = set()
    for superdiag_list in labels_superdiag['superdiagnostic']:
        all_superclasses.update(superdiag_list)

    for superclass in sorted(all_superclasses):
        print(f"  Processing {superclass}...")

        # Get indices of records with this superclass
        indices = []
        for idx, superdiag_list in enumerate(labels_superdiag['superdiagnostic']):
            if superclass in superdiag_list:
                indices.append(idx)

        if len(indices) == 0:
            continue

        # Get data for these indices
        class_data = data[indices]
        n_samples, time_steps, n_leads = class_data.shape

        stats_per_lead = {
            'lead': [],
            'mean_num_peaks': [],
            'std_num_peaks': [],
            'pct_flipped_records': []
        }

        for lead_idx in range(n_leads):
            lead_name = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f'Lead_{lead_idx}'

            num_peaks_list = []
            flipped_count = 0

            for sample_idx in range(n_samples):
                signal = class_data[sample_idx, :, lead_idx]

                # Peak detection
                try:
                    peaks_up, _ = find_peaks(signal, distance=sampling_rate // 5)
                    peaks_down, _ = find_peaks(-signal, distance=sampling_rate // 5)
                    total_peaks = len(peaks_up) + len(peaks_down)
                    num_peaks_list.append(total_peaks)
                except Exception:
                    num_peaks_list.append(0)

                # Flipped detection using new robust method
                flip_result = detect_flipped_records(signal, sampling_rate)
                if flip_result['is_inverted']:
                    flipped_count += 1

            stats_per_lead['lead'].append(lead_name)
            stats_per_lead['mean_num_peaks'].append(np.mean(num_peaks_list))
            stats_per_lead['std_num_peaks'].append(np.std(num_peaks_list))
            stats_per_lead['pct_flipped_records'].append(100 * flipped_count / n_samples)

        superclass_stats[superclass] = pd.DataFrame(stats_per_lead)

    return superclass_stats


def plot_statistics_by_superclass(superclass_stats, output_folder):
    """
    Create plots for peak and flipped statistics by superdiagnostic class.
    """
    print("\nCreating superclass-specific visualizations...")

    superclass_folder = os.path.join(output_folder, 'plots', 'by_superclass')
    os.makedirs(superclass_folder, exist_ok=True)

    for superclass, stats_df in superclass_stats.items():
        # Peak statistics plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.bar(stats_df['lead'], stats_df['mean_num_peaks'],
               yerr=stats_df['std_num_peaks'], capsize=5, color='steelblue', alpha=0.7)
        ax.set_title(f'Mean Number of Peaks per Lead - {superclass}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Lead')
        ax.set_ylabel('Number of Peaks')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(superclass_folder, f'peak_statistics_{superclass}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Flipped records plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        bars = ax.bar(stats_df['lead'], stats_df['pct_flipped_records'], color='crimson', alpha=0.7)
        ax.set_title(f'Percentage of Flipped Records per Lead - {superclass}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Lead')
        ax.set_ylabel('Percentage (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(superclass_folder, f'flipped_records_{superclass}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()


def plot_flipped_signal_examples(data, flipped_indices_per_lead, output_folder, n_examples=10):
    """
    Plot examples of flipped signals, showing all leads for each record.
    Highlights R-peaks detected by biosppy QRS detection.
    """
    print(f"\nPlotting {n_examples} examples of flipped signals (all leads with R-peaks)...")

    examples_folder = os.path.join(output_folder, 'plots', 'flipped_examples')
    os.makedirs(examples_folder, exist_ok=True)

    # Collect all (record_idx, lead_idx) pairs that are flipped
    flipped_pairs = []
    for lead_idx, lead_name in enumerate(LEAD_NAMES):
        for rec_idx in flipped_indices_per_lead.get(lead_name, []):
            flipped_pairs.append((rec_idx, lead_idx))

    # Select up to n_examples unique records
    unique_rec_indices = list({rec_idx for rec_idx, _ in flipped_pairs})
    if len(unique_rec_indices) == 0:
        print("  No flipped records found to plot.")
        return

    selected_indices = np.random.choice(unique_rec_indices, min(n_examples, len(unique_rec_indices)), replace=False)

    for rec_idx in selected_indices:
        signal = data[rec_idx]
        fig, axes = plt.subplots(12, 1, figsize=(18, 22))
        fig.suptitle(f'Flipped ECG Example - Record {rec_idx}\n(Red leads = inverted, R-peaks marked)',
                     fontsize=16, fontweight='bold')

        for lead_idx in range(min(12, signal.shape[1])):
            lead_name = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f'Lead_{lead_idx}'
            is_flipped = rec_idx in flipped_indices_per_lead.get(lead_name, [])
            color = 'crimson' if is_flipped else 'navy'

            # Get the raw signal for this lead
            raw_signal = signal[:, lead_idx]

            # Plot the signal
            axes[lead_idx].plot(raw_signal, linewidth=0.8, color=color, label='ECG signal')
            axes[lead_idx].set_ylabel(lead_name, fontweight='bold', fontsize=11)
            axes[lead_idx].grid(True, alpha=0.3)
            axes[lead_idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

            # Detect and plot R-peaks using the new robust method
            try:
                flip_result = detect_flipped_records(raw_signal, SAMPLING_RATE)
                r_peaks = flip_result['r_peaks']

                if len(r_peaks) > 0:
                    # Plot R-peaks
                    axes[lead_idx].plot(r_peaks, raw_signal[r_peaks], 'ro',
                                       markersize=6, label='R-peaks', zorder=5)

                    # Add text annotation for inversion status
                    if is_flipped:
                        axes[lead_idx].text(0.98, 0.95,
                                          f'INVERTED\n({flip_result["negative_qrs_ratio"]*100:.1f}% neg QRS)',
                                          transform=axes[lead_idx].transAxes,
                                          fontsize=9, verticalalignment='top', horizontalalignment='right',
                                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except Exception as e:
                pass

            if lead_idx == 0:
                axes[lead_idx].legend(loc='upper left', fontsize=9)

            if lead_idx == 11:
                axes[lead_idx].set_xlabel('Time (samples)', fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(examples_folder, f'flipped_ecg_example_{rec_idx}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: flipped_ecg_example_{rec_idx}.png")


def plot_label_distributions(distributions, output_folder):
    # No-op: no label distribution plots/bar charts generated
    pass


def plot_sample_ecgs(data, labels_df, output_folder, n_samples=5):
    """
    Plot sample ECG signals for visualization.
    """
    print("\nPlotting sample ECGs...")

    samples_folder = os.path.join(output_folder, 'plots', 'sample_ecgs')
    os.makedirs(samples_folder, exist_ok=True)

    # Select random samples
    indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)

    for idx in indices:
        signal = data[idx]

        fig, axes = plt.subplots(12, 1, figsize=(16, 20))
        fig.suptitle(f'ECG Sample {idx} - All 12 Leads', fontsize=16, fontweight='bold')

        for lead_idx in range(min(12, signal.shape[1])):
            lead_name = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f'Lead_{lead_idx}'
            axes[lead_idx].plot(signal[:, lead_idx], linewidth=0.8)
            axes[lead_idx].set_ylabel(lead_name, fontweight='bold')
            axes[lead_idx].grid(True, alpha=0.3)

            # Highlight peaks
            try:
                peaks_up, _ = find_peaks(signal[:, lead_idx], distance=SAMPLING_RATE // 5)
                axes[lead_idx].plot(peaks_up, signal[peaks_up, lead_idx], 'ro', markersize=4)
            except Exception:
                pass

            if lead_idx == 11:
                axes[lead_idx].set_xlabel('Time (samples)')

        plt.tight_layout()
        plt.savefig(os.path.join(samples_folder, f'sample_ecg_{idx}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()


def generate_summary_report(stats_df, distributions, labels_df, output_folder):
    """
    Generate a comprehensive text summary report.
    """
    print("\nGenerating summary report...")

    report_path = os.path.join(output_folder, 'data', 'eda_summary_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PTB-XL EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Dataset overview
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total number of records: {len(labels_df)}\n")
        f.write(f"Number of leads: {len(stats_df)}\n")
        f.write(f"Sampling rate: {SAMPLING_RATE} Hz\n\n")

        # Signal statistics
        f.write("SIGNAL STATISTICS (Across All Leads)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall mean signal value: {stats_df['mean_signal_mean'].mean():.4f} ± {stats_df['mean_signal_mean'].std():.4f}\n")
        f.write(f"Overall mean signal std: {stats_df['mean_signal_std'].mean():.4f} ± {stats_df['mean_signal_std'].std():.4f}\n")
        f.write(f"Overall mean number of peaks: {stats_df['mean_num_peaks'].mean():.2f} ± {stats_df['mean_num_peaks'].std():.2f}\n")
        f.write(f"Total flipped records percentage: {stats_df['pct_flipped_records'].mean():.2f}%\n\n")

        # Per-lead statistics
        f.write("PER-LEAD STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\n")

        # Label distributions
        f.write("LABEL DISTRIBUTIONS\n")
        f.write("-" * 80 + "\n")

        for level_name, counts in distributions.items():
            f.write(f"\n{level_name.upper()} Level:\n")
            f.write(f"  Total unique labels: {len(counts)}\n")
            f.write(f"  Most common labels:\n")
            for label, count in counts.head(10).items():
                f.write(f"    - {label}: {count} ({100*count/len(labels_df):.2f}%)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Summary report saved to: {report_path}")


def main():
    """
    Main execution function for PTB-XL EDA.
    """
    print("=" * 80)
    print("PTB-XL EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    # Create output folder structure
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'data'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'plots'), exist_ok=True)

    # Load dataset
    print(f"\nLoading PTB-XL dataset from {DATA_FOLDER}...")
    data, labels = utils.load_dataset(DATA_FOLDER, SAMPLING_RATE)
    print(f"Loaded {len(data)} records with shape {data[0].shape}")
    print(f"Labels shape: {labels.shape}")

    # Compute signal statistics
    stats_df, flipped_indices_per_lead = compute_signal_statistics(data, SAMPLING_RATE)
    print("\nSignal Statistics Summary:")
    print(stats_df.to_string(index=False))

    # Save statistics to CSV
    stats_path = os.path.join(OUTPUT_FOLDER, 'data', 'signal_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSignal statistics saved to: {stats_path}")

    # Analyze label distributions
    distributions, labels_diag, labels_subdiag, labels_superdiag = analyze_label_distributions(
        labels, DATA_FOLDER
    )

    # Save label distributions
    data_folder = os.path.join(OUTPUT_FOLDER, 'data')
    for level_name, counts in distributions.items():
        dist_path = os.path.join(data_folder, f'distribution_{level_name}.csv')
        counts.to_csv(dist_path, header=['count'])
        print(f"Label distribution ({level_name}) saved to: {dist_path}")

    # Create visualizations
    plot_signal_statistics(stats_df, OUTPUT_FOLDER)
    plot_label_distributions(distributions, OUTPUT_FOLDER)
    plot_sample_ecgs(data, labels, OUTPUT_FOLDER, n_samples=3)

    # Plot flipped signal examples
    plot_flipped_signal_examples(data, flipped_indices_per_lead, OUTPUT_FOLDER, n_examples=10)

    # Compute and plot statistics by superclass
    superclass_stats = compute_statistics_by_superclass(data, labels_superdiag, SAMPLING_RATE)
    plot_statistics_by_superclass(superclass_stats, OUTPUT_FOLDER)

    # Save superclass statistics to CSV
    for superclass, stats in superclass_stats.items():
        stats_path = os.path.join(data_folder, f'stats_{superclass}.csv')
        stats.to_csv(stats_path, index=False)
        print(f"Superclass statistics ({superclass}) saved to: {stats_path}")

    # Generate summary report
    generate_summary_report(stats_df, distributions, labels, OUTPUT_FOLDER)

    # Additional analysis: Multi-label statistics
    print("\n" + "=" * 80)
    print("MULTI-LABEL STATISTICS")
    print("=" * 80)

    # Diagnostic level
    if 'diagnostic' in labels_diag.columns:
        diag_lengths = labels_diag['diagnostic_len']
        print(f"\nDiagnostic level:")
        print(f"  Mean labels per record: {diag_lengths.mean():.2f} ± {diag_lengths.std():.2f}")
        print(f"  Max labels per record: {diag_lengths.max()}")
        print(f"  Records with no labels: {(diag_lengths == 0).sum()} ({100*(diag_lengths == 0).sum()/len(diag_lengths):.2f}%)")

    # Subdiagnostic level
    if 'subdiagnostic' in labels_subdiag.columns:
        subdiag_lengths = labels_subdiag['subdiagnostic_len']
        print(f"\nSubdiagnostic level:")
        print(f"  Mean labels per record: {subdiag_lengths.mean():.2f} ± {subdiag_lengths.std():.2f}")
        print(f"  Max labels per record: {subdiag_lengths.max()}")
        print(f"  Records with no labels: {(subdiag_lengths == 0).sum()} ({100*(subdiag_lengths == 0).sum()/len(subdiag_lengths):.2f}%)")

    # Superdiagnostic level
    if 'superdiagnostic' in labels_superdiag.columns:
        superdiag_lengths = labels_superdiag['superdiagnostic_len']
        print(f"\nSuperdiagnostic level:")
        print(f"  Mean labels per record: {superdiag_lengths.mean():.2f} ± {superdiag_lengths.std():.2f}")
        print(f"  Max labels per record: {superdiag_lengths.max()}")
        print(f"  Records with no labels: {(superdiag_lengths == 0).sum()} ({100*(superdiag_lengths == 0).sum()/len(superdiag_lengths):.2f}%)")

    print("\n" + "=" * 80)
    print("EDA COMPLETE!")
    print(f"All results and visualizations saved to: {OUTPUT_FOLDER}")
    print("=" * 80)


if __name__ == "__main__":
    main()
