"""
Utility functions for PTB-XL EDA

This module contains utility functions for:
- Signal preprocessing
- QRS detection
- Flipped signal detection
- Signal statistics computation
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from tqdm import tqdm

# Import biosppy for ECG processing
try:
    from biosppy.signals import ecg
    BIOSPPY_AVAILABLE = True
except ImportError:
    BIOSPPY_AVAILABLE = False
    print("Warning: biosppy not available. Using fallback method for flipped detection.")


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


def compute_signal_statistics(data, lead_names, sampling_rate=100):
    """
    Compute comprehensive statistics for ECG signals.

    Args:
        data: Array of shape (n_samples, time_steps, n_leads)
        lead_names: List of lead names
        sampling_rate: Sampling rate in Hz

    Returns:
        DataFrame with statistics per lead, and lists of flipped record indices
    """
    n_samples, time_steps, n_leads = data.shape

    stats_per_lead = {
        'lead': [],
        'mean_signal_mean': [],
        'std_signal_mean': [],
        'means': [],
        'mean_signal_std': [],
        'std_signal_std': [],
        'stds': [],
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
        lead_name = lead_names[lead_idx] if lead_idx < len(lead_names) else f'Lead_{lead_idx}'

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
        stats_per_lead['means'].append(signal_means)
        stats_per_lead['mean_signal_std'].append(np.mean(signal_stds))
        stats_per_lead['std_signal_std'].append(np.std(signal_stds))
        stats_per_lead['stds'].append(signal_stds)
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


def compute_statistics_by_superclass(data, labels_superdiag, lead_names, sampling_rate=100):
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
            lead_name = lead_names[lead_idx] if lead_idx < len(lead_names) else f'Lead_{lead_idx}'

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
