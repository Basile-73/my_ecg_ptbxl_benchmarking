"""
ECG Preprocessing utilities for denoising experiments.
Based on selection_denoising.ipynb
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from typing import Tuple


def remove_bad_labels(data: np.ndarray, labels: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Remove ECG signals with problematic annotations.
    Following Hu et al. 2024: exclude signals with baseline drift, static noise,
    burst noise, and electrode problems.
    """
    good_labels = labels[
        labels[['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems']]
        .isna().all(axis=1)
    ]
    good_signals = data[good_labels.index.to_numpy()]
    print(f'Removed {len(labels) - len(good_labels)} samples, kept {len(good_labels)}')
    return good_signals, good_labels


def select_best_lead(data: np.ndarray, sampling_rate: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the single lead with the fewest peaks for each record.
    Following Dias et al. 2024: leads with fewer peaks likely have less noise.
    """
    selected_indices = []
    selected_data = []

    for record in data:
        counts = []
        for ch in range(record.shape[1]):
            sig = record[:, ch]
            try:
                peaks_up, _ = find_peaks(sig, distance=sampling_rate // 5)
                peaks_down, _ = find_peaks(-sig, distance=sampling_rate // 5)
                counts.append(len(peaks_up) + len(peaks_down))
            except Exception:
                counts.append(np.inf)

        best_ch = np.argmin(counts)
        selected_indices.append(best_ch)
        selected_data.append(record[:, best_ch])

    return np.array(selected_data)[..., np.newaxis], np.array(selected_indices)


def bandpass_filter(data: np.ndarray, lowcut: float = 1.0, highcut: float = 45.0,
                    fs: int = 500, order: int = 2) -> np.ndarray:
    """
    Apply 2nd-order Butterworth bandpass filter to remove baseline wander.
    Following Dias et al. 2024: 1-45 Hz bandpass.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    if data.ndim == 2:
        filtered = np.array([filtfilt(b, a, record) for record in data])
    else:
        filtered = np.array([filtfilt(b, a, record, axis=0) for record in data])

    return filtered


def normalize_signals(data: np.ndarray, method: str = 'standard',
                     stats: dict = None, axis: str = 'channel') -> Tuple[np.ndarray, dict]:
    """
    Normalize ECG signals.

    Args:
        data: ECG data array (n_samples, n_timepoints) or (n_samples, n_timepoints, n_channels)
        method: 'standard' (z-score) or 'minmax' (0-1 scaling)
        stats: Optional pre-computed statistics to apply (for val/test sets)
        axis: Normalization axis strategy:
            - 'channel' (default): Per-record, per-channel normalization
                * 3D data: Each channel of each record normalized independently (mean=0, std=1 per channel)
                * 2D data: Each record normalized independently (mean=0, std=1 per record)
            - 'all': Global normalization across all dimensions (single mean/std for entire dataset)

    Returns:
        Tuple of (normalized_data, statistics_dict)
    """
    if stats is None:
        # Compute stats
        if method == 'standard':
            if axis == 'channel':
                # Per-record, per-channel normalization
                if data.ndim == 3:
                    # 3D: compute per record, per channel (normalize along time axis only)
                    # Each channel of each record gets its own mean/std
                    mean = np.mean(data, axis=1, keepdims=True)  # Shape: (n_samples, 1, n_channels)
                    std = np.std(data, axis=1, keepdims=True)    # Shape: (n_samples, 1, n_channels)
                else:
                    # 2D: compute per record (normalize along time axis only)
                    mean = np.mean(data, axis=1, keepdims=True)  # Shape: (n_samples, 1)
                    std = np.std(data, axis=1, keepdims=True)    # Shape: (n_samples, 1)
            elif axis == 'all':
                # Global normalization across all dimensions
                mean = np.mean(data)
                std = np.std(data)
            else:
                raise ValueError(f"Unknown axis strategy: {axis}. Use 'channel' or 'all'")

            normalized = (data - mean) / (std + 1e-8)

            # For 'channel' mode, we don't save stats since each sample has its own
            # For 'all' mode, we save global stats for applying to val/test
            if axis == 'all':
                stats = {'mean': float(mean), 'std': float(std), 'method': method, 'axis': axis}
            else:
                stats = {'method': method, 'axis': axis}

        elif method == 'minmax':
            if axis == 'channel':
                # Per-record, per-channel normalization
                if data.ndim == 3:
                    # 3D: compute per record, per channel (normalize along time axis only)
                    min_val = np.min(data, axis=1, keepdims=True)  # Shape: (n_samples, 1, n_channels)
                    max_val = np.max(data, axis=1, keepdims=True)  # Shape: (n_samples, 1, n_channels)
                else:
                    # 2D: compute per record (normalize along time axis only)
                    min_val = np.min(data, axis=1, keepdims=True)  # Shape: (n_samples, 1)
                    max_val = np.max(data, axis=1, keepdims=True)  # Shape: (n_samples, 1)
            elif axis == 'all':
                # Global normalization across all dimensions
                min_val = np.min(data)
                max_val = np.max(data)
            else:
                raise ValueError(f"Unknown axis strategy: {axis}. Use 'channel' or 'all'")

            normalized = (data - min_val) / (max_val - min_val + 1e-8)

            # For 'channel' mode, we don't save stats since each sample has its own
            # For 'all' mode, we save global stats for applying to val/test
            if axis == 'all':
                stats = {'min': float(min_val), 'max': float(max_val), 'method': method, 'axis': axis}
            else:
                stats = {'method': method, 'axis': axis}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    else:
        # Apply existing stats
        axis_strategy = stats.get('axis', 'channel')  # Default to 'channel' for backward compatibility

        if axis_strategy == 'channel':
            # For 'channel' mode, compute stats per-record (same as training)
            # Each validation/test sample normalized independently
            if stats['method'] == 'standard':
                if data.ndim == 3:
                    mean = np.mean(data, axis=1, keepdims=True)
                    std = np.std(data, axis=1, keepdims=True)
                else:
                    mean = np.mean(data, axis=1, keepdims=True)
                    std = np.std(data, axis=1, keepdims=True)
                normalized = (data - mean) / (std + 1e-8)

            elif stats['method'] == 'minmax':
                if data.ndim == 3:
                    min_val = np.min(data, axis=1, keepdims=True)
                    max_val = np.max(data, axis=1, keepdims=True)
                else:
                    min_val = np.min(data, axis=1, keepdims=True)
                    max_val = np.max(data, axis=1, keepdims=True)
                normalized = (data - min_val) / (max_val - min_val + 1e-8)
            else:
                raise ValueError(f"Unknown normalization method: {stats['method']}")

        elif axis_strategy == 'all':
            # For 'all' mode, use global stats from training set
            if stats['method'] == 'standard':
                mean = stats['mean']
                std = stats['std']
                normalized = (data - mean) / (std + 1e-8)

            elif stats['method'] == 'minmax':
                min_val = stats['min']
                max_val = stats['max']
                normalized = (data - min_val) / (max_val - min_val + 1e-8)
            else:
                raise ValueError(f"Unknown normalization method: {stats['method']}")
        else:
            raise ValueError(f"Unknown axis strategy: {axis_strategy}")

    return normalized, stats
