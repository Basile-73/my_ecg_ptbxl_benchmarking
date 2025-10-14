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
                     stats: dict = None) -> Tuple[np.ndarray, dict]:
    """
    Normalize ECG signals.
    """
    if stats is None:
        # Compute stats
        if method == 'standard':
            mean = np.mean(data)
            std = np.std(data)
            normalized = (data - mean) / (std + 1e-8)
            stats = {'mean': float(mean), 'std': float(std), 'method': method}
        elif method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            stats = {'min': float(min_val), 'max': float(max_val), 'method': method}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    else:
        # Apply existing stats
        if stats['method'] == 'standard':
            normalized = (data - stats['mean']) / (stats['std'] + 1e-8)
        elif stats['method'] == 'minmax':
            normalized = (data - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {stats['method']}")
    
    return normalized, stats
