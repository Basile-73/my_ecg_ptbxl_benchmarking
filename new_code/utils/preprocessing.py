"""
ECG Preprocessing utilities.
Adapted from mycode/denoising/denoising_utils/preprocessing.py
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def remove_bad_labels_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records with problematic annotations from PTB-XL metadata DataFrame.
    Following Hu et al. 2024: exclude signals with baseline drift, static noise,
    burst noise, and electrode problems.

    Args:
        df: PTB-XL metadata DataFrame with noise annotation columns

    Returns:
        Filtered DataFrame with bad labels removed
    """
    noise_cols = ['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems']
    mask = df[noise_cols].isna().all(axis=1)
    filtered_df = df[mask].reset_index(drop=True)
    print(f'remove_bad_labels: removed {(~mask).sum()} records, kept {mask.sum()}')
    return filtered_df


def select_best_lead_for_record(signal: np.ndarray, sampling_rate: int) -> int:
    """
    Select the lead with the fewest peaks for a single record.
    Following Dias et al. 2024: leads with fewer peaks likely have less noise.

    Args:
        signal: Single record signal array of shape (n_timepoints, n_channels)
        sampling_rate: Sampling rate in Hz

    Returns:
        Index of the best (least noisy) lead
    """
    counts = []
    for ch in range(signal.shape[1]):
        sig = signal[:, ch]
        try:
            peaks_up, _ = find_peaks(sig, distance=sampling_rate // 5)
            peaks_down, _ = find_peaks(-sig, distance=sampling_rate // 5)
            counts.append(len(peaks_up) + len(peaks_down))
        except Exception:
            counts.append(np.inf)
    return int(np.argmin(counts))


def moving_average_filter(X: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a moving average filter to each row of a 2-D signal array.

    Smooths the ECG to reduce high-frequency noise, which can be effective
    for ST-T wave start and endpoint detection.

    Reference:
        Alqudah, A. and Alqudah, A. (2021). Morphological Based Method for
        Automated Extraction and Classification of ECG ST-T Wave.
        Instrumentation Mesure Metrologie, 20(1), 15-22.
        doi: 10.18280/i2m.200103

    Args:
        X: Signal array of shape (n_samples, signal_length).
        window_size: Number of samples in the averaging window.

    Returns:
        Smoothed array with the same shape as X.

        We use a window of 10 instead of 7 because our signal is sampled at 360 Hz versus 250 Hz in the reference paper, so we need a slightly larger window to achieve similar smoothing.
    """
    kernel = np.ones(window_size) / window_size
    return np.array([np.convolve(row, kernel, mode='same') for row in X])
