"""
Quick test script to verify the new flipped lead detection method.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
from utils import utils

# Import detection function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ptbxl_eda import detect_flipped_records, preprocess_signal, detect_qrs_complexes, BIOSPPY_AVAILABLE

print("=" * 80)
print("Testing Flipped Lead Detection")
print("=" * 80)
print(f"Biosppy available: {BIOSPPY_AVAILABLE}")

# Load a small sample of PTB-XL data
DATA_FOLDER = '/local/home/bamorel/my_ecg_ptbxl_benchmarking/data/ptbxl/'
SAMPLING_RATE = 100

print("\nLoading PTB-XL dataset...")
data, labels = utils.load_dataset(DATA_FOLDER, SAMPLING_RATE)
print(f"Loaded {len(data)} records")

# Test on a few random records
n_test = 5
test_indices = np.random.choice(len(data), n_test, replace=False)

print(f"\nTesting on {n_test} random records...")
for rec_idx in test_indices:
    print(f"\n--- Record {rec_idx} ---")
    signal = data[rec_idx]

    # Test on Lead II (index 1) - typically a good lead for QRS detection
    lead_idx = 1
    lead_signal = signal[:, lead_idx]

    print(f"  Lead II signal shape: {lead_signal.shape}")
    print(f"  Signal range: [{lead_signal.min():.3f}, {lead_signal.max():.3f}]")

    # Run detection
    result = detect_flipped_records(lead_signal, SAMPLING_RATE)

    print(f"  Is inverted: {result['is_inverted']}")
    print(f"  Negative QRS ratio: {result['negative_qrs_ratio']:.2%}")
    print(f"  Mean QRS area: {result['mean_qrs_area']:.3f}")
    print(f"  Mean R amplitude: {result['mean_r_amplitude']:.3f}")
    print(f"  Number of R-peaks detected: {len(result['r_peaks'])}")

print("\n" + "=" * 80)
print("Test completed successfully!")
print("=" * 80)
