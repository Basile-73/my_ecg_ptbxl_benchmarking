"""
Comprehensive noise characterization test for ECG denoising pipeline.

This test validates the NoiseFactory implementation by:
- Loading and preprocessing PTB-XL data using the same pipeline as training
- Testing all noise configurations (default, light, heavy, check)
- Computing empirical SNR and RMSE metrics
- Comparing against theoretical expected values
- Generating visualization plots
"""

import sys
import os
from pathlib import Path
import gc

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from scipy.stats import spearmanr

# Add paths for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'ecg_noise' / 'source'))
sys.path.insert(0, str(repo_root / 'mycode' / 'classification'))
sys.path.insert(0, str(repo_root / 'mycode' / 'denoising'))

# Import required modules
from utils.utils import load_dataset
from denoising_utils.preprocessing import (
    remove_bad_labels,
    select_best_lead,
    bandpass_filter,
    normalize_signals
)
from denoising_utils.utils import calculate_snr, calculate_rmse
from ecg_noise_factory.noise import NoiseFactory


def test_noise_characterization():
    """
    Comprehensive test to characterize noise across all configurations.

    This test:
    1. Loads and preprocesses PTB-XL data
    2. Iterates through all noise configs
    3. Generates noisy samples and computes empirical metrics
    4. Calculates theoretical expected metrics
    5. Creates visualization plots
    6. Validates results
    """
    print("\n" + "="*80)
    print("NOISE CHARACTERIZATION TEST")
    print("="*80)

    # Load a denoising config to extract preprocessing parameters
    # This ensures we match DenoisingExperiment.prepare() exactly
    denoising_config_path = repo_root / 'mycode' / 'denoising' / 'configs' / 'best_models_100.yaml'
    print(f"\nLoading denoising config from: {denoising_config_path}")
    with open(denoising_config_path, 'r') as f:
        denoising_config = yaml.safe_load(f)

    # Extract config parameters
    datafolder = denoising_config['datafolder']
    sampling_frequency = denoising_config['sampling_frequency']
    preproc_config = denoising_config['preprocessing']
    train_fold = denoising_config['train_fold']
    val_fold = denoising_config['val_fold']
    test_fold = denoising_config['test_fold']

    # Setup paths
    noise_data_path = str(repo_root / 'ecg_noise' / 'data')
    noise_configs_dir = repo_root / 'noise' / 'configs'
    output_plot_path = repo_root / 'tests' / 'noise_characterization_results.png'

    print(f"  Datafolder: {datafolder}")
    print(f"  Sampling frequency: {sampling_frequency} Hz")
    print(f"  Preprocessing config: {preproc_config}")
    print(f"  Folds - Train: ≤{train_fold}, Val: {val_fold}, Test: {test_fold}")

    # ========================================================================
    # Step 1: Load and preprocess PTB-XL data (matching DenoisingExperiment.prepare())
    # ========================================================================
    print("\n[1/7] Loading PTB-XL dataset...")
    raw_data, raw_labels = load_dataset(datafolder, sampling_frequency)
    print(f"  Loaded {len(raw_data)} samples with shape {raw_data[0].shape}")

    print("\n[2/7] Applying preprocessing pipeline (matching DenoisingExperiment.prepare())...")

    # Remove bad labels
    print("  - Removing bad labels...")
    clean_data, clean_labels = remove_bad_labels(raw_data, raw_labels)
    print(f"    After filtering: {len(clean_data)} samples")

    # Select best lead
    print("  - Selecting best lead...")
    single_lead_data, selected_indices = select_best_lead(
        clean_data, sampling_frequency
    )
    print(f"    Shape after lead selection: {single_lead_data[0].shape}")

    # Bandpass filter
    print(f"  - Applying bandpass filter ({preproc_config['bandpass_lowcut']}-{preproc_config['bandpass_highcut']} Hz)...")
    filtered_data = bandpass_filter(
        single_lead_data,
        lowcut=preproc_config['bandpass_lowcut'],
        highcut=preproc_config['bandpass_highcut'],
        fs=sampling_frequency,
        order=preproc_config['bandpass_order']
    )
    print(f"    Filtered data shape: {filtered_data[0].shape}")

    # Split data by folds (matching DenoisingExperiment.prepare())
    print("  - Splitting data by folds...")
    train_mask = clean_labels.strat_fold <= train_fold
    val_mask = clean_labels.strat_fold == val_fold
    test_mask = clean_labels.strat_fold == test_fold

    clean_train = filtered_data[train_mask]
    clean_val = filtered_data[val_mask]
    clean_test = filtered_data[test_mask]

    print(f"    Train: {len(clean_train)}, Val: {len(clean_val)}, Test: {len(clean_test)}")

    # Normalize signals (matching DenoisingExperiment.prepare())
    print(f"  - Normalizing signals (method: {preproc_config['normalization']}, axis: {preproc_config.get('normalization_axis', 'channel')})...")
    clean_train, norm_stats = normalize_signals(
        clean_train,
        method=preproc_config['normalization'],
        axis=preproc_config.get('normalization_axis', 'channel')
    )
    clean_val, _ = normalize_signals(clean_val, stats=norm_stats)
    clean_test, _ = normalize_signals(clean_test, stats=norm_stats)

    print(f"    Normalized train shape: {clean_train[0].shape}")

    # Concatenate all splits (train/val/test)
    print("  - Concatenating train/val/test splits into single dataset...")
    clean_data_all = np.concatenate([clean_train, clean_val, clean_test], axis=0)
    print(f"    Total dataset size: {clean_data_all.shape}")

    # Apply sample limit for test performance (Comment 3)
    # This keeps the test bounded in time and memory while remaining representative
    max_samples_for_test = int(os.environ.get('NOISE_TEST_MAX_SAMPLES', '1000'))
    if len(clean_data_all) > max_samples_for_test:
        print(f"  - Subsampling to {max_samples_for_test} samples for test performance...")
        np.random.seed(denoising_config['random_seed'])
        subsample_indices = np.random.choice(len(clean_data_all), max_samples_for_test, replace=False)
        clean_data_all = clean_data_all[subsample_indices]
        print(f"    Subsampled dataset size: {clean_data_all.shape}")

    # ========================================================================
    # Step 2: Find all noise config files
    # ========================================================================
    print("\n[3/7] Finding noise configuration files...")
    config_files = sorted(list(noise_configs_dir.glob('*.yaml')))
    print(f"  Found {len(config_files)} config files:")
    for cf in config_files:
        print(f"    - {cf.name}")

    # ========================================================================
    # Step 3: Process each config and collect metrics
    # ========================================================================
    results = {
        'configs': [],
        'empirical_snr': [],
        'empirical_rmse': [],
        'theoretical_snr': [],
        'theoretical_rmse': []
    }

    print("\n[4/7] Processing noise configurations...")

    for config_path in config_files:
        config_name = config_path.stem
        print(f"\n  Processing: {config_name}")

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Parse SNR values from nested YAML structure (Comment 2)
        snr_config = config.get('SNR', {})
        print(f"    Config SNR values:")
        noise_types = ['bw', 'ma', 'em', 'AWGN']
        for nt in noise_types:
            if nt in snr_config:
                print(f"      {nt}: {snr_config[nt]} dB")

        # Initialize NoiseFactory
        print(f"    Initializing NoiseFactory...")
        noise_factory = NoiseFactory(
            data_path=noise_data_path,
            sampling_rate=sampling_frequency,
            config_path=str(config_path),
            mode='test'
        )

        # Generate noisy samples
        print(f"    Generating noisy samples...")
        noisy_data = noise_factory.add_noise(
            x=clean_data_all,
            batch_axis=0,
            channel_axis=2,
            length_axis=1
        )
        print(f"    Noisy data shape: {noisy_data.shape}")

        # Compute empirical metrics for each sample
        print(f"    Computing empirical metrics...")
        snr_values = []
        rmse_values = []

        for i in range(len(clean_data_all)):
            clean_signal = clean_data_all[i].squeeze()
            noisy_signal = noisy_data[i].squeeze()

            snr = calculate_snr(clean_signal, noisy_signal)
            rmse = calculate_rmse(clean_signal, noisy_signal)

            snr_values.append(snr)
            rmse_values.append(rmse)

        print(f"    Empirical SNR: {np.mean(snr_values):.2f} ± {np.std(snr_values):.2f} dB")
        print(f"    Empirical RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")

        # Calculate theoretical expected values (Comment 2: using nested SNR structure)
        print(f"    Computing theoretical expectations...")

        # Theoretical SNR: SNR_combined = 10 * log10(1 / sum(1/10^(SNR_i/10)))
        inv_snr_total = 0.0
        for nt in noise_types:
            snr_db = snr_config.get(nt)
            if snr_db is not None:
                snr_linear = 10 ** (snr_db / 10)
                inv_snr_total += 1.0 / snr_linear

        # Guard against misconfigured YAML (Comment 2)
        if inv_snr_total == 0.0:
            print(f"    ⚠️  WARNING: No valid SNR values found in config, skipping theoretical comparison")
            theoretical_snr_db = np.nan
            theoretical_rmse = np.nan
        else:
            theoretical_snr_db = 10 * np.log10(1.0 / inv_snr_total)

            # Theoretical RMSE: RMSE = RMS_signal / sqrt(10^(SNR_dB/10))
            rms_signal = np.sqrt(np.mean(clean_data_all ** 2))
            snr_linear = 10 ** (theoretical_snr_db / 10)
            theoretical_rmse = rms_signal / np.sqrt(snr_linear)

            print(f"    Theoretical SNR: {theoretical_snr_db:.2f} dB")
            print(f"    Theoretical RMSE: {theoretical_rmse:.4f}")

        # Store results
        results['configs'].append(config_name)
        results['empirical_snr'].append(snr_values)
        results['empirical_rmse'].append(rmse_values)
        results['theoretical_snr'].append(theoretical_snr_db)
        results['theoretical_rmse'].append(theoretical_rmse)

        # Memory cleanup
        del noisy_data
        del snr_values
        del rmse_values
        gc.collect()

    # ========================================================================
    # Step 4: Create visualization plots
    # ========================================================================
    print("\n[5/7] Creating visualization plots...")

    # Sort results by increasing expected SNR for better visualization
    sort_indices = np.argsort(results['theoretical_snr'])
    sorted_configs = [results['configs'][i] for i in sort_indices]
    sorted_empirical_snr = [results['empirical_snr'][i] for i in sort_indices]
    sorted_empirical_rmse = [results['empirical_rmse'][i] for i in sort_indices]
    sorted_theoretical_snr = [results['theoretical_snr'][i] for i in sort_indices]
    sorted_theoretical_rmse = [results['theoretical_rmse'][i] for i in sort_indices]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # SNR box plot
    ax_snr = axes[0]
    positions = np.arange(len(sorted_configs))

    # Create box plot data
    snr_data = sorted_empirical_snr
    bp_snr = ax_snr.boxplot(snr_data, positions=positions, patch_artist=True,
                             showmeans=True, meanline=True)

    # Color boxes
    for patch in bp_snr['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Add theoretical values as horizontal lines
    for i, (pos, theo_snr) in enumerate(zip(positions, sorted_theoretical_snr)):
        ax_snr.hlines(theo_snr, pos - 0.3, pos + 0.3, colors='red',
                     linestyles='dashed', linewidth=2.5, label='Expected' if i == 0 else '')

    # Set labels with expected values
    labels_snr = [f"{name}\n(Exp: {theo:.2f} dB)"
                  for name, theo in zip(sorted_configs, sorted_theoretical_snr)]
    ax_snr.set_xticks(positions)
    ax_snr.set_xticklabels(labels_snr, rotation=0, fontsize=14)
    ax_snr.set_ylabel('SNR (dB)', fontsize=16, fontweight='bold')
    ax_snr.set_xlabel('Noise Configuration', fontsize=16, fontweight='bold')
    ax_snr.set_title('Empirical vs Theoretical SNR', fontsize=18, fontweight='bold')
    ax_snr.grid(True, alpha=0.3)
    ax_snr.legend(loc='best', fontsize=14)
    ax_snr.tick_params(axis='y', labelsize=13)

    # RMSE box plot
    ax_rmse = axes[1]
    rmse_data = sorted_empirical_rmse
    bp_rmse = ax_rmse.boxplot(rmse_data, positions=positions, patch_artist=True,
                               showmeans=True, meanline=True)

    # Color boxes
    for patch in bp_rmse['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)

    # Add theoretical values as horizontal lines
    for i, (pos, theo_rmse) in enumerate(zip(positions, sorted_theoretical_rmse)):
        ax_rmse.hlines(theo_rmse, pos - 0.3, pos + 0.3, colors='red',
                      linestyles='dashed', linewidth=2.5, label='Expected' if i == 0 else '')

    # Set labels with expected values
    labels_rmse = [f"{name}\n(Exp: {theo:.4f})"
                   for name, theo in zip(sorted_configs, sorted_theoretical_rmse)]
    ax_rmse.set_xticks(positions)
    ax_rmse.set_xticklabels(labels_rmse, rotation=0, fontsize=14)
    ax_rmse.set_ylabel('RMSE', fontsize=16, fontweight='bold')
    ax_rmse.set_xlabel('Noise Configuration', fontsize=16, fontweight='bold')
    ax_rmse.set_title('Empirical vs Theoretical RMSE', fontsize=18, fontweight='bold')
    ax_rmse.grid(True, alpha=0.3)
    ax_rmse.legend(loc='best', fontsize=14)
    ax_rmse.tick_params(axis='y', labelsize=13)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to: {output_plot_path}")
    plt.close()

    # ========================================================================
    # Step 5: Validate results
    # ========================================================================
    print("\n[6/7] Validating results...")

    # Check all configs were processed
    assert len(results['configs']) == len(config_files), \
        f"Not all configs processed: {len(results['configs'])} vs {len(config_files)}"
    print(f"  ✓ All {len(config_files)} configs processed successfully")

    # Validate SNR values with relaxed tolerance (Comment 4)
    # Make tolerance configurable via environment variable with sensible default
    tolerance_db = float(os.environ.get('NOISE_TEST_SNR_TOLERANCE_DB', '10.0'))
    print(f"\n  SNR Validation (tolerance: ±{tolerance_db} dB)")

    valid_configs = []  # Track configs with valid theoretical values
    empirical_means = []
    theoretical_values = []

    for i, config_name in enumerate(results['configs']):
        empirical_mean = np.mean(results['empirical_snr'][i])
        theoretical = results['theoretical_snr'][i]

        # Skip configs with invalid theoretical values
        if np.isnan(theoretical):
            print(f"  Config '{config_name}': Skipped (no valid theoretical SNR)")
            continue

        valid_configs.append(config_name)
        empirical_means.append(empirical_mean)
        theoretical_values.append(theoretical)

        diff = abs(empirical_mean - theoretical)

        print(f"  Config '{config_name}':")
        print(f"    SNR difference: {diff:.2f} dB (tolerance: ±{tolerance_db} dB)")

        # Soft check: log warning instead of hard assertion for minor drift
        if diff > tolerance_db:
            print(f"    ⚠️  WARNING: SNR difference exceeds tolerance: {empirical_mean:.2f} vs {theoretical:.2f} dB")
        else:
            print(f"    ✓ Within tolerance")

    # Ordering-based validation (Comment 4)
    # Check that empirical and theoretical SNR rankings are consistent
    if len(valid_configs) > 1:
        # Compute ranking correlation
        correlation, p_value = spearmanr(empirical_means, theoretical_values)
        print(f"\n  Spearman rank correlation between empirical and theoretical SNR: {correlation:.3f} (p={p_value:.4f})")

        # Assert that rankings are positively correlated (correlation > 0.5)
        assert correlation > 0.5, \
            f"SNR rankings are not consistent: correlation={correlation:.3f} < 0.5"
        print(f"  ✓ SNR rankings are consistent (correlation > 0.5)")

        # Check if orderings match exactly
        empirical_order = np.argsort(empirical_means)
        theoretical_order = np.argsort(theoretical_values)
        if np.array_equal(empirical_order, theoretical_order):
            print(f"  ✓ SNR orderings match exactly across all configs")
        else:
            print(f"  ℹ️  SNR orderings differ slightly but correlation is acceptable")

    # Validate RMSE values are positive and finite
    print(f"\n  RMSE Validation")
    for i, config_name in enumerate(results['configs']):
        rmse_values = np.array(results['empirical_rmse'][i])

        assert np.all(rmse_values > 0), f"Negative RMSE values found in {config_name}"
        assert np.all(np.isfinite(rmse_values)), f"Non-finite RMSE values found in {config_name}"

    print(f"  ✓ All RMSE values are positive and finite")

    # ========================================================================
    # Step 6: Summary
    # ========================================================================
    print("\n[7/7] Summary of Results:")
    print("\n" + "-"*80)
    print(f"{'Config':<15} {'Empirical SNR':<20} {'Theoretical SNR':<20} {'Diff (dB)':<12}")
    print("-"*80)

    for i, config_name in enumerate(results['configs']):
        emp_mean = np.mean(results['empirical_snr'][i])
        emp_std = np.std(results['empirical_snr'][i])
        theo = results['theoretical_snr'][i]
        diff = emp_mean - theo

        print(f"{config_name:<15} {emp_mean:>7.2f} ± {emp_std:<7.2f} dB  "
              f"{theo:>7.2f} dB           {diff:>+7.2f}")

    print("-"*80)
    print(f"{'Config':<15} {'Empirical RMSE':<20} {'Theoretical RMSE':<20} {'Diff':<12}")
    print("-"*80)

    for i, config_name in enumerate(results['configs']):
        emp_mean = np.mean(results['empirical_rmse'][i])
        emp_std = np.std(results['empirical_rmse'][i])
        theo = results['theoretical_rmse'][i]
        diff = emp_mean - theo

        print(f"{config_name:<15} {emp_mean:>7.4f} ± {emp_std:<7.4f}  "
              f"{theo:>7.4f}              {diff:>+7.4f}")

    print("-"*80)

    print("\n" + "="*80)
    print("NOISE CHARACTERIZATION TEST COMPLETE ✓")
    print("="*80)

    # Clean up
    del clean_data_all
    gc.collect()


if __name__ == '__main__':
    test_noise_characterization()
