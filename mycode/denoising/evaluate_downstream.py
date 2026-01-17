"""
Evaluate denoising models on downstream ECG classification task.

This script:
1. Loads pre-trained classification models (xresnet1d101, inception1d)
2. Applies noise to 12-lead ECG validation data
3. Denoises each lead using trained denoising models
4. Classifies the denoised signals
5. Computes AUC metrics with confidence intervals
"""
import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '../classification'))
sys.path.insert(0, os.path.join(script_dir, '../../ecg_noise/source'))


from denoising_utils.preprocessing import normalize_signals, bandpass_filter, normalize_robust, denormalize_robust
from ecg_noise_factory.noise import NoiseFactory
from utils.utils import load_dataset, apply_standardizer

sys.path.insert(0, os.path.join(script_dir, '../../'))
from new_code.utils.getters import get_model
from new_code.visualisation.maps import COLOR_MAP, OUR_MODELS, NAME_MAP, EXCLUDE_MODELS, CLASSIFICATION_MODEL_NAMES, CLASSIFICATION_MODEL_NAMES, plot_font_sizes


def load_config(config_path='code/denoising/configs/denoising_config.yaml'):
    """Load denoising configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resample_signal(signal_data, original_rate, target_rate):
    """
    Resample ECG signal from original_rate to target_rate.

    Args:
        signal_data: Array of shape (n_samples, n_timesteps, n_channels)
        original_rate: Original sampling rate (Hz)
        target_rate: Target sampling rate (Hz)

    Returns:
        Resampled signal array
    """
    if original_rate == target_rate:
        return signal_data

    print(f"⚠️  Resampling from {original_rate}Hz to {target_rate}Hz...")

    resampled = []
    n_samples = signal_data.shape[0]

    for i in range(n_samples):
        # Resample each channel independently
        channels = []
        for ch in range(signal_data.shape[2]):
            sig = signal_data[i, :, ch]
            # Calculate number of samples for target rate
            num_samples = int(len(sig) * target_rate / original_rate)
            resampled_sig = scipy_signal.resample(sig, num_samples) # TODO check if using Fraction and resample_poly is better here.
            channels.append(resampled_sig)

        resampled.append(np.stack(channels, axis=1))

    return np.array(resampled, dtype=np.float32)


def load_classification_model(model_name, base_exp_folder, n_classes, input_shape, sampling_rate):
    """
    Load a pre-trained classification model.

    Args:
        model_name: Name of the model (e.g., 'fastai_xresnet1d101')
        base_exp_folder: Path to base experiment folder with trained models
        n_classes: Number of classification classes
        input_shape: Shape of input data
        sampling_rate: Sampling rate of the data

    Returns:
        Loaded model object
    """
    # Temporarily prioritize classification path to avoid namespace conflicts with denoising models
    import sys
    import importlib

    # Save original sys.path
    original_sys_path = sys.path.copy()

    try:
        # Move classification path to the front
        classification_path = os.path.join(os.path.dirname(__file__), '../classification')
        classification_path = os.path.abspath(classification_path)

        # Remove all denoising model paths temporarily
        sys.path = [p for p in sys.path if 'denoising_models' not in p]

        # Add classification path at the beginning
        sys.path.insert(0, classification_path)

        # Clear any cached imports of 'models' module
        if 'models' in sys.modules:
            del sys.modules['models']
        if 'models.fastai_model' in sys.modules:
            del sys.modules['models.fastai_model']

        # Now import
        from classification_models.fastai_model import fastai_model

        model_path = os.path.join(base_exp_folder, 'models', model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classification model not found: {model_path}")

        print(f"Loading classification model: {model_name}")

        model = fastai_model(
            model_name,
            n_classes,
            sampling_rate,
            model_path,
            input_shape
        )

        return model
    finally:
        # Restore original sys.path
        sys.path = original_sys_path


def denoise_12lead_signal(noisy_12lead, denoising_model, device,
                          classification_sf, denoising_sf,
                          batch_size=32, stage1_model=None):
    """
    Denoise a 12-lead ECG signal by processing each lead independently.

    Args:
        noisy_12lead: Array of shape (n_samples, n_timesteps, 12)
        denoising_model: Trained denoising model (Stage1 or Stage2)
        device: torch device
        batch_size: Batch size for processing
        stage1_model: Stage1 model (required for Stage2 models)

    Returns:
        Denoised signal of same shape
    """
    n_samples, n_timesteps, n_leads = noisy_12lead.shape

    denoising_model.eval()
    is_stage2 = stage1_model is not None

    # Detect if main denoising model is MECGE
    is_mecge = hasattr(denoising_model, 'denoising') # refactore MECGE implementation does not have denoising method.

    if is_stage2:
        stage1_model.eval()
        # Detect if Stage1 model is MECGE
        is_stage1_mecge = hasattr(stage1_model, 'denoising')

        # Guard: Stage2 MECGE is an unusual configuration
        if is_mecge:
            print(f"  WARNING: Stage2 MECGE detected. MECGE expects single-channel input.")
            print(f"  Will use MECGE on original noisy signal, bypassing Stage1 concatenation.")
    else:
        is_stage1_mecge = False

    # Informative logging
    if is_mecge and not is_stage2:
        print(f"  Using MECGE denoising method for {n_leads}-lead processing")
    elif is_mecge and is_stage2:
        print(f"  Using MECGE denoising method (Stage2 configuration, special handling)")
    else:
        print(f"  Using standard forward pass for {n_leads}-lead processing")

    # Resample from classification_sf to denoising_sf
    noisy_12lead = resample_signal(noisy_12lead, classification_sf, denoising_sf)
    denoised = np.zeros_like(noisy_12lead)

    # Process each lead
    for lead_idx in range(n_leads):
        lead_data = noisy_12lead[:, :, lead_idx:lead_idx+1]  # (n_samples, n_timesteps, 1)

        # Process in batches
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch = lead_data[batch_start:batch_end]

            # Convert to torch tensor: (batch, 1, 1, time)
            batch_tensor = torch.FloatTensor(batch).permute(0, 2, 1).unsqueeze(1).to(device)

            with torch.no_grad():
                if is_stage2:
                    # Special case: Stage2 MECGE expects single-channel, not concatenated input
                    # Branch early to skip unnecessary Stage1 computation
                    if is_mecge:
                        # Use original noisy signal for MECGE, bypassing Stage1 entirely
                        denoised_batch = denoising_model.denoising(batch_tensor)  # (batch, 1, 1, time) # TODO: Update MECGE with feeding
                    else:
                        # Standard Stage2 model: need to concatenate noisy signal with Stage1 output
                        # 1. Get Stage1 output using appropriate inference method
                        if is_stage1_mecge:
                            stage1_output = stage1_model.denoising(batch_tensor)  # (batch, 1, 1, time) # TODO: Update MECGE with feeding
                        else:
                            stage1_output = stage1_model(batch_tensor)  # (batch, 1, 1, time)

                        # 2. Concatenate along channel dimension: (batch, 2, 1, time)
                        stage2_input = torch.cat([batch_tensor, stage1_output], dim=1)

                        # 3. Pass through Stage2
                        denoised_batch = denoising_model(stage2_input)  # (batch, 1, 1, time)
                else:
                    # For Stage1: direct pass using appropriate inference method
                    if is_mecge:
                        denoised_batch = denoising_model.denoising(batch_tensor)  # (batch, 1, 1, time) # TODO: Update MECGE with feeding
                    else:
                        denoised_batch = denoising_model(batch_tensor)  # (batch, 1, 1, time)

                denoised_batch = denoised_batch.squeeze(1).permute(0, 2, 1).cpu().numpy()

            denoised[batch_start:batch_end, :, lead_idx] = denoised_batch[:, :, 0]

    denoised = resample_signal(denoised, denoising_sf, classification_sf)

    return denoised


def compute_bootstrap_ci(y_true, y_pred, n_bootstraps=100, confidence_level=0.95, metric='auc'):
    """
    Compute bootstrap confidence intervals for AUC or BCE.

    Args:
        y_true: True labels (binary multi-label format)
        y_pred: Model predictions. Must be:
                - Probabilities (0-1) for metric='auc'
                - Unnormalized logits for metric='bce' (BCEWithLogitsLoss expects raw model outputs)
        n_bootstraps: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        metric: 'auc' or 'bce' to specify which metric to compute

    Returns:
        Dictionary with mean, lower, and upper bounds
    """
    scores = []
    n_samples = len(y_true)

    np.random.seed(42)

    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # For AUC: check if we have at least one positive sample per class
        # For BCE: no filtering needed, all samples are valid
        if metric == 'auc' and y_true_boot.sum(axis=0).min() == 0:
            continue

        try:
            if metric == 'auc':
                score = roc_auc_score(y_true_boot, y_pred_boot, average='macro')
            elif metric == 'bce':
                # BCE expects unnormalized logits (raw model outputs before sigmoid)
                bce_loss = nn.BCEWithLogitsLoss()
                y_true_tensor = torch.FloatTensor(y_true_boot)
                y_pred_tensor = torch.FloatTensor(y_pred_boot)
                score = bce_loss(y_pred_tensor, y_true_tensor).item()
            else:
                raise ValueError(f"Unknown metric: {metric}")
            scores.append(score)
        except:
            continue

    if len(scores) == 0:
        return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}

    scores = np.array(scores)
    alpha = 1 - confidence_level

    return {
        'mean': np.mean(scores),
        'lower': np.percentile(scores, 100 * alpha / 2),
        'upper': np.percentile(scores, 100 * (1 - alpha / 2))
    }


def evaluate_downstream(config_path='code/denoising/configs/denoising_config.yaml', base_exp='exp0',
                       classification_sampling_rate=500, classifier_names=None):
    """
    Main evaluation function.

    Args:
        config_path: Path to denoising config file
        base_exp: Name of base classification experiment (e.g., 'exp0')
        classification_sampling_rate: Sampling rate used for classification models
        classifier_names: List of classification model names to evaluate
    """
    # Default classification models if none specified
    if classifier_names is None:
        classifier_names = ['fastai_xresnet1d101', 'fastai_inception1d']
    print("\n" + "="*80)
    print("DOWNSTREAM CLASSIFICATION EVALUATION")
    print("="*80)

    # Load config
    config = load_config(config_path)
    denoising_exp_folder = os.path.join(config['outputfolder'], config['experiment_name'])
    denoising_sampling_rate = config['sampling_frequency']

    # Create results folder
    results_folder = os.path.join(denoising_exp_folder, 'downstream_results')
    os.makedirs(results_folder, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and
                         config['hardware']['use_cuda'] else 'cpu')
    print(f"Using device: {device}")

    # Check for sampling rate mismatch
    if denoising_sampling_rate != classification_sampling_rate:
        print(f"\n⚠️  WARNING: Sampling rate mismatch!")
        print(f"   Denoising models trained at: {denoising_sampling_rate}Hz")
        print(f"   Classification models trained at: {classification_sampling_rate}Hz")
        print(f"   Signals will be resampled from {denoising_sampling_rate}Hz to {classification_sampling_rate}Hz")
    else:
        print(f"\n✓ Sampling rates match: {classification_sampling_rate}Hz")

    # ========================================================================
    # Load 12-lead validation data
    # ========================================================================
    print("\n" + "-"*80)
    print("Loading 12-lead validation data...")
    print("-"*80)

    datafolder = config['datafolder']
    val_fold = config['val_fold']
    test_fold = config['test_fold']

    # Load PTB-XL data at classification sampling rate
    data, raw_labels = load_dataset(datafolder, classification_sampling_rate)
    print(f"Loaded: {data.shape[0]} samples at {classification_sampling_rate}Hz")

    # Extract validation fold
    X_val_12lead = data[raw_labels.strat_fold == val_fold]
    X_val_12lead_original = X_val_12lead.copy()
    X_train_12lead = data[~raw_labels.strat_fold.isin([val_fold, test_fold])]
    print(f"Validation samples: {len(X_val_12lead)}")
    print(f"Shape: {X_val_12lead[0].shape} (timesteps, channels)")

    # Load classification labels and scaler from base experiment
    base_exp_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        config['classification_outputfolder'], base_exp
        )
    if not os.path.exists(base_exp_path):
        # Try without the '..' if outputfolder already includes the right path
        base_exp_path = os.path.join('../../output', base_exp)

    if not os.path.exists(base_exp_path):
        raise FileNotFoundError(f"Base experiment not found: {base_exp_path}")

    print(f"Base experiment folder: {base_exp_path}")

    y_val = np.load(os.path.join(base_exp_path, 'data', 'y_val.npy'), allow_pickle=True)
    n_classes = y_val.shape[1]
    print(f"Number of classes: {n_classes}")

    # Load and apply classification standardizer # TODO bring this back later
    # with open(os.path.join(base_exp_path, 'data', 'standard_scaler.pkl'), 'rb') as f:
    #     scaler = pickle.load(f)
    #  # TODO: Apply denoising standardizer here + denoising pre-processing

    # compute robust statistics from training data for normalization
    median = np.median(X_train_12lead)
    iqr = np.percentile(X_train_12lead, 75) - np.percentile(X_train_12lead, 25)
    X_val_12lead = normalize_robust(X_val_12lead, median, iqr)
    if config.get('bandpass', True):
        X_val_12lead = bandpass_filter(X_val_12lead, fs=classification_sampling_rate) # TODO: Make sure this is not applied twice

    print("✓ Applied denoising standardizer (Robust Normalization)")

    # Store clean version
    X_val_clean = X_val_12lead.copy()

    # ========================================================================
    # Add noise to each lead
    # ========================================================================
    print("\n" + "-"*80)
    print("Adding noise to 12-lead data...")
    print("-"*80)

    noise_data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            config['noise_data_path']
            )

    noise_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            config['noise_config_path']
            )

    print(f"Using noise config: {noise_config_path}")

    # Create noise factory at classification sampling rate
    noise_factory = NoiseFactory(
        data_path=noise_data_path,
        sampling_rate=classification_sampling_rate,
        config_path=noise_config_path,
        mode='eval'  # Use eval mode for final evaluation
    )

    X_val_noisy = noise_factory.add_noise(
        x=X_val_12lead, batch_axis=0, channel_axis=2, length_axis=1
    )

    print("✓ Noise added to all 12 leads")

    # ========================================================================
    # Load denoising models
    # ========================================================================
    print("\n" + "-"*80)
    print("Loading denoising models...")
    print("-"*80)

    denoising_models = {}
    stage1_models_cache = {}  # Cache loaded Stage1 models for Stage2 use
    model_configs = config['models']

    for model_config in model_configs:
        model_name = model_config['name']
        model_type = model_config['type']

        # Check if model exists
        model_path = model_config['model_path']

        if not os.path.exists(model_path):
            print(f"⚠️  Model {model_name} not found, skipping...")
            continue

        # Load model
        is_stage2 = model_config['is_stage_2']
        input_length = denoising_sampling_rate * 10

        model = get_model(
            model_type,
            sequence_length=input_length,  # Use denoising input length
            model_config = model_config,
            is_stage2=is_stage2
        )

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        # For Stage2 models, also load the corresponding Stage1 model
        stage1_model = None
        if is_stage2: # TODO update this section
            stage1_name = model_config.get('stage_1_type', None)
            if stage1_name:
                # Check if we already loaded this Stage1 model
                if stage1_name in stage1_models_cache:
                    stage1_model = stage1_models_cache[stage1_name]
                    print(f"  Using cached Stage1 model: {stage1_name}")
                else:
                    # Load the Stage1 model
                    stage1_model_path = model_config['stage_1_weights_path']
                    if os.path.exists(stage1_model_path):
                        # Determine Stage1 model type from its name or config
                        # IMPORTANT: Check for 'imunet' BEFORE 'unet' since 'imunet' contains 'unet'
                        stage1_type = model_config['stage_1_type']
                        input_length = denoising_sampling_rate * 10

                        stage1_model = get_model( # TODO: Update get_model
                            stage1_type,
                            sequence_length=input_length,
                            model_config=model_config, # access mamba_params for stage 1 model
                            is_stage2=False
                        )
                        stage1_model.load_state_dict(torch.load(stage1_model_path, map_location=device, weights_only=True))
                        stage1_model.to(device)
                        stage1_model.eval()

                        # Cache it
                        stage1_models_cache[stage1_name] = stage1_model
                        print(f"  Loaded Stage1 model: {stage1_name} (type: {stage1_type})")
                    else:
                        print(f"  ⚠️  Warning: Stage1 model {stage1_name} not found for {model_name}")
            else:
                print(f"  ⚠️  Warning: No stage1_model specified for {model_name}")

        denoising_models[model_name] = {
            'model': model,
            'type': model_type,
            'is_stage2': is_stage2,
            'stage1_model': stage1_model
        }

        print(f"✓ Loaded: {model_name}")

    print(f"\nTotal denoising models loaded: {len(denoising_models)}")

    # ========================================================================
    # Load classification models
    # ========================================================================
    print("\n" + "-"*80)
    print("Loading classification models...")
    print("-"*80)

    classification_models = {}
    # classifier_names will be passed as parameter

    for clf_name in classifier_names:
        try:
            clf_model = load_classification_model(
                clf_name,
                base_exp_path,
                n_classes,
                X_val_clean[0].shape,
                classification_sampling_rate
            )
            classification_models[clf_name] = clf_model
            print(f"✓ Loaded: {clf_name}")
        except Exception as e:
            print(f"⚠️  Failed to load {clf_name}: {e}")

    if len(classification_models) == 0:
        print("ERROR: No classification models loaded!")
        return

    # ========================================================================
    # Evaluate all combinations
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATING ALL COMBINATIONS")
    print("="*80)

    results = []

    # Baseline: Clean data
    print("\n--- Baseline: Clean Data ---")
    for clf_name, clf_model in classification_models.items():
        print(f"\nClassifying with {clf_name}...")
        # revert normalization and add classification pre-processing
        with open(os.path.join(base_exp_path, 'data', 'standard_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        X_val_clean_original_p = apply_standardizer(X_val_12lead_original, scaler)
        y_pred_clean = clf_model.predict(X_val_clean_original_p)

        auc_point = roc_auc_score(y_val, y_pred_clean, average='macro')
        ci = compute_bootstrap_ci(y_val, y_pred_clean, n_bootstraps=100)

        # Compute BCE (requires unnormalized logits from classifier)
        # Note: y_pred_clean contains raw model outputs (logits), not probabilities
        bce_loss = nn.BCEWithLogitsLoss()
        bce_point = bce_loss(torch.FloatTensor(y_pred_clean), torch.FloatTensor(y_val)).item()
        bce_ci = compute_bootstrap_ci(y_val, y_pred_clean, n_bootstraps=100, metric='bce')

        results.append({
            'denoising_model': 'clean',
            'classification_model': clf_name,
            'auc': auc_point,
            'auc_mean': ci['mean'],
            'auc_lower': ci['lower'],
            'auc_upper': ci['upper'],
            'bce': bce_point,
            'bce_mean': bce_ci['mean'],
            'bce_lower': bce_ci['lower'],
            'bce_upper': bce_ci['upper']
        })

        print(f"  AUC: {auc_point:.4f} (95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}])")
        print(f"  BCE: {bce_point:.4f} (95% CI: [{bce_ci['lower']:.4f}, {bce_ci['upper']:.4f}])")

    # Baseline: Noisy data (no denoising)
    print("\n--- Baseline: Noisy Data (no denoising) ---")
    for clf_name, clf_model in classification_models.items():
        print(f"\nClassifying with {clf_name}...")
        # revert normalization and add classification pre-processing
        X_val_noisy_denorm = X_val_noisy.copy()
        X_val_noisy_denorm = denormalize_robust(X_val_noisy_denorm, median, iqr)
        with open(os.path.join(base_exp_path, 'data', 'standard_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        X_val_noisy_denorm = apply_standardizer(X_val_noisy_denorm, scaler)
        y_pred_noisy = clf_model.predict(X_val_noisy_denorm)

        auc_point = roc_auc_score(y_val, y_pred_noisy, average='macro')
        ci = compute_bootstrap_ci(y_val, y_pred_noisy, n_bootstraps=100)

        # Compute BCE (requires unnormalized logits from classifier)
        # Note: y_pred_noisy contains raw model outputs (logits), not probabilities
        bce_loss = nn.BCEWithLogitsLoss()
        bce_point = bce_loss(torch.FloatTensor(y_pred_noisy), torch.FloatTensor(y_val)).item()
        bce_ci = compute_bootstrap_ci(y_val, y_pred_noisy, n_bootstraps=100, metric='bce')

        results.append({
            'denoising_model': 'noisy',
            'classification_model': clf_name,
            'auc': auc_point,
            'auc_mean': ci['mean'],
            'auc_lower': ci['lower'],
            'auc_upper': ci['upper'],
            'bce': bce_point,
            'bce_mean': bce_ci['mean'],
            'bce_lower': bce_ci['lower'],
            'bce_upper': bce_ci['upper']
        })

        print(f"  AUC: {auc_point:.4f} (95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}])")
        print(f"  BCE: {bce_point:.4f} (95% CI: [{bce_ci['lower']:.4f}, {bce_ci['upper']:.4f}])")

    # Denoised data
    print("\n--- Denoised Data ---")
    for denoise_name, denoise_info in tqdm(denoising_models.items(), desc="Denoising models"):
        print(f"\n{denoise_name}:")

        # Denoise the noisy data
        print(f"  Denoising 12 leads...")

        # For Stage2 models, pass the Stage1 model as well
        stage1_model = denoise_info.get('stage1_model', None)

        X_val_denoised = denoise_12lead_signal(
            X_val_noisy,
            denoise_info['model'],
            device,
            classification_sf=classification_sampling_rate,
            denoising_sf=denoising_sampling_rate,
            batch_size=32,
            stage1_model=stage1_model
        )

        # TODO: Revert denoising standardization applied earlyer
        X_val_denoised = denormalize_robust(X_val_denoised, median, iqr)
        # TODO: Apply classification standardization
        with open(os.path.join(base_exp_path, 'data', 'standard_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        X_val_denoised = apply_standardizer(X_val_denoised, scaler)

        # Classify with each classifier
        for clf_name, clf_model in classification_models.items():
            print(f"  Classifying with {clf_name}...")
            y_pred_denoised = clf_model.predict(X_val_denoised)

            auc_point = roc_auc_score(y_val, y_pred_denoised, average='macro')
            ci = compute_bootstrap_ci(y_val, y_pred_denoised, n_bootstraps=100)

            # Compute BCE (requires unnormalized logits from classifier)
            # Note: y_pred_denoised contains raw model outputs (logits), not probabilities
            bce_loss = nn.BCEWithLogitsLoss()
            bce_point = bce_loss(torch.FloatTensor(y_pred_denoised), torch.FloatTensor(y_val)).item()
            bce_ci = compute_bootstrap_ci(y_val, y_pred_denoised, n_bootstraps=100, metric='bce')

            results.append({
                'denoising_model': denoise_name,
                'classification_model': clf_name,
                'auc': auc_point,
                'auc_mean': ci['mean'],
                'auc_lower': ci['lower'],
                'auc_upper': ci['upper'],
                'bce': bce_point,
                'bce_mean': bce_ci['mean'],
                'bce_lower': bce_ci['lower'],
                'bce_upper': bce_ci['upper']
            })

            print(f"    AUC: {auc_point:.4f} (95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}])")
            print(f"    BCE: {bce_point:.4f} (95% CI: [{bce_ci['lower']:.4f}, {bce_ci['upper']:.4f}])")

    # ========================================================================
    # Save and visualize results
    # ========================================================================
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    results_df = pd.DataFrame(results)

    # Save to CSV
    results_path = os.path.join(results_folder, 'downstream_classification_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")

    # Print summary
    print("\n" + "-"*80)
    print("Summary Table")
    print("-"*80)
    print(results_df.to_string(index=False))

    # Create visualizations
    plot_downstream_results(results_df, results_folder)

    print("\n✓ Downstream evaluation complete!")
    print(f"   Results saved to: {results_folder}")


def plot_downstream_results(results_df, output_folder):
    """Create visualizations of downstream classification results.

    Creates PNG files for both AUC and BCE metrics per classification model.
    """
    # Generate AUC plots
    plot_metric_bars(results_df, output_folder, metric='auc')

    # Generate BCE plots
    plot_metric_bars(results_df, output_folder, metric='bce')

    # Generate combined AUC+BCE plots
    plot_metric_bars_combined(results_df, output_folder)

    # Create improvement heatmaps
    create_improvement_heatmap(results_df, output_folder, metric='auc')
    create_improvement_heatmap(results_df, output_folder, metric='bce')


def plot_metric_bars(results_df, output_folder, metric='auc'):
    """Create bar plots for a specific metric (AUC or BCE).

    Creates one PNG file per classification model showing all denoising approaches.

    Args:
        results_df: DataFrame with evaluation results
        output_folder: Path to save plots
        metric: 'auc' or 'bce'
    """
    # Comprehensive color map for consistent styling across all plots
    color_map = COLOR_MAP

    sns.set_style("whitegrid")

    classifiers = results_df['classification_model'].unique()

    # Determine if lower is better (BCE) or higher is better (AUC)
    lower_is_better = (metric == 'bce')
    metric_label = 'BCE (Binary Cross Entropy)' if metric == 'bce' else 'AUC (macro)'

    # Create one figure per classifier
    for clf_name in classifiers:
        fig, ax = plt.subplots(figsize=(10, len(results_df['denoising_model'].unique()) * 0.5))

        # Filter data for this classifier
        clf_data = results_df[results_df['classification_model'] == clf_name].copy()

        # Exclude models that are in EXCLUDE_MODELS
        clf_data = clf_data[~clf_data['denoising_model'].isin(EXCLUDE_MODELS)]

        # Order models according to colormap for consistent visual grouping
        all_denoise_models = clf_data['denoising_model'].unique().tolist()
        colormap_order = list(color_map.keys())
        # Filter to include only models that are in the data (excluding 'clean', 'noisy', and excluded models)
        ordered_models = [m for m in colormap_order if m in all_denoise_models and m not in ['clean', 'noisy'] and m not in EXCLUDE_MODELS]
        # Find any models not in the colormap (excluding excluded models)
        unlisted_models = [m for m in all_denoise_models if m not in color_map and m not in ['clean', 'noisy'] and m not in EXCLUDE_MODELS]
        # Combine them: colormap order first, then unlisted
        sorted_models = ordered_models + unlisted_models
        # Add baseline models: noisy first, clean last
        if 'noisy' in all_denoise_models:
            sorted_models = ['noisy'] + sorted_models
        if 'clean' in all_denoise_models:
            sorted_models = sorted_models + ['clean']
        # Reverse the order for plotting (bottom to top)
        sorted_models = sorted_models[::-1]
        # Reorder DataFrame
        clf_data['model_order'] = clf_data['denoising_model'].apply(lambda x: sorted_models.index(x) if x in sorted_models else len(sorted_models))
        clf_data = clf_data.sort_values('model_order', ascending=True)
        clf_data = clf_data.drop('model_order', axis=1)

        # Prepare data for plotting
        denoise_models = clf_data['denoising_model'].values
        metric_values = clf_data[metric].values
        metric_lowers = clf_data[f'{metric}_lower'].values
        metric_uppers = clf_data[f'{metric}_upper'].values

        # Calculate error bars
        yerr_lower = metric_values - metric_lowers
        yerr_upper = metric_uppers - metric_values

        # Assign colors using color_map for consistency
        colors = []
        for model in denoise_models:
            if model == 'clean':
                colors.append('#ABABAB')  # Green for clean baseline
            elif model == 'noisy':
                colors.append('#808080')  # Red for noisy baseline
            else:
                colors.append(color_map.get(model, '#cccccc'))  # Use color_map, default to grey

        # Create display names using NAME_MAP and add (ours) for our models
        display_names = []
        for model in denoise_models:
            display_name = NAME_MAP.get(model, model)
            if model in OUR_MODELS:
                display_name = f"{display_name} (ours)"
            display_names.append(display_name)

        # Create horizontal bar plot
        y_pos = np.arange(len(denoise_models))

        # Find baseline metric values
        noisy_value = None
        clean_value = None
        for i, model in enumerate(denoise_models):
            if model == 'noisy':
                noisy_value = metric_values[i]
            elif model == 'clean':
                clean_value = metric_values[i]

        # Add hatched regions covering entire plot height (in background)
        y_min = -0.5
        y_max = len(denoise_models) - 0.5

        # For AUC (higher is better): shade below noisy and above clean
        # For BCE (lower is better): shade above noisy and below clean
        if lower_is_better:
            # BCE: shade region above noisy (worse performance)
            if noisy_value is not None:
                x_max_limit = metric_values.max() * 1.1
                ax.fill_betweenx([y_min, y_max], noisy_value, x_max_limit,
                                color='lightgrey', alpha=0.2, hatch='///',
                                edgecolor='grey', linewidth=0.5, zorder=0)
            # BCE: shade region below clean (better than clean)
            if clean_value is not None:
                ax.fill_betweenx([y_min, y_max], 0, clean_value,
                                color='lightgrey', alpha=0.2, hatch='///',
                                edgecolor='grey', linewidth=0.5, zorder=0)
        else:
            # AUC: shade region below noisy (worse performance)
            if noisy_value is not None:
                ax.fill_betweenx([y_min, y_max], 0, noisy_value,
                                color='lightgrey', alpha=0.2, hatch='///',
                                edgecolor='grey', linewidth=0.5, zorder=0)
            # AUC: shade region above clean (better than clean)
            if clean_value is not None:
                ax.fill_betweenx([y_min, y_max], clean_value, 1.0,
                                color='lightgrey', alpha=0.2, hatch='///',
                                edgecolor='grey', linewidth=0.5, zorder=0)

        bars = ax.barh(y_pos, metric_values, xerr=[yerr_lower, yerr_upper],
                      color=colors, alpha=0.8, edgecolor='black',
                      linewidth=1, capsize=4)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names, fontsize=plot_font_sizes['ticks'])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel(metric_label, fontsize=plot_font_sizes['axis_labels'], fontweight='bold')
        clf_display_name = CLASSIFICATION_MODEL_NAMES.get(clf_name, clf_name)
        # ax.set_title(f'Downstream ECG Classification Performance - {clf_display_name}',
        #             fontsize=plot_font_sizes['title'], fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels at appropriate position based on metric direction
        for i, (value, lower, upper) in enumerate(zip(metric_values, metric_lowers, metric_uppers)):
            # For BCE, put label to the left of lower bound; for AUC, to the right of upper bound
            if lower_is_better:
                ax.text(lower - 0.001, i, f'{value:.4f}',
                       ha='right', va='center', fontsize=plot_font_sizes['value_labels'], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='none', alpha=0.7))
            else:
                ax.text(upper + 0.001, i, f'{value:.4f}',
                       ha='left', va='center', fontsize=plot_font_sizes['value_labels'], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='none', alpha=0.7))

        # Set x-axis limits dynamically based on best and worst performing models
        if lower_is_better:
            x_min = max(0, metric_values.min() - 0.02)
            x_max = metric_values.max() + 0.02
        else:
            x_min = max(0.5, metric_values.min() - 0.01)
            x_max = min(1.0, metric_values.max() + 0.02)
        ax.set_xlim([x_min, x_max])

        # Add vertical dotted line at the best performing model (excluding clean)
        if lower_is_better:
            # For BCE, best is minimum
            best_value = float('inf')
            for i, model in enumerate(denoise_models):
                if model != 'clean' and metric_values[i] < best_value:
                    best_value = metric_values[i]
            if best_value != float('inf'):
                ax.axvline(x=best_value, color='darkgrey', linestyle=':', linewidth=2,
                          alpha=0.7, zorder=1, label=f'Best: {best_value:.4f}')
        else:
            # For AUC, best is maximum
            best_value = 0
            for i, model in enumerate(denoise_models):
                if model != 'clean' and metric_values[i] > best_value:
                    best_value = metric_values[i]
            if best_value > 0:
                ax.axvline(x=best_value, color='darkgrey', linestyle=':', linewidth=2,
                          alpha=0.7, zorder=1, label=f'Best: {best_value:.4f}')

        plt.tight_layout()

        # Save plot with classifier name and metric in filename
        safe_clf_name = clf_name.replace('/', '_').replace('\\', '_')
        plot_path = os.path.join(output_folder, f'downstream_{metric}_{safe_clf_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ {metric.upper()} visualization saved to: {plot_path}")


def plot_metric_bars_combined(results_df, output_folder):
    """Create combined bar plots showing both AUC and BCE metrics side by side.

    Creates one PNG file per classification model with two subplots.

    Args:
        results_df: DataFrame with evaluation results
        output_folder: Path to save plots
    """
    color_map = COLOR_MAP
    sns.set_style("whitegrid")

    classifiers = results_df['classification_model'].unique()

    for clf_name in classifiers:
        # Filter data for this classifier
        clf_data = results_df[results_df['classification_model'] == clf_name].copy()
        clf_data = clf_data[~clf_data['denoising_model'].isin(EXCLUDE_MODELS)]

        # Order models
        all_denoise_models = clf_data['denoising_model'].unique().tolist()
        colormap_order = list(color_map.keys())
        ordered_models = [m for m in colormap_order if m in all_denoise_models and m not in ['clean', 'noisy'] and m not in EXCLUDE_MODELS]
        unlisted_models = [m for m in all_denoise_models if m not in color_map and m not in ['clean', 'noisy'] and m not in EXCLUDE_MODELS]
        sorted_models = ordered_models + unlisted_models
        if 'noisy' in all_denoise_models:
            sorted_models = ['noisy'] + sorted_models
        if 'clean' in all_denoise_models:
            sorted_models = sorted_models + ['clean']
        sorted_models = sorted_models[::-1]
        clf_data['model_order'] = clf_data['denoising_model'].apply(lambda x: sorted_models.index(x) if x in sorted_models else len(sorted_models))
        clf_data = clf_data.sort_values('model_order', ascending=True)
        clf_data = clf_data.drop('model_order', axis=1)

        denoise_models = clf_data['denoising_model'].values
        n_models = len(denoise_models)

        # Create figure with two subplots side by side (squeezed horizontally)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, n_models * 0.5), sharey=True)

        # Prepare display names and colors
        colors = []
        for model in denoise_models:
            if model == 'clean':
                colors.append('#ABABAB')
            elif model == 'noisy':
                colors.append('#808080')
            else:
                colors.append(color_map.get(model, '#cccccc'))

        display_names = []
        for model in denoise_models:
            display_name = NAME_MAP.get(model, model)
            if model in OUR_MODELS:
                display_name = f"{display_name} (ours)"
            display_names.append(display_name)

        y_pos = np.arange(len(denoise_models))
        y_min = -0.5
        y_max = len(denoise_models) - 0.5

        # Plot AUC (left subplot)
        metric = 'auc'
        auc_values = clf_data[metric].values
        auc_lowers = clf_data[f'{metric}_lower'].values
        auc_uppers = clf_data[f'{metric}_upper'].values
        yerr_lower_auc = auc_values - auc_lowers
        yerr_upper_auc = auc_uppers - auc_values

        # Find baseline values for AUC
        noisy_auc = None
        clean_auc = None
        for i, model in enumerate(denoise_models):
            if model == 'noisy':
                noisy_auc = auc_values[i]
            elif model == 'clean':
                clean_auc = auc_values[i]

        # AUC hatched regions
        if noisy_auc is not None:
            ax1.fill_betweenx([y_min, y_max], 0, noisy_auc,
                            color='lightgrey', alpha=0.2, hatch='///',
                            edgecolor='grey', linewidth=0.5, zorder=0)
        if clean_auc is not None:
            ax1.fill_betweenx([y_min, y_max], clean_auc, 1.0,
                            color='lightgrey', alpha=0.2, hatch='///',
                            edgecolor='grey', linewidth=0.5, zorder=0)

        ax1.barh(y_pos, auc_values, xerr=[yerr_lower_auc, yerr_upper_auc],
                color=colors, alpha=0.8, edgecolor='black',
                linewidth=1, capsize=4)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(display_names, fontsize=plot_font_sizes['ticks'])
        ax1.set_ylim([y_min, y_max])
        ax1.set_xlabel('AUC (macro)', fontsize=plot_font_sizes['axis_labels'], fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Add value labels for AUC
        for i, (value, lower, upper) in enumerate(zip(auc_values, auc_lowers, auc_uppers)):
            ax1.text(upper + 0.001, i, f'{value:.4f}',
                   ha='left', va='center', fontsize=plot_font_sizes['value_labels'], fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='none', alpha=0.7))

        # Set x-axis limits for AUC
        x_min_auc = max(0.5, auc_values.min() - 0.01)
        x_max_auc = min(1.0, auc_values.max() + 0.02)
        ax1.set_xlim([x_min_auc, x_max_auc])

        # Best AUC line
        best_auc = 0
        for i, model in enumerate(denoise_models):
            if model != 'clean' and auc_values[i] > best_auc:
                best_auc = auc_values[i]
        if best_auc > 0:
            ax1.axvline(x=best_auc, color='darkgrey', linestyle=':', linewidth=2,
                      alpha=0.7, zorder=1)

        # Plot BCE (right subplot)
        metric = 'bce'
        bce_values = clf_data[metric].values
        bce_lowers = clf_data[f'{metric}_lower'].values
        bce_uppers = clf_data[f'{metric}_upper'].values
        yerr_lower_bce = bce_values - bce_lowers
        yerr_upper_bce = bce_uppers - bce_values

        # Find baseline values for BCE
        noisy_bce = None
        clean_bce = None
        for i, model in enumerate(denoise_models):
            if model == 'noisy':
                noisy_bce = bce_values[i]
            elif model == 'clean':
                clean_bce = bce_values[i]

        # BCE hatched regions (lower is better)
        if noisy_bce is not None:
            x_max_limit = bce_values.max() * 1.1
            ax2.fill_betweenx([y_min, y_max], noisy_bce, x_max_limit,
                            color='lightgrey', alpha=0.2, hatch='///',
                            edgecolor='grey', linewidth=0.5, zorder=0)
        if clean_bce is not None:
            ax2.fill_betweenx([y_min, y_max], 0, clean_bce,
                            color='lightgrey', alpha=0.2, hatch='///',
                            edgecolor='grey', linewidth=0.5, zorder=0)

        ax2.barh(y_pos, bce_values, xerr=[yerr_lower_bce, yerr_upper_bce],
                color=colors, alpha=0.8, edgecolor='black',
                linewidth=1, capsize=4)

        ax2.set_ylim([y_min, y_max])
        ax2.set_xlabel('BCE (Binary Cross Entropy)', fontsize=plot_font_sizes['axis_labels'], fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Add value labels for BCE (on the left of lower bound)
        for i, (value, lower, upper) in enumerate(zip(bce_values, bce_lowers, bce_uppers)):
            ax2.text(lower - 0.001, i, f'{value:.4f}',
                   ha='right', va='center', fontsize=plot_font_sizes['value_labels'], fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='none', alpha=0.7))

        # Set x-axis limits for BCE
        x_min_bce = max(0, bce_values.min() - 0.02)
        x_max_bce = bce_values.max() + 0.02
        ax2.set_xlim([x_min_bce, x_max_bce])

        # Best BCE line (minimum)
        best_bce = float('inf')
        for i, model in enumerate(denoise_models):
            if model != 'clean' and bce_values[i] < best_bce:
                best_bce = bce_values[i]
        if best_bce != float('inf'):
            ax2.axvline(x=best_bce, color='darkgrey', linestyle=':', linewidth=2,
                      alpha=0.7, zorder=1)

        plt.tight_layout()

        # Save combined plot
        safe_clf_name = clf_name.replace('/', '_').replace('\\', '_')
        plot_path = os.path.join(output_folder, f'downstream_combined_{safe_clf_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Combined AUC+BCE visualization saved to: {plot_path}")


def create_improvement_heatmap(results_df, output_folder, metric='auc'):
    """Create heatmap showing metric improvements over noisy baseline.

    Args:
        results_df: DataFrame with evaluation results
        output_folder: Path to save plots
        metric: 'auc' or 'bce'
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    classifiers = sorted(results_df['classification_model'].unique())

    # For BCE, lower is better, so improvement is negative (reduction)
    lower_is_better = (metric == 'bce')
    metric_label = 'BCE' if metric == 'bce' else 'AUC'

    # Get noisy baseline for each classifier
    noisy_values = {}
    for clf in classifiers:
        noisy_value = results_df[
            (results_df['classification_model'] == clf) &
            (results_df['denoising_model'] == 'noisy')
        ][metric].values[0]
        noisy_values[clf] = noisy_value

    # Calculate improvements
    improvements = []
    denoise_models = []

    for denoise_model in results_df['denoising_model'].unique():
        if denoise_model in ['clean', 'noisy']:
            continue

        denoise_models.append(denoise_model)
        row = []

        for clf in classifiers:
            denoised_value = results_df[
                (results_df['classification_model'] == clf) &
                (results_df['denoising_model'] == denoise_model)
            ][metric].values

            if len(denoised_value) > 0:
                if lower_is_better:
                    # For BCE, improvement is reduction (negative change is good)
                    improvement = noisy_values[clf] - denoised_value[0]
                else:
                    # For AUC, improvement is increase (positive change is good)
                    improvement = denoised_value[0] - noisy_values[clf]
                row.append(improvement)
            else:
                row.append(0)

        improvements.append(row)

    if len(improvements) > 0:
        improvements = np.array(improvements)

        # Create heatmap
        im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto',
                      vmin=-0.05, vmax=0.05)

        # Set ticks
        ax.set_xticks(np.arange(len(classifiers)))
        ax.set_yticks(np.arange(len(denoise_models)))
        ax.set_xticklabels(classifiers, fontsize=10)
        ax.set_yticklabels(denoise_models, fontsize=9)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        for i in range(len(denoise_models)):
            for j in range(len(classifiers)):
                text = ax.text(j, i, f'{improvements[i, j]:.4f}',
                             ha="center", va="center", color="black",
                             fontsize=9, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if lower_is_better:
            cbar.set_label(f'{metric_label} Reduction from Noisy Baseline', fontsize=11, fontweight='bold')
        else:
            cbar.set_label(f'{metric_label} Improvement over Noisy Baseline', fontsize=11, fontweight='bold')

        ax.set_title(f'Denoising Impact on Classification Performance ({metric_label})',
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Classification Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Denoising Model', fontsize=11, fontweight='bold')

        plt.tight_layout()

        # Save plot with metric in filename
        plot_path = os.path.join(output_folder, f'downstream_improvement_heatmap_{metric}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ {metric_label} heatmap saved to: {plot_path}")

# results_df = pd.read_csv('/local/home/bamorel/my_ecg_ptbxl_benchmarking/mycode/denoising/output/all_100_nbp/downstream_results/downstream_classification_results.csv')
# output_folder = '/local/home/bamorel/my_ecg_ptbxl_benchmarking/mycode/denoising/output/all_100_nbp/downstream_results/'
# plot_downstream_results(results_df, output_folder)


def main():
    """Main entry point."""
    # Define all available classification models
    ALL_CLASSIFIERS = [
        'fastai_xresnet1d101',
        'fastai_inception1d',
        'fastai_resnet1d_wang',
        'fastai_lstm',
        'fastai_lstm_bidir',
        'fastai_fcn_wang'
    ]

    parser = argparse.ArgumentParser(
        description='Evaluate denoising models on downstream ECG classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default models (xresnet1d101, inception1d)
  python evaluate_downstream.py

  # Evaluate all 6 models
  python evaluate_downstream.py --classifiers all

  # Evaluate specific models
  python evaluate_downstream.py --classifiers fastai_xresnet1d101 fastai_lstm

  # With custom experiment settings
  python evaluate_downstream.py --base-exp exp1 --classification-fs 500 --classifiers all
        """
    )
    parser.add_argument('--config', type=str, default='mycode/denoising/configs/new_run.yaml',
                       help='Path to denoising config file')
    parser.add_argument('--base-exp', type=str, default='exp0',
                       help='Name of base classification experiment')
    parser.add_argument('--classification-fs', type=int, default=100,
                       help='Sampling frequency used for classification models (Hz)')
    parser.add_argument('--classifiers', type=str, nargs='+',
                       default=['fastai_xresnet1d101', 'fastai_inception1d'],
                       help='Classification models to evaluate. Use "all" for all 6 models, '
                            'or specify space-separated model names. '
                            f'Available: {", ".join(ALL_CLASSIFIERS)}')
    args = parser.parse_args()

    # Handle 'all' keyword
    if len(args.classifiers) == 1 and args.classifiers[0].lower() == 'all':
        classifier_names = ALL_CLASSIFIERS
        print(f"Using all {len(ALL_CLASSIFIERS)} classification models")
    else:
        classifier_names = args.classifiers

    evaluate_downstream(
        config_path=args.config,
        base_exp=args.base_exp,
        classification_sampling_rate=args.classification_fs,
        classifier_names=classifier_names
    )


if __name__ == '__main__':
    main()
