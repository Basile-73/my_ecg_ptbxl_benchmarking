"""
Evaluate denoising models on downstream ECG classification task with per-record results.

This script:
1. Loads pre-trained classification models
2. Applies noise configurations to 12-lead ECG validation data
3. Denoises each lead using trained denoising models
4. Classifies the denoised signals
5. Computes per-record correctness for each diagnostic class
6. Stores results as CSV files with expanded records (one row per record-diagnosis pair)
"""
import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '../classification'))
sys.path.insert(0, os.path.join(script_dir, '../../ecg_noise/source'))

from denoising_utils.utils import get_model
from denoising_utils.preprocessing import normalize_signals, bandpass_filter
from ecg_noise_factory.noise import NoiseFactory
from utils.utils import load_dataset, apply_standardizer, compute_label_aggregations, load_labels_only

# Mapping of diagnostic types to their corresponding classification experiments
DIAGNOSTIC_TYPE_TO_EXP = {
    'superdiagnostic': 'exp1.1.1',
    'subdiagnostic': 'exp1.1',
    'diagnostic': 'exp1'
}


def load_config(config_path='code/denoising/configs/denoising_config.yaml'):
    """Load denoising configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
        from models.fastai_model import fastai_model

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


def denoise_12lead_signal(noisy_12lead, denoising_model, device, batch_size=32,
                          stage1_model=None):
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
    denoised = np.zeros_like(noisy_12lead)

    denoising_model.eval()
    is_stage2 = stage1_model is not None

    # Detect if main denoising model is MECGE
    is_mecge = hasattr(denoising_model, 'denoising')

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
                        denoised_batch = denoising_model.denoising(batch_tensor)  # (batch, 1, 1, time)
                    else:
                        # Standard Stage2 model: need to concatenate noisy signal with Stage1 output
                        # 1. Get Stage1 output using appropriate inference method
                        if is_stage1_mecge:
                            stage1_output = stage1_model.denoising(batch_tensor)  # (batch, 1, 1, time)
                        else:
                            stage1_output = stage1_model(batch_tensor)  # (batch, 1, 1, time)

                        # 2. Concatenate along channel dimension: (batch, 2, 1, time)
                        stage2_input = torch.cat([batch_tensor, stage1_output], dim=1)

                        # 3. Pass through Stage2
                        denoised_batch = denoising_model(stage2_input)  # (batch, 1, 1, time)
                else:
                    # For Stage1: direct pass using appropriate inference method
                    if is_mecge:
                        denoised_batch = denoising_model.denoising(batch_tensor)  # (batch, 1, 1, time)
                    else:
                        denoised_batch = denoising_model(batch_tensor)  # (batch, 1, 1, time)

                denoised_batch = denoised_batch.squeeze(1).permute(0, 2, 1).cpu().numpy()

            denoised[batch_start:batch_end, :, lead_idx] = denoised_batch[:, :, 0]

    return denoised


def load_denoising_model_for_inference(model_config, denoising_exp_folder, device, input_length, stage1_models_cache):
    """
    Load a denoising model for inference (one at a time).

    Args:
        model_config: Model configuration dict
        denoising_exp_folder: Path to experiment folder
        device: torch device
        input_length: Input signal length
        stage1_models_cache: Cache dict for Stage1 models

    Returns:
        Tuple of (model, stage1_model) or (None, None) if loading fails
    """
    model_name = model_config['name']
    model_type = model_config['type']

    try:
        # Check if model exists
        model_folder = os.path.join(denoising_exp_folder, 'models', model_name)
        model_path = os.path.join(model_folder, 'best_model.pth')

        if not os.path.exists(model_path):
            print(f"⚠️  Model {model_name} not found at {model_path}")
            return None, None

        # Load model
        is_stage2 = model_type.lower() in ['stage2', 'drnet']
        model = get_model(
            model_type,
            input_length=input_length,
            is_stage2=is_stage2
        )

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # For Stage2 models, also load the corresponding Stage1 model
        stage1_model = None
        if is_stage2:
            stage1_name = model_config.get('stage1_model', None)
            if stage1_name:
                if stage1_name in stage1_models_cache:
                    stage1_model = stage1_models_cache[stage1_name]
                else:
                    stage1_model_path = os.path.join(
                        denoising_exp_folder, 'models', stage1_name, 'best_model.pth'
                    )
                    if os.path.exists(stage1_model_path):
                        stage1_type = stage1_name.lower()
                        if 'imunet' in stage1_type:
                            stage1_type = 'imunet'
                        elif 'fcn' in stage1_type:
                            stage1_type = 'fcn'
                        elif 'unet' in stage1_type:
                            stage1_type = 'unet'
                        else:
                            stage1_type = stage1_name.lower()

                        stage1_model = get_model(
                            stage1_type,
                            input_length=input_length,
                            is_stage2=False
                        )
                        stage1_model.load_state_dict(torch.load(stage1_model_path, map_location=device))
                        stage1_model.to(device)
                        stage1_model.eval()

                        stage1_models_cache[stage1_name] = stage1_model

        return model, stage1_model

    except Exception as e:
        print(f"⚠️  Error loading model {model_name}: {e}")
        return None, None


def load_validation_data_with_labels(datafolder, sampling_rate, val_fold):
    """
    Load PTB-XL validation data with labels and apply label aggregations.

    Args:
        datafolder: Path to PTB-XL data folder
        sampling_rate: Sampling rate for loading data
        val_fold: Validation fold number

    Returns:
        Tuple of (X_val, record_ids, label_dfs) where:
        - X_val: Validation signals array
        - record_ids: List of record IDs (ecg_id)
        - label_dfs: Dict with keys 'superdiagnostic', 'subdiagnostic', 'diagnostic'
    """
    print("\n" + "-"*80)
    print("Loading validation data with labels...")
    print("-"*80)

    # Load PTB-XL data
    data, raw_labels = load_dataset(datafolder, sampling_rate)
    print(f"Loaded: {data.shape[0]} samples at {sampling_rate}Hz")

    # Filter to validation fold
    val_mask = raw_labels.strat_fold == val_fold
    X_val = data[val_mask]
    Y_val_raw = raw_labels[val_mask].copy()

    # Comment 4: Guard against misconfigured validation folds
    if len(X_val) == 0:
        raise ValueError(
            f"No validation samples found for fold {val_fold} in {datafolder} "
            f"at {sampling_rate}Hz. Check PTB-XL split configuration."
        )

    print(f"Validation samples: {len(X_val)}")
    print(f"Shape: {X_val[0].shape} (timesteps, channels)")

    # Apply label aggregations for each diagnostic type
    label_dfs = {}
    diagnostic_types = ['superdiagnostic', 'subdiagnostic', 'diagnostic']

    for diag_type in diagnostic_types:
        print(f"\nComputing {diag_type} aggregations...")
        label_df = compute_label_aggregations(Y_val_raw.copy(), datafolder, diag_type)
        label_dfs[diag_type] = label_df
        print(f"  ✓ {diag_type}: {len(label_df)} records")

    # Comment 1: Filter to records that have at least one label in any diagnostic type
    # Policy: Use union of all records with at least one label across all types
    # This ensures we only process records that have meaningful diagnostic information
    valid_record_mask = pd.Series(False, index=Y_val_raw.index)

    for diag_type in diagnostic_types:
        label_df = label_dfs[diag_type]
        # Check which records have non-empty labels (diagnostic_len > 0)
        if f'{diag_type}_len' in label_df.columns:
            has_labels = label_df[f'{diag_type}_len'] > 0
            valid_record_mask |= has_labels
        else:
            # Fallback: check if the diagnostic type column contains non-empty lists
            has_labels = label_df[diag_type].apply(
                lambda x: isinstance(x, list) and len(x) > 0 if isinstance(x, list) else pd.notna(x)
            )
            valid_record_mask |= has_labels

    # Extract record IDs only for records with labels
    record_ids = Y_val_raw.index[valid_record_mask].tolist()
    initial_count = len(Y_val_raw)
    filtered_count = len(record_ids)
    print(f"\nFiltered to {filtered_count} records with labels (from {initial_count} total)")

    # Recompute X_val and Y_val_raw to match the filtered record_ids
    X_val = X_val[valid_record_mask.values]
    Y_val_raw = Y_val_raw[valid_record_mask]

    # Also filter label_dfs to only include valid records
    for diag_type in diagnostic_types:
        label_dfs[diag_type] = label_dfs[diag_type].loc[record_ids]
        print(f"  {diag_type}: {len(label_dfs[diag_type])} records after filtering")

    return X_val, record_ids, label_dfs


def create_base_dataframes(record_ids, diag_type, mlb):
    """
    Create base DataFrame with record_id and diagnostic_class columns for ALL classes.

    Args:
        record_ids: List of record IDs
        diag_type: Diagnostic type string (e.g., 'superdiagnostic')
        mlb: MultiLabelBinarizer object for this diagnostic type

    Returns:
        DataFrame with record_id and diagnostic_class columns
    """
    print("\n" + "-"*80)
    print(f"Creating base DataFrame for {diag_type}...")
    print("-"*80)

    rows = []

    # Get ALL classes from mlb
    all_classes = list(mlb.classes_)

    # Create rows for ALL classes for each record
    for record_id in record_ids:
        for diag_class in all_classes:
            rows.append({
                'record_id': record_id,
                'diagnostic_class': diag_class
            })

    # Convert to DataFrame
    base_df = pd.DataFrame(rows, columns=['record_id', 'diagnostic_class'])
    print(f"  {diag_type}: {len(base_df)} rows (expanded from {len(record_ids)} records × {len(all_classes)} classes)")

    return base_df


def compute_per_record_probabilities(y_true, y_pred, record_ids, mlb):
    """
    Compute per-record probabilities for ALL diagnostic classes.

    Args:
        y_true: Ground truth multi-hot labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes) - must be in [0, 1] range
                For fastai models, sigmoid should be applied before passing to this function
        record_ids: List of record IDs
        mlb: MultiLabelBinarizer from classification experiment (for correct class ordering)

    Returns:
        Dict mapping (record_id, diagnostic_class) -> (probability, true_label)
        where probability is a float (0.0 to 1.0), and true_label is 0 or 1
    """
    probability_dict = {}

    # Create mapping from class names to indices using mlb.classes_
    class_to_idx = {cls: idx for idx, cls in enumerate(mlb.classes_)}

    # Process ALL classes for each record
    for record_idx, record_id in enumerate(record_ids):
        for diag_class in mlb.classes_:
            class_idx = class_to_idx[diag_class]

            # Get true label and predicted probability
            true_label = int(y_true[record_idx, class_idx])
            pred_prob = y_pred[record_idx, class_idx]

            # Store probability and true_label
            probability_dict[(record_id, diag_class)] = (pred_prob, true_label)

    return probability_dict


def expand_and_populate_dataframes(base_dataframes, probability_results, model_names, classifier_names):
    """
    Expand base DataFrames with true_label and probability columns for each model/classifier combination.

    Args:
        base_dataframes: Dict of base DataFrames for each diagnostic type
        probability_results: Dict of dicts mapping model -> (record_id, class) -> (probability, true_label)
        model_names: List of model names (including 'clean', 'noisy', and denoising models)
        classifier_names: List of classifier names

    Returns:
        Dict with populated DataFrames for each diagnostic type
    """
    populated_dataframes = {}

    for diag_type, base_df in base_dataframes.items():
        # Create a copy to populate
        result_df = base_df.copy()

        # Add true_label column (extract from any model's results - all have same true_label)
        # Use robust lookup that checks all model/classifier pairs for the first valid match
        def get_true_label(row):
            key = (row['record_id'], row['diagnostic_class'])
            for model_name in model_names:
                for clf_name in classifier_names:
                    result = probability_results.get(model_name, {}).get(clf_name, {}).get(key)
                    if result is not None and not (isinstance(result, tuple) and len(result) >= 2 and pd.isna(result[1])):
                        return result[1] if isinstance(result, tuple) and len(result) >= 2 else np.nan
            return np.nan

        result_df['true_label'] = result_df.apply(get_true_label, axis=1)

        # Add columns for each model/classifier combination
        for model_name in model_names:
            for clf_name in classifier_names:
                col_name = f"{model_name}_{clf_name}"

                # Populate column with probability (first element of tuple)
                result_df[col_name] = result_df.apply(
                    lambda row: probability_results.get(model_name, {}).get(clf_name, {}).get(
                        (row['record_id'], row['diagnostic_class']), (np.nan, np.nan)
                    )[0],  # Extract probability (first element of tuple)
                    axis=1
                )

        populated_dataframes[diag_type] = result_df

    return populated_dataframes


def evaluate_downstream_by_record(config_path='code/denoising/configs/denoising_config.yaml',
                                  base_exp='exp0',
                                  classification_sampling_rate=100,
                                  classifier_names=None,
                                  use_diagnostic_specific_exps=True):
    """
    Main evaluation function.

    Evaluates denoising models on downstream ECG classification, computing per-record
    correctness for each diagnostic class. Each diagnostic type (superdiagnostic,
    subdiagnostic, diagnostic) is evaluated using its corresponding classification
    experiment to ensure correct label encoding.

    Args:
        config_path: Path to denoising config file
        base_exp: Name of base classification experiment (used only if use_diagnostic_specific_exps=False)
        classification_sampling_rate: Sampling rate used for classification models
        classifier_names: List of classification model names to evaluate
        use_diagnostic_specific_exps: If True, use exp1.1.1/exp1.1/exp1 for super/sub/diagnostic.
                                     If False, use base_exp for all diagnostic types.
    """
    # Default classification models if none specified
    if classifier_names is None:
        classifier_names = ['fastai_xresnet1d101', 'fastai_inception1d']

    print("\n" + "="*80)
    print("DOWNSTREAM CLASSIFICATION BY RECORD EVALUATION")
    print("="*80)

    # Load config
    config = load_config(config_path)
    denoising_exp_folder = os.path.join(config['outputfolder'], config['experiment_name'])
    denoising_sampling_rate = config['sampling_frequency']

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and
                         config['hardware']['use_cuda'] else 'cpu')
    print(f"Using device: {device}")

    # Check for sampling rate mismatch
    if denoising_sampling_rate != classification_sampling_rate:
        print(f"\n⚠️  WARNING: Sampling rate mismatch!")
        print(f"   Denoising models trained at: {denoising_sampling_rate}Hz")
        print(f"   Classification models trained at: {classification_sampling_rate}Hz")
    else:
        print(f"\n✓ Sampling rates match: {classification_sampling_rate}Hz")

    # ========================================================================
    # Load validation data with labels (once, shared across all diagnostic types)
    # ========================================================================
    datafolder = config['datafolder']
    val_fold = config['val_fold']

    X_val_raw, record_ids, label_dfs = load_validation_data_with_labels(
        datafolder, classification_sampling_rate, val_fold
    )
    # Note: label_dfs no longer used for DataFrame creation; using mlb.classes_ and y_true instead

    # ========================================================================
    # Read noise configurations
    # ========================================================================
    print("\n" + "-"*80)
    print("Reading noise configurations...")
    print("-"*80)

    if 'evaluation' in config and 'qui_plot' in config['evaluation']:
        noise_configs = config['evaluation']['qui_plot'].get('noise_configs', [])
        print(f"Found {len(noise_configs)} noise configurations in qui_plot")
    else:
        noise_configs = []
        print("No qui_plot configurations found, using default")

    if len(noise_configs) == 0:
        # Fallback to default
        noise_configs = [{
            'name': 'default',
            'path': config['noise_config_path']
        }]

    for nc in noise_configs:
        print(f"  - {nc['name']}: {nc['path']}")

    # Create results folder
    results_folder = os.path.join(denoising_exp_folder, 'downstream_results', 'by_sample')
    os.makedirs(results_folder, exist_ok=True)

    # Get model configs for later use
    model_configs = config['models']

    # Noise data path (shared across all diagnostic types)
    noise_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        config['noise_data_path']
    )

    # Determine which diagnostic types to process
    diagnostic_types = ['superdiagnostic', 'subdiagnostic', 'diagnostic']

    # ========================================================================
    # PROCESS EACH DIAGNOSTIC TYPE INDEPENDENTLY
    # ========================================================================
    # Initialize global predictions cache: predictions_cache[diag_type][config_name][model_name][clf_name]
    predictions_cache = {}
    all_base_dataframes = {}

    # Process each diagnostic type with its corresponding experiment
    for diag_type in tqdm(diagnostic_types, desc="Processing diagnostic types", position=0):
        print(f"\n{'='*80}")
        print(f"PROCESSING DIAGNOSTIC TYPE: {diag_type.upper()}")
        print(f"{'='*80}")

        # ====================================================================
        # Determine experiment for this diagnostic type
        # ====================================================================
        if use_diagnostic_specific_exps:
            exp_name = DIAGNOSTIC_TYPE_TO_EXP[diag_type]
        else:
            exp_name = base_exp

        print(f"Using classification experiment: {exp_name}")

        # ====================================================================
        # Load classification experiment data
        # ====================================================================
        print("\n" + "-"*80)
        print(f"Loading classification experiment data for {diag_type}...")
        print("-"*80)

        base_exp_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            config['classification_outputfolder'], exp_name
        )
        if not os.path.exists(base_exp_path):
            base_exp_path = os.path.join('../../output', exp_name)

        if not os.path.exists(base_exp_path):
            raise FileNotFoundError(
                f"Classification experiment not found: {base_exp_path}\n"
                f"Please run the classification experiments first using reproduce_results.py.\n"
                f"Required experiments: exp1.1.1 (superdiagnostic), exp1.1 (subdiagnostic), exp1 (diagnostic)"
            )

        print(f"Experiment folder: {base_exp_path}")

        # Load y_val and mlb for this diagnostic type
        y_val = np.load(os.path.join(base_exp_path, 'data', 'y_val.npy'), allow_pickle=True)
        n_classes = y_val.shape[1]
        print(f"Number of classes: {n_classes}")

        with open(os.path.join(base_exp_path, 'data', 'mlb.pkl'), 'rb') as f:
            mlb = pickle.load(f)
        print(f"Loaded MultiLabelBinarizer with {len(mlb.classes_)} classes")
        print(f"Sample classes: {list(mlb.classes_[:5])}")

        # Load and apply classification standardizer
        with open(os.path.join(base_exp_path, 'data', 'standard_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)

        X_val = apply_standardizer(X_val_raw.copy(), scaler)
        print("✓ Applied classification standardizer (StandardScaler)")

        # Store clean version
        X_val_clean = X_val.copy()

        # ====================================================================
        # Create base DataFrame for this diagnostic type
        # ====================================================================
        base_dataframe = create_base_dataframes(record_ids, diag_type, mlb)
        all_base_dataframes[diag_type] = base_dataframe
        print(f"Created base DataFrame: {len(base_dataframe)} rows")

        # ====================================================================
        # Initialize predictions cache for this diagnostic type
        # ====================================================================
        predictions_cache[diag_type] = {}

        # ====================================================================
        # PHASE 1: GENERATE AND CACHE PREDICTIONS FOR THIS DIAGNOSTIC TYPE
        # ====================================================================
        print("\n" + "="*80)
        print(f"PHASE 1: GENERATING PREDICTIONS FOR {diag_type.upper()}")
        print("="*80)

        for noise_config in tqdm(noise_configs, desc=f"Noise configs ({diag_type})", position=1):
            config_name = noise_config['name']
            config_path = noise_config['path']

            # Convert relative path to absolute
            if not os.path.isabs(config_path):
                config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    config_path
                )

            print(f"\n{'='*80}")
            print(f"Noise config: {config_name}")
            print(f"{'='*80}")

            # Initialize cache for this config
            predictions_cache[diag_type][config_name] = {}

            # Create NoiseFactory
            noise_factory = NoiseFactory(
                data_path=noise_data_path,
                sampling_rate=classification_sampling_rate,
                config_path=config_path,
                mode='eval'
            )

            # Generate noisy validation data
            X_val_noisy = noise_factory.add_noise(
                x=X_val_clean, batch_axis=0, channel_axis=2, length_axis=1
            )
            print(f"✓ Generated noisy validation data")

            # Store data for later use
            predictions_cache[diag_type][config_name]['X_clean'] = X_val_clean.copy()
            predictions_cache[diag_type][config_name]['X_noisy'] = X_val_noisy.copy()

            # ================================================================
            # Generate clean baseline predictions
            # ================================================================
            print(f"\n--- Generating Clean Predictions ---")
            predictions_cache[diag_type][config_name]['clean'] = {}

            for clf_name in tqdm(classifier_names, desc="Clean predictions", position=2, leave=False):
                try:
                    # Load classifier
                    clf_model = load_classification_model(
                        clf_name,
                        base_exp_path,
                        n_classes,
                        X_val_clean[0].shape,
                        classification_sampling_rate
                    )

                    # Generate and cache predictions
                    y_pred_clean = clf_model.predict(X_val_clean)
                    predictions_cache[diag_type][config_name]['clean'][clf_name] = y_pred_clean

                    # Clean up
                    del clf_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n⚠️  Error with {clf_name} on clean data: {e}")
                    continue

            print(f"✓ Clean predictions cached for {len(predictions_cache[diag_type][config_name]['clean'])} classifiers")

            # ================================================================
            # Generate noisy baseline predictions
            # ================================================================
            print(f"\n--- Generating Noisy Predictions ---")
            predictions_cache[diag_type][config_name]['noisy'] = {}

            for clf_name in tqdm(classifier_names, desc="Noisy predictions", position=2, leave=False):
                try:
                    # Load classifier
                    clf_model = load_classification_model(
                        clf_name,
                        base_exp_path,
                        n_classes,
                        X_val_clean[0].shape,
                        classification_sampling_rate
                    )

                    # Generate and cache predictions
                    y_pred_noisy = clf_model.predict(X_val_noisy)
                    predictions_cache[diag_type][config_name]['noisy'][clf_name] = y_pred_noisy

                    # Clean up
                    del clf_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n⚠️  Error with {clf_name} on noisy data: {e}")
                    continue

            print(f"✓ Noisy predictions cached for {len(predictions_cache[diag_type][config_name]['noisy'])} classifiers")

            # ================================================================
            # Generate denoised predictions
            # ================================================================
            print(f"\n--- Generating Denoised Predictions ---")

            # Local cache for Stage1 models (per noise config)
            stage1_models_cache = {}

            for model_config in tqdm(model_configs, desc="Denoising models", position=2):
                model_name = model_config['name']

                try:
                    # Load denoising model
                    model, stage1_model = load_denoising_model_for_inference(
                        model_config,
                        denoising_exp_folder,
                        device,
                        X_val.shape[1],
                        stage1_models_cache
                    )

                    if model is None:
                        print(f"\n⚠️  Skipping {model_name}: failed to load")
                        continue

                    # Denoise the noisy data
                    X_val_denoised = denoise_12lead_signal(
                        X_val_noisy,
                        model,
                        device,
                        batch_size=32,
                        stage1_model=stage1_model
                    )

                    # Initialize cache for this model
                    predictions_cache[diag_type][config_name][model_name] = {}

                    # Generate predictions with each classifier
                    for clf_name in tqdm(classifier_names, desc=f"{model_name} predictions", position=3, leave=False):
                        try:
                            # Load classifier
                            clf_model = load_classification_model(
                                clf_name,
                                base_exp_path,
                                n_classes,
                                X_val_clean[0].shape,
                                classification_sampling_rate
                            )

                            # Generate and cache predictions
                            y_pred_denoised = clf_model.predict(X_val_denoised)
                            predictions_cache[diag_type][config_name][model_name][clf_name] = y_pred_denoised

                            # Clean up
                            del clf_model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        except Exception as e:
                            print(f"\n⚠️  Error with {clf_name} on {model_name}: {e}")
                            continue

                    # Clean up denoising model
                    del model
                    if stage1_model is not None and model_config.get('stage1_model') not in stage1_models_cache:
                        del stage1_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    print(f"\n  ✓ {model_name}: predictions cached for {len(predictions_cache[diag_type][config_name][model_name])} classifiers")

                except Exception as e:
                    print(f"\n⚠️  Error processing {model_name}: {e}")
                    continue

            # Clean up Stage1 models cache
            for stage1_name in list(stage1_models_cache.keys()):
                del stage1_models_cache[stage1_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ====================================================================
        # PHASE 2: COMPUTE PER-RECORD PROBABILITIES FOR THIS DIAGNOSTIC TYPE
        # ====================================================================
        print("\n" + "="*80)
        print(f"PHASE 2: COMPUTING PROBABILITIES FOR {diag_type.upper()}")
        print("="*80)

        for noise_config in tqdm(noise_configs, desc=f"Computing probabilities ({diag_type})", position=1):
            config_name = noise_config['name']

            print(f"\n{'='*80}")
            print(f"Computing probabilities for: {config_name}")
            print(f"{'='*80}")

            # Get model names from cached predictions (excluding data keys)
            model_names = [k for k in predictions_cache[diag_type][config_name].keys() if k not in ['X_clean', 'X_noisy']]

            # Initialize probabilities storage
            all_probabilities = {}

            # Process each model
            for model_name in model_names:
                all_probabilities[model_name] = {}

                # Check if predictions exist for this model
                if model_name not in predictions_cache[diag_type][config_name]:
                    print(f"⚠️  No predictions found for {model_name}, skipping...")
                    continue

                # Process each classifier
                for clf_name in classifier_names:
                    # Check if predictions exist for this classifier
                    if clf_name not in predictions_cache[diag_type][config_name][model_name]:
                        print(f"⚠️  No predictions found for {model_name}+{clf_name}, skipping...")
                        continue

                    # Retrieve cached predictions
                    y_pred = predictions_cache[diag_type][config_name][model_name][clf_name]

                    # Apply sigmoid to convert logits to probabilities for fastai models
                    # (wavelet models already output probabilities)
                    if 'fastai' in clf_name:
                        y_pred_prob = 1 / (1 + np.exp(-y_pred))
                    else:
                        y_pred_prob = y_pred

                    # Compute probabilities for this diagnostic type
                    probabilities = compute_per_record_probabilities(
                        y_val, y_pred_prob, record_ids, mlb
                    )
                    all_probabilities[model_name][clf_name] = probabilities

            # ================================================================
            # Save results for this diagnostic type
            # ================================================================
            print(f"\n--- Saving {diag_type} probability results for {config_name} ---")

            # Populate DataFrame
            populated_df = expand_and_populate_dataframes(
                {diag_type: all_base_dataframes[diag_type]},
                all_probabilities,
                model_names,
                classifier_names
            )[diag_type]

            # Save to CSV
            output_path = os.path.join(
                results_folder,
                f"{diag_type}_{config_name}_by_record_probabilities.csv"
            )
            populated_df.to_csv(output_path, index=False)
            print(f"  ✓ Saved: {output_path}")
            print(f"    Rows: {len(populated_df)}, Columns: {len(populated_df.columns)}")

    # ========================================================================
    # Print summary
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {results_folder}")
    print(f"\nSummary:")
    print(f"  - Records: {len(record_ids)}")
    for diag_type in diagnostic_types:
        exp_used = DIAGNOSTIC_TYPE_TO_EXP[diag_type] if use_diagnostic_specific_exps else base_exp
        print(f"  - {diag_type}: {len(all_base_dataframes[diag_type])} expanded rows (using {exp_used})")
    print(f"  - Noise configs processed: {len(noise_configs)}")
    # Count unique denoising models from predictions cache
    denoising_model_count = 0
    if len(diagnostic_types) > 0 and len(noise_configs) > 0:
        first_diag = diagnostic_types[0]
        first_config = noise_configs[0]['name']
        if first_diag in predictions_cache and first_config in predictions_cache[first_diag]:
            denoising_model_count = len([k for k in predictions_cache[first_diag][first_config].keys()
                                        if k not in ['X_clean', 'X_noisy', 'clean', 'noisy']])
    print(f"  - Denoising models: {denoising_model_count}")
    print(f"  - Classification models: {len(classifier_names)}")


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
        description='Evaluate denoising models on downstream ECG classification with per-record results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default models (xresnet1d101, inception1d)
  python evaluate_downstream_by_record.py

  # Evaluate all 6 models
  python evaluate_downstream_by_record.py --classifiers all

  # Evaluate specific models
  python evaluate_downstream_by_record.py --classifiers fastai_xresnet1d101 fastai_lstm

  # With custom experiment settings
  python evaluate_downstream_by_record.py --base-exp exp1 --classification-fs 500 --classifiers all
        """
    )
    parser.add_argument('--config', type=str, default='code/denoising/configs/denoising_config.yaml',
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

    evaluate_downstream_by_record(
        config_path=args.config,
        base_exp=args.base_exp,
        classification_sampling_rate=args.classification_fs,
        classifier_names=classifier_names
    )


if __name__ == '__main__':
    main()
