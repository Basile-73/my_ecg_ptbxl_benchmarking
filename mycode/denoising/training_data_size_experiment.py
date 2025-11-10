"""
Training Data Size Experiment for ECG Denoising Models

This script orchestrates experiments to evaluate denoising model performance
across different training set sizes while maintaining constant total training steps.

Key Features:
- Tests multiple training fold sizes (default: 1, 4, 8)
- Adjusts epochs to maintain equal total training steps across all experiments
- Trains all models specified in base config for each fold size
- Saves comprehensive metadata for downstream analysis
"""
import os
import sys
import yaml
import json
import copy
import argparse
import math
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '../classification'))

# Import required modules
from train import DenoisingExperiment
from utils.utils import load_dataset
from denoising_utils.preprocessing import remove_bad_labels

# Configuration constants
TRAIN_FOLDS = [1, 4, 8]
VAL_FOLD = 9
TEST_FOLD = 10


def calculate_training_sizes(datafolder, sampling_frequency, model_configs, train_folds):
    """
    Calculate training set sizes and steps per epoch for each fold configuration.

    Args:
        datafolder: Path to PTB-XL data directory
        sampling_frequency: Sampling rate for data loading
        model_configs: List of model configuration dictionaries (to get batch sizes)
        train_folds: List of fold sizes to calculate for

    Returns:
        Dictionary mapping fold -> model_name -> {'train_size': int, 'steps_per_epoch': int}
    """
    print("\n" + "="*80)
    print("CALCULATING TRAINING SIZES")
    print("="*80)

    # Load PTB-XL dataset
    print(f"Loading dataset from {datafolder} at {sampling_frequency}Hz...")
    data, raw_labels = load_dataset(datafolder, sampling_frequency)
    print(f"✓ Loaded {len(data)} samples")

    # Remove bad labels (this affects sample count)
    print("\nRemoving bad labels...")
    clean_data, clean_labels = remove_bad_labels(data, raw_labels)
    print(f"✓ After removing bad labels: {len(clean_data)} samples")

    # Calculate sizes for each fold and each model
    print("\nCalculating training sizes for each fold and model...")
    training_sizes = {}

    for fold in train_folds:
        train_mask = clean_labels.strat_fold <= fold
        train_size = int(np.sum(train_mask))

        print(f"\n  Fold {fold}: {train_size} training samples")

        training_sizes[fold] = {}

        for model_config in model_configs:
            model_name = model_config['name']
            batch_size = model_config['batch_size']
            steps_per_epoch = train_size // batch_size  # Accounting for drop_last=True

            # Guard against zero steps_per_epoch
            if steps_per_epoch == 0:
                raise ValueError(
                    f"Batch size ({batch_size}) exceeds training set size ({train_size}) "
                    f"for fold {fold} and model {model_name}. "
                    f"Please reduce batch size or use more training data."
                )

            training_sizes[fold][model_name] = {
                'train_size': train_size,
                'batch_size': batch_size,
                'steps_per_epoch': steps_per_epoch
            }

            print(f"    {model_name:<20} (batch={batch_size:2d}): {steps_per_epoch:4d} steps/epoch")

    return training_sizes


def calculate_epochs_for_constant_steps(training_sizes_dict, reference_fold, reference_epochs):
    """
    Calculate epochs needed for each fold to achieve constant total training steps.
    Uses ceiling to meet or exceed target steps.

    Args:
        training_sizes_dict: Dictionary from calculate_training_sizes()
                            fold -> model_name -> {'train_size', 'batch_size', 'steps_per_epoch'}
        reference_fold: Fold size to use as reference for total steps
        reference_epochs: Number of epochs for reference fold

    Returns:
        Dictionary mapping fold -> model_name -> adjusted_epochs
    """
    print("\n" + "="*80)
    print("CALCULATING ADJUSTED EPOCHS FOR CONSTANT TOTAL STEPS")
    print("="*80)

    print(f"Reference configuration: Fold {reference_fold}, {reference_epochs} epochs")

    # Calculate epochs for each fold and model
    epochs_dict = {}

    # First, calculate reference total steps for each model
    reference_steps = {}
    for model_name, model_info in training_sizes_dict[reference_fold].items():
        ref_steps = model_info['steps_per_epoch']
        reference_total_steps = ref_steps * reference_epochs
        reference_steps[model_name] = reference_total_steps
        print(f"  {model_name}: {reference_total_steps} reference total steps")

    print("\nAdjusted epochs for each fold and model:")
    for fold in sorted(training_sizes_dict.keys()):
        print(f"\n  Fold {fold}:")
        epochs_dict[fold] = {}

        for model_name, model_info in training_sizes_dict[fold].items():
            steps_per_epoch = model_info['steps_per_epoch']

            # Guard against division by zero
            if steps_per_epoch == 0:
                raise ValueError(
                    f"Steps per epoch is zero for fold {fold}, model {model_name}. "
                    f"This should have been caught earlier. Check batch size configuration."
                )

            reference_total_steps = reference_steps[model_name]

            # Use ceiling to meet or exceed target steps
            adjusted_epochs = max(1, math.ceil(reference_total_steps / steps_per_epoch))

            epochs_dict[fold][model_name] = adjusted_epochs

            actual_total_steps = adjusted_epochs * steps_per_epoch
            diff_pct = 100 * (actual_total_steps - reference_total_steps) / reference_total_steps

            print(f"    {model_name:<20}: {adjusted_epochs:4d} epochs "
                  f"({actual_total_steps:6d} total steps, {diff_pct:+.1f}% from reference)")

            # Warning for significant differences
            if abs(diff_pct) > 5:
                print(f"      ⚠️  Warning: {abs(diff_pct):.1f}% difference from reference total steps")

    # Warning for large epoch differences
    all_epochs = [epochs_dict[fold][model] for fold in epochs_dict for model in epochs_dict[fold]]
    if all_epochs:
        max_epochs = max(all_epochs)
        min_epochs = min(all_epochs)
        if max_epochs > 10 * min_epochs:
            print(f"\n⚠️  Warning: Large epoch difference detected (max/min = {max_epochs/min_epochs:.1f}x)")
            print(f"   This may lead to training instabilities or very long training times.")

    return epochs_dict


def create_modified_config(base_config, fold_size, epochs_dict_for_fold, output_base):
    """
    Create a modified config for a specific fold size.

    Args:
        base_config: Original configuration dictionary
        fold_size: Training fold size for this config
        epochs_dict_for_fold: Dictionary mapping model_name -> adjusted_epochs for this fold
        output_base: Base output directory

    Returns:
        Modified configuration dictionary
    """
    # Deep copy to avoid modifying original
    config = copy.deepcopy(base_config)

    # Update fold configuration
    config['train_fold'] = fold_size
    config['val_fold'] = VAL_FOLD
    config['test_fold'] = TEST_FOLD

    # Update experiment name
    config['experiment_name'] = f'training_data_size_fold{fold_size}'

    # Update output folder
    config['outputfolder'] = os.path.join(output_base, 'training_data_size_experiment')

    # Update epochs for each model individually
    for model_config in config['models']:
        model_name = model_config['name']

        # Set epochs for this specific model
        if model_name in epochs_dict_for_fold:
            model_config['epochs'] = epochs_dict_for_fold[model_name]
        else:
            print(f"  ⚠️  Warning: Model {model_name} not found in epochs_dict, using default")

        # Handle Stage2 models - remove or update stage1_model_path to use current fold's Stage1
        if model_config.get('type', '').lower() in ['stage2', 'drnet']:
            stage1_model = model_config.get('stage1_model', None)
            if not stage1_model:
                print(f"  ⚠️  Warning: Stage2 model {model_name} has no stage1_model specified")
            else:
                # Remove stage1_model_path so train.py uses the Stage1 from current fold
                # train.py first checks current run's models/<stage1_model>/best_model.pth
                if 'stage1_model_path' in model_config:
                    del model_config['stage1_model_path']
                    print(f"  ✓ Removed stage1_model_path for {model_name} to use current fold's {stage1_model}")

    return config


def save_experiment_metadata(output_folder, base_config_path, training_sizes,
                             epochs_dict, model_names, fold_experiments):
    """
    Save comprehensive metadata about the experiment.

    Args:
        output_folder: Path to experiment output folder
        base_config_path: Path to original config file
        training_sizes: Dictionary from calculate_training_sizes()
                       fold -> model_name -> {'train_size', 'batch_size', 'steps_per_epoch'}
        epochs_dict: Dictionary from calculate_epochs_for_constant_steps()
                    fold -> model_name -> adjusted_epochs
        model_names: List of model names trained
        fold_experiments: Dictionary mapping fold -> experiment_folder
    """
    print("\n" + "="*80)
    print("SAVING EXPERIMENT METADATA")
    print("="*80)

    # Create metadata dictionary
    metadata = {
        'base_config_path': base_config_path,
        'experiment_date': datetime.now().isoformat(),
        'folds_tested': sorted(list(training_sizes.keys())),
        'val_fold': VAL_FOLD,
        'test_fold': TEST_FOLD,
        'models_trained': model_names,
        'fold_experiments': {}
    }

    # Add per-fold information
    for fold in training_sizes.keys():
        # Get common train_size (same for all models in a fold)
        first_model = list(training_sizes[fold].keys())[0]
        train_size = training_sizes[fold][first_model]['train_size']

        # Store per-model information
        models_info = {}
        for model_name in model_names:
            if model_name in training_sizes[fold] and model_name in epochs_dict[fold]:
                model_info = training_sizes[fold][model_name]
                epochs_used = epochs_dict[fold][model_name]
                steps_per_epoch = model_info['steps_per_epoch']
                total_steps = steps_per_epoch * epochs_used

                models_info[model_name] = {
                    'batch_size': model_info['batch_size'],
                    'steps_per_epoch': steps_per_epoch,
                    'epochs_used': epochs_used,
                    'total_steps': total_steps
                }

        metadata['fold_experiments'][str(fold)] = {
            'train_size': train_size,
            'models': models_info,
            'experiment_folder': fold_experiments.get(fold, 'N/A')
        }

    # Save to JSON
    os.makedirs(output_folder, exist_ok=True)
    metadata_path = os.path.join(output_folder, 'experiment_metadata.json')

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_path}")

    # Print summary table per model
    print("\n" + "-"*80)
    print("EXPERIMENT SUMMARY")
    print("-"*80)

    for model_name in model_names:
        print(f"\nModel: {model_name}")
        print(f"{'Fold':<8} {'Train Size':<12} {'Batch Size':<12} {'Steps/Epoch':<15} {'Epochs':<10} {'Total Steps':<12}")
        print("-"*80)

        for fold in sorted(training_sizes.keys()):
            if model_name in training_sizes[fold] and model_name in epochs_dict[fold]:
                model_info = training_sizes[fold][model_name]
                train_size = model_info['train_size']
                batch_size = model_info['batch_size']
                steps_per_epoch = model_info['steps_per_epoch']
                epochs_used = epochs_dict[fold][model_name]
                total_steps = steps_per_epoch * epochs_used

                print(f"{fold:<8} {train_size:<12} {batch_size:<12} {steps_per_epoch:<15} {epochs_used:<10} {total_steps:<12}")

        print("-"*80)


def run_experiment(base_config_path, output_base, train_folds, reference_fold):
    """
    Main experiment execution logic.

    Args:
        base_config_path: Path to base configuration file
        output_base: Base output directory
        train_folds: List of fold sizes to test
        reference_fold: Reference fold for total steps calculation
    """
    print("\n" + "="*80)
    print("TRAINING DATA SIZE EXPERIMENT")
    print("="*80)
    print(f"Config: {base_config_path}")
    print(f"Output: {output_base}")
    print(f"Training folds: {train_folds}")
    print(f"Reference fold: {reference_fold}")

    # Load base configuration
    print("\nLoading base configuration...")
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    print(f"✓ Configuration loaded")
    print(f"  Models: {[m['name'] for m in base_config['models']]}")
    print(f"  Base epochs: {base_config['models'][0].get('epochs', 'N/A')}")

    # Extract parameters
    datafolder = base_config['datafolder']
    sampling_frequency = base_config['sampling_frequency']
    model_configs = base_config['models']
    reference_epochs = base_config['models'][0]['epochs']

    # Calculate training sizes for all folds and models
    training_sizes = calculate_training_sizes(
        datafolder,
        sampling_frequency,
        model_configs,
        train_folds
    )

    # Calculate adjusted epochs for constant total steps per model
    epochs_dict = calculate_epochs_for_constant_steps(
        training_sizes,
        reference_fold,
        reference_epochs
    )

    # Print experiment plan
    print("\n" + "="*80)
    print("EXPERIMENT PLAN")
    print("="*80)

    for model_config in model_configs:
        model_name = model_config['name']
        print(f"\nModel: {model_name} (batch_size={model_config['batch_size']})")
        print(f"{'Fold':<8} {'Train Size':<12} {'Steps/Epoch':<15} {'Epochs':<10} {'Total Steps':<12}")
        print("-"*80)

        for fold in train_folds:
            if model_name in training_sizes[fold] and model_name in epochs_dict[fold]:
                train_size = training_sizes[fold][model_name]['train_size']
                steps_per_epoch = training_sizes[fold][model_name]['steps_per_epoch']
                epochs = epochs_dict[fold][model_name]
                total_steps = steps_per_epoch * epochs
                print(f"{fold:<8} {train_size:<12} {steps_per_epoch:<15} {epochs:<10} {total_steps:<12}")

        print("-"*80)

    print("="*80)

    # Validate epochs are reasonable
    for fold in epochs_dict.keys():
        for model_name, epochs in epochs_dict[fold].items():
            if epochs < 1 or epochs > 1000:
                print(f"\n⚠️  Warning: Fold {fold}, model {model_name} has unusual epoch count: {epochs}")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Experiment cancelled.")
                    return

    # Track experiment folders
    fold_experiments = {}
    model_names = [m['name'] for m in base_config['models']]

    # Train models for each fold
    experiment_start_time = datetime.now()

    for fold_idx, fold in enumerate(train_folds):
        fold_start_time = datetime.now()

        print("\n" + "="*80)
        print(f"TRAINING WITH {fold} FOLD(S) OF DATA ({fold_idx + 1}/{len(train_folds)})")
        print("="*80)

        try:
            # Set random seed for reproducibility
            torch.manual_seed(42 + fold)
            np.random.seed(42 + fold)

            # Create modified config
            modified_config = create_modified_config(
                base_config,
                fold,
                epochs_dict[fold],
                output_base
            )

            # Save temporary config file
            temp_config_path = os.path.join(
                script_dir,
                f'temp_config_fold{fold}.yaml'
            )

            with open(temp_config_path, 'w') as f:
                yaml.dump(modified_config, f)

            print(f"✓ Created temporary config: {temp_config_path}")

            # Run experiment
            print(f"\nInitializing DenoisingExperiment...")
            experiment = DenoisingExperiment(temp_config_path)

            print(f"Preparing data (preprocessing and splitting)...")
            experiment.prepare()

            print(f"Training models (this may take a while)...")
            experiment.perform()

            # Store experiment folder path
            experiment_folder = os.path.join(
                modified_config['outputfolder'],
                modified_config['experiment_name']
            )
            fold_experiments[fold] = experiment_folder

            # Cleanup
            del experiment
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Remove temporary config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

            # Print timing information
            fold_elapsed = (datetime.now() - fold_start_time).total_seconds()
            print(f"\n✓ Fold {fold} complete in {fold_elapsed/60:.1f} minutes")

            # Estimate remaining time
            if fold_idx < len(train_folds) - 1:
                avg_time_per_fold = (datetime.now() - experiment_start_time).total_seconds() / (fold_idx + 1)
                remaining_folds = len(train_folds) - (fold_idx + 1)
                estimated_remaining = avg_time_per_fold * remaining_folds
                print(f"  Estimated time remaining: {estimated_remaining/60:.1f} minutes")

        except Exception as e:
            print(f"\n❌ ERROR training fold {fold}: {str(e)}")
            print(f"   Continuing with remaining folds...")
            fold_experiments[fold] = f"FAILED: {str(e)}"

            # Cleanup on error
            if 'temp_config_path' in locals() and os.path.exists(temp_config_path):
                os.remove(temp_config_path)

            continue

    # Save experiment metadata
    output_folder = os.path.join(output_base, 'training_data_size_experiment')
    save_experiment_metadata(
        output_folder,
        base_config_path,
        training_sizes,
        epochs_dict,
        model_names,
        fold_experiments
    )

    # Print final summary
    total_elapsed = (datetime.now() - experiment_start_time).total_seconds()
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"Results saved to: {output_folder}")
    print("\nNext steps:")
    print("  1. Review training histories in each fold's models/*/history.json")
    print("  2. Evaluate models using evaluate_similarity.py or evaluate_downstream.py")
    print("  3. Compare performance across different training set sizes")
    print("="*80)


def main():
    """Command-line interface and entry point."""
    parser = argparse.ArgumentParser(
        description='Run training data size experiment for ECG denoising models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: test folds 1, 4, 8 with fold 8 as reference
  python training_data_size_experiment.py

  # Custom fold sizes
  python training_data_size_experiment.py --folds 1 2 4 8 --reference-fold 8

  # Custom config and output location
  python training_data_size_experiment.py \\
    --config mycode/denoising/configs/custom_config.yaml \\
    --output-base mycode/denoising/output/my_experiment
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='mycode/denoising/configs/denoising_config_channel.yaml',
        help='Path to base configuration file'
    )

    parser.add_argument(
        '--output-base',
        type=str,
        default='mycode/denoising/output',
        help='Base output directory for experiment results'
    )

    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        default=[1, 4, 8],
        help='List of training fold sizes to test (e.g., 1 4 8)'
    )

    parser.add_argument(
        '--reference-fold',
        type=int,
        default=8,
        help='Reference fold size for total steps calculation'
    )

    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"❌ ERROR: Config file not found: {args.config}")
        sys.exit(1)

    # Validate reference fold is in folds list
    if args.reference_fold not in args.folds:
        print(f"❌ ERROR: Reference fold {args.reference_fold} must be in folds list {args.folds}")
        sys.exit(1)

    # Run experiment
    try:
        run_experiment(
            base_config_path=args.config,
            output_base=args.output_base,
            train_folds=sorted(args.folds),
            reference_fold=args.reference_fold
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
