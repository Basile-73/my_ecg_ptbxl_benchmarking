"""
Main script to run ECG denoising experiments.
Follows structure similar to noise_experiment.py
"""
import sys
import os

# Add paths FIRST before any local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '../'))
sys.path.insert(0, os.path.join(script_dir, '../../ecg_noise/source'))

import yaml
import numpy as np
import torch
import json

# Import from local denoising_utils package
from denoising_utils.preprocessing import remove_bad_labels, select_best_lead, bandpass_filter, normalize_signals
from denoising_utils.utils import get_model, create_online_dataloaders, create_stage2_dataloaders
from denoising_utils.training import train_model, predict_with_model
from ecg_noise_factory.noise import NoiseFactory

# Import load_dataset from parent utils directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../classification'))
from utils.utils import load_dataset


class DenoisingExperiment:
    """Experiment class for ECG denoising, similar to NoiseRobustnessExperiment."""

    def __init__(self, config_path: str = 'code/denoising/configs/denoising_config.yaml'):
        """Initialize experiment with config file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set random seeds for reproducibility
        seed = self.config['random_seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if self.config['hardware']['deterministic']:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # Setup paths
        self.datafolder = self.config['datafolder']
        self.outputfolder = self.config['outputfolder']
        self.exp_folder = os.path.join(self.outputfolder, self.config['experiment_name'])

        # Create folder structure
        os.makedirs(self.exp_folder, exist_ok=True)
        os.makedirs(os.path.join(self.exp_folder, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.exp_folder, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.exp_folder, 'results'), exist_ok=True)

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() and
                                   self.config['hardware']['use_cuda'] else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize noise factory
        noise_data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            self.config['noise_data_path']
            )

        # Load noise config path from main config
        self.noise_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            self.config['noise_config_path']
            )

        print(f"Using noise config: {self.noise_config_path}")

        # Create two noise factories to avoid data leakage
        # - Train mode: uses training noise samples (for training data)
        # - Test mode: uses test noise samples (for validation/test data during training)
        # - Eval mode: reserved for post-training evaluation in evaluate_results.py
        self.noise_factory_train = NoiseFactory(
            data_path=noise_data_path,
            sampling_rate=self.config['sampling_frequency'],
            config_path=self.noise_config_path,
            mode='train'
        )

        self.noise_factory_test = NoiseFactory(
            data_path=noise_data_path,
            sampling_rate=self.config['sampling_frequency'],
            config_path=self.noise_config_path,
            mode='test'
        )

        print("Created train and test noise factories (prevents data leakage)")

    def prepare(self):
        """Prepare and preprocess data."""
        print("\n" + "="*80)
        print("PREPARING DATA")
        print("="*80)

        # Load PTB-XL data
        print("\nLoading PTB-XL dataset...")
        data, labels = load_dataset(self.datafolder, self.config['sampling_frequency'])
        print(f"Loaded: {data.shape[0]} samples")

        # Preprocessing pipeline
        print("\n--- Preprocessing Pipeline ---")
        print("1. Removing bad labels...")
        clean_data, clean_labels = remove_bad_labels(data, labels)

        print("2. Selecting best lead...")
        single_lead_data, selected_indices = select_best_lead(
            clean_data, self.config['sampling_frequency']
        )

        print("3. Applying bandpass filter...")
        preproc_config = self.config['preprocessing']
        filtered_data = bandpass_filter(
            single_lead_data,
            lowcut=preproc_config['bandpass_lowcut'],
            highcut=preproc_config['bandpass_highcut'],
            fs=self.config['sampling_frequency'],
            order=preproc_config['bandpass_order']
        )

        # Split data by folds
        print("\n--- Splitting Data ---")
        train_mask = clean_labels.strat_fold <= self.config['train_fold']
        val_mask = clean_labels.strat_fold == self.config['val_fold']
        test_mask = clean_labels.strat_fold == self.config['test_fold']

        clean_train = filtered_data[train_mask]
        clean_val = filtered_data[val_mask]
        clean_test = filtered_data[test_mask]

        print(f"Train: {len(clean_train)}, Val: {len(clean_val)}, Test: {len(clean_test)}")

        # Normalize
        print("\n4. Normalizing signals...")
        clean_train, norm_stats = normalize_signals(
            clean_train,
            method=preproc_config['normalization'],
            axis=preproc_config.get('normalization_axis', 'channel')
        )
        clean_val, _ = normalize_signals(clean_val, stats=norm_stats)
        clean_test, _ = normalize_signals(clean_test, stats=norm_stats)

        # Save clean data and stats
        np.save(os.path.join(self.exp_folder, 'data', 'clean_train.npy'), clean_train)
        np.save(os.path.join(self.exp_folder, 'data', 'clean_val.npy'), clean_val)
        np.save(os.path.join(self.exp_folder, 'data', 'clean_test.npy'), clean_test)
        with open(os.path.join(self.exp_folder, 'data', 'norm_stats.json'), 'w') as f:
            json.dump(norm_stats, f)

        self.clean_train = clean_train
        self.clean_val = clean_val
        self.clean_test = clean_test
        self.input_shape = clean_train[0].shape

        print(f"\n✓ Data preparation complete. Input shape: {self.input_shape}")

    def _add_noise_to_data(self, X_data):
        """Add noise using NoiseFactory (same as noise_experiment.py)."""
        X_array = np.array(X_data)
        # NoiseFactory expects (batch, channel, length)
        # Our data is (batch, length, channel), so transpose
        X_noisy = self.noise_factory.add_noise(
            x=X_array, batch_axis=0, channel_axis=2, length_axis=1
        )
        return X_noisy

    def perform(self):
        """Train models with online noise generation."""
        print("\n" + "="*80)
        print("TRAINING MODELS (with online noising)")
        print("="*80)
        print("Note: Training uses 'train' mode noise samples")
        print("      Validation/Test use 'test' mode to avoid data leakage")
        print("      Post-training evaluation will use 'eval' mode")

        # Dictionary to store trained Stage1 model paths for Stage2 training
        stage1_models = {}  # Will store: {model_name: model_path, f'{model_name}_type': model_type}

        # Train each model
        for model_config in self.config['models']:
            model_name = model_config['name']
            model_type = model_config['type']

            # Check if this is a Stage2 model
            is_stage2 = model_type.lower() in ['stage2', 'drnet']
            stage1_dependency = model_config.get('stage1_model', None) if is_stage2 else None

            print(f"\n{'='*80}")
            print(f"Training: {model_name} ({model_type})")
            if is_stage2 and stage1_dependency:
                print(f"Stage2 model using Stage1: {stage1_dependency}")
            print(f"{'='*80}")

            # For Stage2, check if required Stage1 model is available
            if is_stage2 and stage1_dependency:
                # First check if Stage1 was trained in current run
                if stage1_dependency in stage1_models:
                    stage1_model_path = stage1_models[stage1_dependency]
                    stage1_model_type = stage1_models[f'{stage1_dependency}_type']
                    print(f"Using Stage1 model from current run: {stage1_dependency}")
                else:
                    # Try to find pre-trained Stage1 model from previous run
                    # Check if model config specifies a path
                    stage1_path_key = f'stage1_model_path'
                    if stage1_path_key in model_config:
                        stage1_model_path = model_config[stage1_path_key]
                        print(f"Using pre-trained Stage1 model from config: {stage1_model_path}")
                    else:
                        # Default: look in standard output folder structure
                        stage1_model_path = os.path.join(
                            self.exp_folder, 'models', stage1_dependency, 'best_model.pth'
                        )
                        print(f"Looking for pre-trained Stage1 model: {stage1_model_path}")

                    # Check if file exists
                    if not os.path.exists(stage1_model_path):
                        print(f"ERROR: Stage1 model '{stage1_dependency}' not found!")
                        print(f"  Expected at: {stage1_model_path}")
                        print(f"  Either:")
                        print(f"    1. Train '{stage1_dependency}' in this run by adding it to models list")
                        print(f"    2. Specify 'stage1_model_path' in config pointing to trained model")
                        print(f"  Skipping {model_name}...")
                        continue

                    # Infer Stage1 model type from dependency name
                    stage1_model_type = stage1_dependency
                    print(f"Found pre-trained Stage1 model: {stage1_model_path}")

                # Load trained Stage1 model from disk
                print(f"Loading Stage1 model: {stage1_model_type}")
                stage1_model = get_model(stage1_model_type, input_length=self.input_shape[0], is_stage2=False)
                stage1_model.load_state_dict(torch.load(stage1_model_path, map_location=self.device))
                stage1_model = stage1_model.to(self.device)
                stage1_model.eval()

                # Create Stage2 dataloaders (with Stage1 predictions)
                print("Creating Stage2 dataloaders with Stage1 predictions...")
                train_loader, val_loader, test_loader = create_stage2_dataloaders(
                    self.clean_train, self.clean_val, self.clean_test,
                    self.noise_factory_train, self.noise_factory_test,
                    stage1_model, self.device,
                    batch_size=model_config.get('batch_size', 32),
                    pin_memory=self.config['dataloader'].get('pin_memory', True)
                )

                # Clean up Stage1 model after creating dataloaders to free memory
                del stage1_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Stage1 model unloaded from memory")
            else:
                # Create online dataloaders (generates new noise each epoch)
                print("Creating dataloaders with online noise generation...")
                train_loader, val_loader, test_loader = create_online_dataloaders(
                    self.clean_train, self.clean_val, self.clean_test,
                    self.noise_factory_train, self.noise_factory_test,
                    batch_size=model_config.get('batch_size', 32),
                    pin_memory=self.config['dataloader'].get('pin_memory', True)
                )

            # Load model
            model = get_model(model_type, input_length=self.input_shape[0], is_stage2=is_stage2)

            # Setup paths
            model_folder = os.path.join(self.exp_folder, 'models', model_name)
            os.makedirs(model_folder, exist_ok=True)
            model_path = os.path.join(model_folder, 'best_model.pth')

            # Train
            training_config = {
                'epochs': model_config.get('epochs', 50),
                'lr': model_config.get('lr', 1e-3),
                'optimizer': self.config['training'].get('optimizer', 'adam'),
                'scheduler': self.config['training'].get('scheduler', {}),
                'early_stopping': self.config['training'].get('early_stopping', {})
            }

            history = train_model(
                model, train_loader, val_loader, training_config, model_path, self.device
            )

            # Save history
            if self.config['output']['save_history']:
                history_path = os.path.join(model_folder, 'history.json')
                with open(history_path, 'w') as f:
                    json.dump(history, f)

            # Generate predictions (do this before deleting model)
            print(f"\nGenerating predictions for {model_name}...")
            predictions = predict_with_model(model, test_loader, self.device)

            # Save predictions
            if self.config['output']['save_predictions']:
                pred_path = os.path.join(model_folder, 'predictions.npy')
                np.save(pred_path, predictions)

            # Store Stage1 model paths for potential Stage2 use (not the model itself)
            if not is_stage2:
                stage1_models[model_name] = model_path
                stage1_models[f'{model_name}_type'] = model_type
                print(f"Stage1 model path stored: {model_path}")

            # Clean up model from memory after training and predictions
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Model {model_name} unloaded from memory")

            print(f"✓ {model_name} complete")

        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Run ECG denoising experiments')
    parser.add_argument('--config', type=str, default='code/denoising/configs/denoising_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    # Create and run experiment
    experiment = DenoisingExperiment(args.config)

    # Prepare data
    experiment.prepare()

    # Train models
    experiment.perform()

    print(f"\n✓ Experiment complete. Results in: {experiment.exp_folder}")
    print("\nNext steps:")
    print("  1. Run evaluation: python evaluate_results.py")
    print("  2. Check results in output folder")


if __name__ == '__main__':
    main()
