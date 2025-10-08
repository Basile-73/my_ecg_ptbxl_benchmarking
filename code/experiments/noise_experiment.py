import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ecg_noise/source'))

from utils import utils
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat
from pathlib import Path

from ecg_noise_factory.noise import NoiseFactory


class NoiseRobustnessExperiment():
    '''
    Experiment to test the robustness of trained models to ECG noise.
    Uses pre-trained models from previous experiments and evaluates them on noisy data.
    '''

    def __init__(self, experiment_name, base_experiment, model_names, datafolder, outputfolder,
                 noise_config_path='../../ecg_noise/configs/default.yaml', sampling_frequency=100,
                 train_fold=8, val_fold=9, test_fold=10):
        """
        Args:
            experiment_name: Name for this noise experiment (e.g., 'exp0_noise')
            base_experiment: Name of the base experiment with trained models (e.g., 'exp0')
            model_names: List of model names to test (e.g., ['fastai_xresnet1d101'])
            datafolder: Path to PTB-XL data folder
            outputfolder: Path to output folder
            noise_config_path: Path to noise configuration YAML file
            sampling_frequency: Sampling frequency (100 or 500 Hz)
            train_fold: Training fold number (not used, just for compatibility)
            val_fold: Validation fold number
            test_fold: Test fold number
        """
        self.experiment_name = experiment_name
        self.base_experiment = base_experiment
        self.model_names = model_names
        self.datafolder = datafolder
        self.outputfolder = outputfolder
        self.sampling_frequency = sampling_frequency
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold

        # Path to noise data and config
        noise_data_path = os.path.join(os.path.dirname(__file__), '../../ecg_noise/data')
        self.noise_config_path = os.path.join(os.path.dirname(__file__), noise_config_path)

        # Initialize noise factory
        self.noise_factory = NoiseFactory(
            data_path=noise_data_path,
            sampling_rate=sampling_frequency,
            config_path=self.noise_config_path,
            mode='test'  # Use test split of noise data
        )

        # Create folder structure
        if not os.path.exists(self.outputfolder + self.experiment_name):
            os.makedirs(self.outputfolder + self.experiment_name)
            if not os.path.exists(self.outputfolder + self.experiment_name + '/results/'):
                os.makedirs(self.outputfolder + self.experiment_name + '/results/')
            if not os.path.exists(self.outputfolder + self.experiment_name + '/data/'):
                os.makedirs(self.outputfolder + self.experiment_name + '/data/')

    def prepare(self):
        """
        Load the original clean data and labels from the base experiment.
        Also generate noisy versions of the test and validation sets.
        """
        print(f"Preparing noise robustness experiment based on {self.base_experiment}...")

        # Load PTB-XL data (same as base experiment)
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # Load the preprocessed data and labels from base experiment
        base_exp_path = self.outputfolder + self.base_experiment + '/'

        # Load labels
        self.y_val = np.load(base_exp_path + 'data/y_val.npy', allow_pickle=True)
        self.y_test = np.load(base_exp_path + 'data/y_test.npy', allow_pickle=True)

        # Extract validation and test data based on fold information
        self.X_val = self.data[self.raw_labels.strat_fold == self.val_fold]
        self.X_test = self.data[self.raw_labels.strat_fold == self.test_fold]

        # Apply same preprocessing as base experiment
        # Load scaler parameters from base experiment
        with open(base_exp_path + 'data/standard_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Apply scaling (same way as in utils.apply_standardizer)
        X_val_tmp = []
        for x in self.X_val:
            x_shape = x.shape
            X_val_tmp.append(scaler.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
        self.X_val = np.array(X_val_tmp)

        X_test_tmp = []
        for x in self.X_test:
            x_shape = x.shape
            X_test_tmp.append(scaler.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
        self.X_test = np.array(X_test_tmp)

        # Store clean versions
        self.X_val_clean = self.X_val.copy()
        self.X_test_clean = self.X_test.copy()

        # Generate noisy versions
        print("Generating noisy validation set...")
        self.X_val_noisy = self._add_noise_to_data(self.X_val_clean)

        print("Generating noisy test set...")
        self.X_test_noisy = self._add_noise_to_data(self.X_test_clean)

        # Save noisy data for future use
        self.X_val_noisy.dump(self.outputfolder + self.experiment_name + '/data/X_val_noisy.npy')
        self.X_test_noisy.dump(self.outputfolder + self.experiment_name + '/data/X_test_noisy.npy')
        self.y_val.dump(self.outputfolder + self.experiment_name + '/data/y_val.npy')
        self.y_test.dump(self.outputfolder + self.experiment_name + '/data/y_test.npy')

        # Also save clean data for reference
        self.X_val_clean.dump(self.outputfolder + self.experiment_name + '/data/X_val_clean.npy')
        self.X_test_clean.dump(self.outputfolder + self.experiment_name + '/data/X_test_clean.npy')

        print(f"Data preparation complete. Validation samples: {len(self.X_val_noisy)}, Test samples: {len(self.X_test_noisy)}")

    def _add_noise_to_data(self, X_data):
        """
        Add noise to ECG data using the NoiseFactory.

        Args:
            X_data: Array of ECG signals, shape (n_samples, n_timesteps, n_channels)

        Returns:
            Noisy ECG data with the same shape
        """
        # Convert list to array if needed
        X_array = np.array(X_data)

        # NoiseFactory expects (batch, channel, length)
        # PTB-XL data is (batch, length, channel), so we need to transpose
        # Add noise expects axes: batch_axis, channel_axis, length_axis
        X_noisy = self.noise_factory.add_noise(
            x=X_array,
            batch_axis=0,
            channel_axis=2,
            length_axis=1
        )

        return X_noisy

    def perform(self):
        """
        Load pre-trained models and generate predictions on both clean and noisy data.
        No training is performed - only inference.
        """
        print("Generating predictions with pre-trained models...")

        for modelname in self.model_names:
            print(f"\nProcessing model: {modelname}")

            # Path to pre-trained model
            base_model_path = self.outputfolder + self.base_experiment + '/models/' + modelname + '/'

            if not os.path.exists(base_model_path):
                print(f"Warning: Model {modelname} not found in {base_model_path}. Skipping.")
                continue

            # Create output folder for this model
            mpath = self.outputfolder + self.experiment_name + '/models/' + modelname + '/'
            if not os.path.exists(mpath):
                os.makedirs(mpath)
            if not os.path.exists(mpath + 'results/'):
                os.makedirs(mpath + 'results/')

            # Load model predictions on clean data from base experiment
            # (These should already exist from the base experiment)
            y_val_clean_pred = np.load(base_model_path + 'y_val_pred.npy', allow_pickle=True)
            y_test_clean_pred = np.load(base_model_path + 'y_test_pred.npy', allow_pickle=True)

            # Save clean predictions to this experiment folder for reference
            y_val_clean_pred.dump(mpath + 'y_val_clean_pred.npy')
            y_test_clean_pred.dump(mpath + 'y_test_clean_pred.npy')

            # Generate predictions on noisy data
            print(f"  Generating predictions on noisy data...")
            y_val_noisy_pred = self._predict_with_model(modelname, self.X_val_noisy)
            y_test_noisy_pred = self._predict_with_model(modelname, self.X_test_noisy)

            # Save noisy predictions
            y_val_noisy_pred.dump(mpath + 'y_val_noisy_pred.npy')
            y_test_noisy_pred.dump(mpath + 'y_test_noisy_pred.npy')

            print(f"  Predictions saved to {mpath}")

    def _predict_with_model(self, modelname, X_data):
        """
        Load a pre-trained model and generate predictions.

        Args:
            modelname: Name of the model
            X_data: Input data for prediction

        Returns:
            Predictions array
        """
        # Load model configuration from base experiment
        base_model_path = self.outputfolder + self.base_experiment + '/models/' + modelname + '/'

        # Get number of classes from existing predictions
        y_val_base = np.load(self.outputfolder + self.base_experiment + '/data/y_val.npy', allow_pickle=True)
        n_classes = y_val_base.shape[1]

        # Get input shape
        input_shape = X_data[0].shape

        # Determine model type based on name
        if modelname.startswith('fastai_'):
            from models.fastai_model import fastai_model

            # Create model instance (we'll load weights, so parameters don't matter much)
            model = fastai_model(
                modelname,
                n_classes,
                self.sampling_frequency,
                base_model_path,
                input_shape
            )

            # Generate predictions
            predictions = model.predict(X_data)

            return predictions

        elif modelname.startswith('wavelet_'):
            from models.wavelet import WaveletModel

            # Create model instance
            model = WaveletModel(
                modelname,
                n_classes,
                self.sampling_frequency,
                base_model_path,
                input_shape
            )

            # Generate predictions
            predictions = model.predict(X_data)

            return predictions
        else:
            raise NotImplementedError(f"Model type for {modelname} not yet implemented. "
                                      f"Supported: fastai_*, wavelet_*")

    def evaluate(self, n_bootstraping_samples=100, n_jobs=20):
        """
        Evaluate models on both clean and noisy data using bootstrapping for confidence intervals.
        Compare performance degradation due to noise.
        """
        print("\nEvaluating models...")

        # Load labels
        y_val = np.load(self.outputfolder + self.experiment_name + '/data/y_val.npy', allow_pickle=True)
        y_test = np.load(self.outputfolder + self.experiment_name + '/data/y_test.npy', allow_pickle=True)

        # Generate bootstrap samples
        print("Generating bootstrap samples...")
        val_samples = np.array(utils.get_appropriate_bootstrap_samples(y_val, n_bootstraping_samples))
        test_samples = np.array(utils.get_appropriate_bootstrap_samples(y_test, n_bootstraping_samples))

        # Store samples for reproducibility
        val_samples.dump(self.outputfolder + self.experiment_name + '/val_bootstrap_ids.npy')
        test_samples.dump(self.outputfolder + self.experiment_name + '/test_bootstrap_ids.npy')

        # Evaluate each model
        results_summary = []

        for modelname in self.model_names:
            print(f"\nEvaluating model: {modelname}")

            mpath = self.outputfolder + self.experiment_name + '/models/' + modelname + '/'
            rpath = mpath + 'results/'

            if not os.path.exists(mpath):
                print(f"  Model {modelname} not found. Skipping.")
                continue

            # Load predictions
            y_val_clean_pred = np.load(mpath + 'y_val_clean_pred.npy', allow_pickle=True)
            y_test_clean_pred = np.load(mpath + 'y_test_clean_pred.npy', allow_pickle=True)
            y_val_noisy_pred = np.load(mpath + 'y_val_noisy_pred.npy', allow_pickle=True)
            y_test_noisy_pred = np.load(mpath + 'y_test_noisy_pred.npy', allow_pickle=True)

            # Create multiprocessing pool
            pool = multiprocessing.Pool(n_jobs)

            # Evaluate on clean validation data
            print("  Evaluating on clean validation data...")
            val_clean_df = pd.concat(pool.starmap(
                utils.generate_results,
                zip(val_samples, repeat(y_val), repeat(y_val_clean_pred), repeat(None))
            ))
            val_clean_point = utils.generate_results(range(len(y_val)), y_val, y_val_clean_pred, None)
            val_clean_result = pd.DataFrame(
                np.array([
                    val_clean_point.mean().values,
                    val_clean_df.mean().values,
                    val_clean_df.quantile(0.05).values,
                    val_clean_df.quantile(0.95).values
                ]),
                columns=val_clean_df.columns,
                index=['point', 'mean', 'lower', 'upper']
            )

            # Evaluate on noisy validation data
            print("  Evaluating on noisy validation data...")
            val_noisy_df = pd.concat(pool.starmap(
                utils.generate_results,
                zip(val_samples, repeat(y_val), repeat(y_val_noisy_pred), repeat(None))
            ))
            val_noisy_point = utils.generate_results(range(len(y_val)), y_val, y_val_noisy_pred, None)
            val_noisy_result = pd.DataFrame(
                np.array([
                    val_noisy_point.mean().values,
                    val_noisy_df.mean().values,
                    val_noisy_df.quantile(0.05).values,
                    val_noisy_df.quantile(0.95).values
                ]),
                columns=val_noisy_df.columns,
                index=['point', 'mean', 'lower', 'upper']
            )

            # Evaluate on clean test data
            print("  Evaluating on clean test data...")
            test_clean_df = pd.concat(pool.starmap(
                utils.generate_results,
                zip(test_samples, repeat(y_test), repeat(y_test_clean_pred), repeat(None))
            ))
            test_clean_point = utils.generate_results(range(len(y_test)), y_test, y_test_clean_pred, None)
            test_clean_result = pd.DataFrame(
                np.array([
                    test_clean_point.mean().values,
                    test_clean_df.mean().values,
                    test_clean_df.quantile(0.05).values,
                    test_clean_df.quantile(0.95).values
                ]),
                columns=test_clean_df.columns,
                index=['point', 'mean', 'lower', 'upper']
            )

            # Evaluate on noisy test data
            print("  Evaluating on noisy test data...")
            test_noisy_df = pd.concat(pool.starmap(
                utils.generate_results,
                zip(test_samples, repeat(y_test), repeat(y_test_noisy_pred), repeat(None))
            ))
            test_noisy_point = utils.generate_results(range(len(y_test)), y_test, y_test_noisy_pred, None)
            test_noisy_result = pd.DataFrame(
                np.array([
                    test_noisy_point.mean().values,
                    test_noisy_df.mean().values,
                    test_noisy_df.quantile(0.05).values,
                    test_noisy_df.quantile(0.95).values
                ]),
                columns=test_noisy_df.columns,
                index=['point', 'mean', 'lower', 'upper']
            )

            pool.close()

            # Save results
            val_clean_result.to_csv(rpath + 'val_clean_results.csv')
            val_noisy_result.to_csv(rpath + 'val_noisy_results.csv')
            test_clean_result.to_csv(rpath + 'test_clean_results.csv')
            test_noisy_result.to_csv(rpath + 'test_noisy_results.csv')

            print(f"  Results saved to {rpath}")

            # Compute performance degradation
            degradation_val = {
                'model': modelname,
                'split': 'validation',
                'clean_auc': val_clean_result.loc['point', 'macro_auc'],
                'noisy_auc': val_noisy_result.loc['point', 'macro_auc'],
                'auc_drop': val_clean_result.loc['point', 'macro_auc'] - val_noisy_result.loc['point', 'macro_auc'],
                'clean_auc_ci_lower': val_clean_result.loc['lower', 'macro_auc'],
                'clean_auc_ci_upper': val_clean_result.loc['upper', 'macro_auc'],
                'noisy_auc_ci_lower': val_noisy_result.loc['lower', 'macro_auc'],
                'noisy_auc_ci_upper': val_noisy_result.loc['upper', 'macro_auc'],
            }

            degradation_test = {
                'model': modelname,
                'split': 'test',
                'clean_auc': test_clean_result.loc['point', 'macro_auc'],
                'noisy_auc': test_noisy_result.loc['point', 'macro_auc'],
                'auc_drop': test_clean_result.loc['point', 'macro_auc'] - test_noisy_result.loc['point', 'macro_auc'],
                'clean_auc_ci_lower': test_clean_result.loc['lower', 'macro_auc'],
                'clean_auc_ci_upper': test_clean_result.loc['upper', 'macro_auc'],
                'noisy_auc_ci_lower': test_noisy_result.loc['lower', 'macro_auc'],
                'noisy_auc_ci_upper': test_noisy_result.loc['upper', 'macro_auc'],
            }

            results_summary.append(degradation_val)
            results_summary.append(degradation_test)

        # Save summary
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(self.outputfolder + self.experiment_name + '/results/noise_robustness_summary.csv', index=False)

        print("\n" + "="*80)
        print("NOISE ROBUSTNESS EVALUATION SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        print(f"\nFull results saved to: {self.outputfolder + self.experiment_name + '/results/'}")

        return summary_df
