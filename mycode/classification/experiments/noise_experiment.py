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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from ecg_noise_factory.noise import NoiseFactory


class NoiseRobustnessExperiment():
    '''
    Experiment to test the robustness of trained models to ECG noise.
    Uses pre-trained models from previous experiments and evaluates them on noisy data.
    '''

    def __init__(self, experiment_name, base_experiment, model_names, datafolder, outputfolder,
                 noise_config_path='../../../noise/configs/default.yaml', sampling_frequency=100,
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
        noise_data_path = os.path.join(os.path.dirname(__file__), '../../../noise/data')
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
            from classification_models.fastai_model import fastai_model

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
            from classification_models.wavelet import WaveletModel

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

    def _plot_results(self, summary_df):
        """
        Create visualization of noise robustness results.
        Generates multiple plots showing model performance on clean vs noisy data.
        """
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)

        # Filter for test set only
        test_df = summary_df[summary_df['split'] == 'test'].copy()

        if len(test_df) == 0:
            print("Warning: No test results to plot")
            return

        # Sort by AUC drop (most robust first)
        test_df = test_df.sort_values('auc_drop')

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ECG Model Noise Robustness Analysis', fontsize=16, fontweight='bold', y=0.995)

        # Plot 1: Clean vs Noisy AUC with error bars
        ax1 = axes[0, 0]
        x = np.arange(len(test_df))
        width = 0.35

        # Clean AUC bars
        clean_bars = ax1.bar(x - width/2, test_df['clean_auc'], width,
                            label='Clean', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
        # Clean CI error bars
        clean_err = [test_df['clean_auc'] - test_df['clean_auc_ci_lower'],
                     test_df['clean_auc_ci_upper'] - test_df['clean_auc']]
        ax1.errorbar(x - width/2, test_df['clean_auc'], yerr=clean_err,
                    fmt='none', ecolor='darkgreen', capsize=3, capthick=2)

        # Noisy AUC bars
        noisy_bars = ax1.bar(x + width/2, test_df['noisy_auc'], width,
                            label='Noisy', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
        # Noisy CI error bars
        noisy_err = [test_df['noisy_auc'] - test_df['noisy_auc_ci_lower'],
                     test_df['noisy_auc_ci_upper'] - test_df['noisy_auc']]
        ax1.errorbar(x + width/2, test_df['noisy_auc'], yerr=noisy_err,
                    fmt='none', ecolor='darkred', capsize=3, capthick=2)

        ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax1.set_ylabel('AUC (macro)', fontsize=11, fontweight='bold')
        ax1.set_title('Model Performance: Clean vs Noisy Data', fontsize=12, fontweight='bold', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(test_df['model'], rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=10, loc='lower left')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([max(0.5, test_df[['clean_auc', 'noisy_auc']].min().min() - 0.05), 1.0])

        # Plot 2: AUC Drop (Performance Degradation)
        ax2 = axes[0, 1]
        colors = ['#27ae60' if drop < 0.04 else '#f39c12' if drop < 0.06 else '#c0392b'
                  for drop in test_df['auc_drop']]
        bars2 = ax2.barh(x, test_df['auc_drop'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        ax2.set_yticks(x)
        ax2.set_yticklabels(test_df['model'], fontsize=9)
        ax2.set_xlabel('AUC Drop (Clean - Noisy)', fontsize=11, fontweight='bold')
        ax2.set_title('Performance Degradation Due to Noise', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, axis='x')

        # Add threshold lines
        ax2.axvline(x=0.04, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Good (<0.04)')
        ax2.axvline(x=0.06, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Average (<0.06)')
        ax2.legend(fontsize=9, loc='lower right')

        # Add value labels on bars
        for i, (idx, row) in enumerate(test_df.iterrows()):
            ax2.text(row['auc_drop'] + 0.002, i, f"{row['auc_drop']:.4f}",
                    va='center', fontsize=8, fontweight='bold')

        # Plot 3: Scatter plot - Clean AUC vs Noise Robustness
        ax3 = axes[1, 0]
        scatter = ax3.scatter(test_df['clean_auc'], test_df['auc_drop'],
                             s=200, c=test_df['auc_drop'], cmap='RdYlGn_r',
                             alpha=0.7, edgecolors='black', linewidth=1.5)

        # Add model labels
        for idx, row in test_df.iterrows():
            model_short = row['model'].replace('fastai_', '').replace('wavelet_', '')[:15]
            ax3.annotate(model_short, (row['clean_auc'], row['auc_drop']),
                        fontsize=8, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

        ax3.set_xlabel('Clean AUC', fontsize=11, fontweight='bold')
        ax3.set_ylabel('AUC Drop', fontsize=11, fontweight='bold')
        ax3.set_title('Baseline Performance vs Noise Robustness', fontsize=12, fontweight='bold', pad=10)
        ax3.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('AUC Drop', fontsize=10)

        # Add quadrant lines
        median_clean = test_df['clean_auc'].median()
        median_drop = test_df['auc_drop'].median()
        ax3.axvline(x=median_clean, color='gray', linestyle=':', alpha=0.5)
        ax3.axhline(y=median_drop, color='gray', linestyle=':', alpha=0.5)

        # Plot 4: Confidence Interval Widths
        ax4 = axes[1, 1]
        clean_ci_width = test_df['clean_auc_ci_upper'] - test_df['clean_auc_ci_lower']
        noisy_ci_width = test_df['noisy_auc_ci_upper'] - test_df['noisy_auc_ci_lower']

        x4 = np.arange(len(test_df))
        width4 = 0.35
        ax4.bar(x4 - width4/2, clean_ci_width, width4, label='Clean',
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        ax4.bar(x4 + width4/2, noisy_ci_width, width4, label='Noisy',
               color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1)

        ax4.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax4.set_ylabel('CI Width (90%)', fontsize=11, fontweight='bold')
        ax4.set_title('Confidence Interval Widths (Uncertainty)', fontsize=12, fontweight='bold', pad=10)
        ax4.set_xticks(x4)
        ax4.set_xticklabels(test_df['model'], rotation=45, ha='right', fontsize=9)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')

        # Adjust layout
        plt.tight_layout()

        # Save figure
        output_path = self.outputfolder + self.experiment_name + '/results/noise_robustness_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"\n✓ Visualization saved to: {output_path}")

        # Create a second simplified plot for presentations
        self._plot_simple_summary(test_df)

    def _plot_simple_summary(self, test_df):
        """
        Create a simplified, presentation-ready plot.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        x = np.arange(len(test_df))
        width = 0.35

        # Clean vs Noisy bars
        ax.bar(x - width/2, test_df['clean_auc'], width,
              label='Clean Data', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.bar(x + width/2, test_df['noisy_auc'], width,
              label='Noisy Data', color='#e74c3c', alpha=0.9, edgecolor='black', linewidth=1.5)

        # Add value labels on top of bars
        for i, (idx, row) in enumerate(test_df.iterrows()):
            ax.text(i - width/2, row['clean_auc'] + 0.01, f"{row['clean_auc']:.3f}",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(i + width/2, row['noisy_auc'] + 0.01, f"{row['noisy_auc']:.3f}",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Add drop annotation
            ax.text(i, row['noisy_auc'] - 0.03, f"↓{row['auc_drop']:.3f}",
                   ha='center', va='top', fontsize=8, color='red', fontweight='bold')

        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('AUC (macro)', fontsize=13, fontweight='bold')
        ax.set_title('ECG Model Performance: Clean vs Noisy Data',
                    fontsize=15, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(test_df['model'], rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=12, loc='lower left', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim([max(0.5, test_df[['clean_auc', 'noisy_auc']].min().min() - 0.05), 1.05])

        # Add note about robustness
        note_text = "Lower AUC drop indicates better noise robustness"
        ax.text(0.98, 0.02, note_text, transform=ax.transAxes,
               fontsize=9, ha='right', va='bottom', style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save
        output_path = self.outputfolder + self.experiment_name + '/results/noise_robustness_simple.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Simple plot saved to: {output_path}")

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

        # Generate visualizations
        print("\nGenerating visualizations...")
        try:
            self._plot_results(summary_df)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
            import traceback
            traceback.print_exc()

        return summary_df
