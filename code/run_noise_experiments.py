#!/usr/bin/env python
"""
Run noise robustness experiments on ECG classification models.

This script provides a command-line interface to test model robustness
to realistic ECG noise (baseline wander, muscle artifacts, electrode motion, AWGN).

Usage:
    # Test single model (default)
    python run_noise_experiments.py

    # Test specific models
    python run_noise_experiments.py --models fastai_xresnet1d101 fastai_resnet1d_wang

    # Test all available models
    python run_noise_experiments.py --all

    # Quick test (fewer bootstrap samples)
    python run_noise_experiments.py --quick

    # Different noise configuration
    python run_noise_experiments.py --noise-config light

    # Different base experiment
    python run_noise_experiments.py --base-exp exp1
"""

import argparse
import sys
import os
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from experiments.noise_experiment import NoiseRobustnessExperiment


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run noise robustness experiments on pre-trained ECG models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single model (default)
  %(prog)s

  # Test specific models
  %(prog)s --models fastai_xresnet1d101 fastai_resnet1d_wang

  # Test all available models
  %(prog)s --all

  # Quick test (20 bootstrap samples)
  %(prog)s --quick

  # Use light noise configuration
  %(prog)s --noise-config light

  # Test models from exp1 (diagnostic task)
  %(prog)s --base-exp exp1 --all
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='fastai_xresnet1d101',
        help='Single model to test (default: fastai_xresnet1d101)'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        help='Multiple models to test (space-separated)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all available models from base experiment'
    )

    parser.add_argument(
        '--base-exp',
        type=str,
        default='exp0',
        help='Base experiment name (default: exp0)'
    )

    parser.add_argument(
        '--noise-config',
        type=str,
        default='default',
        choices=['default', 'light'],
        help='Noise configuration: default (realistic) or light (less severe)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick evaluation with 20 bootstrap samples instead of 100'
    )

    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=100,
        help='Number of bootstrap samples (default: 100, ignored if --quick is set)'
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=20,
        help='Number of parallel jobs (default: 20)'
    )

    parser.add_argument(
        '--data-folder',
        type=str,
        default='../data/ptbxl/',
        help='Path to PTB-XL data folder'
    )

    parser.add_argument(
        '--output-folder',
        type=str,
        default='../output/',
        help='Path to output folder'
    )

    parser.add_argument(
        '--sampling-rate',
        type=int,
        default=100,
        choices=[100, 500],
        help='Sampling frequency in Hz (default: 100)'
    )

    return parser.parse_args()


def get_available_models(output_folder, base_experiment):
    """Get list of available models from base experiment."""
    base_models_path = Path(output_folder) / base_experiment / 'models'

    if not base_models_path.exists():
        return []

    models = [
        d.name for d in base_models_path.iterdir()
        if d.is_dir() and d.name not in ['naive', 'ensemble', 'Wavelet+NN']
    ]

    return sorted(models)


def main():
    args = parse_args()

    # Check CUDA availability
    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str = str(device)
    else:
        device_str = "CPU (torch not available)"

    # Print header
    print("="*80)
    print("NOISE ROBUSTNESS EXPERIMENT")
    print("="*80)
    print(f"Device: {device_str}")
    print(f"Base experiment: {args.base_exp}")
    print(f"Noise configuration: {args.noise_config}")

    # Determine which models to test
    if args.all:
        models_to_test = get_available_models(args.output_folder, args.base_exp)
        if not models_to_test:
            print(f"\nError: No models found in {args.output_folder}{args.base_exp}/models/")
            print("Please run the base experiment first using reproduce_results.py")
            sys.exit(1)
        print(f"\nTesting all {len(models_to_test)} available models:")
        for i, model in enumerate(models_to_test, 1):
            print(f"  {i}. {model}")
        experiment_name = f'{args.base_exp}_noise_all'
    elif args.models:
        models_to_test = args.models
        print(f"\nTesting {len(models_to_test)} specified models:")
        for i, model in enumerate(models_to_test, 1):
            print(f"  {i}. {model}")
        experiment_name = f'{args.base_exp}_noise'
    else:
        models_to_test = [args.model]
        print(f"\nTesting single model: {args.model}")
        experiment_name = f'{args.base_exp}_noise'

    # Determine bootstrap samples
    if args.quick:
        n_bootstrap = 20
        print(f"\nQuick mode: Using 20 bootstrap samples")
    else:
        n_bootstrap = args.n_bootstrap
        print(f"\nUsing {n_bootstrap} bootstrap samples")

    print("="*80)
    print()

    # Noise config path
    noise_config_path = f'../../ecg_noise/configs/{args.noise_config}.yaml'

    # Create experiment
    try:
        experiment = NoiseRobustnessExperiment(
            experiment_name=experiment_name,
            base_experiment=args.base_exp,
            model_names=models_to_test,
            datafolder=args.data_folder,
            outputfolder=args.output_folder,
            noise_config_path=noise_config_path,
            sampling_frequency=args.sampling_rate
        )
    except Exception as e:
        print(f"Error creating experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run experiment pipeline
    try:
        print("[1/3] Preparing data...")
        print("-" * 80)
        experiment.prepare()

        print("\n[2/3] Generating predictions...")
        print("-" * 80)
        experiment.perform()

        print("\n[3/3] Evaluating performance...")
        print("-" * 80)
        summary = experiment.evaluate(
            n_bootstraping_samples=n_bootstrap,
            n_jobs=args.n_jobs
        )

        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {args.output_folder}{experiment_name}/")

        # Print quick summary
        if len(models_to_test) > 1:
            print("\n" + "="*80)
            print("MODELS RANKED BY NOISE ROBUSTNESS (smallest AUC drop = best)")
            print("="*80)
            test_results = summary[summary['split'] == 'test'].copy()
            test_results = test_results.sort_values('auc_drop')
            print("\nTest Set Performance:")
            print("-" * 80)
            for idx, row in test_results.iterrows():
                print(f"{row['model']:30s} | "
                      f"Clean: {row['clean_auc']:.4f} [{row['clean_auc_ci_lower']:.4f}-{row['clean_auc_ci_upper']:.4f}] | "
                      f"Noisy: {row['noisy_auc']:.4f} [{row['noisy_auc_ci_lower']:.4f}-{row['noisy_auc_ci_upper']:.4f}] | "
                      f"Drop: {row['auc_drop']:.4f}")
        else:
            print("\n" + "="*80)
            print("RESULTS SUMMARY")
            print("="*80)
            test_results = summary[summary['split'] == 'test']
            if len(test_results) > 0:
                row = test_results.iloc[0]
                print(f"\nModel: {row['model']}")
                print(f"  Clean AUC: {row['clean_auc']:.4f} [{row['clean_auc_ci_lower']:.4f}-{row['clean_auc_ci_upper']:.4f}]")
                print(f"  Noisy AUC: {row['noisy_auc']:.4f} [{row['noisy_auc_ci_lower']:.4f}-{row['noisy_auc_ci_upper']:.4f}]")
                print(f"  AUC Drop:  {row['auc_drop']:.4f}")
                print()
                if row['auc_drop'] < 0.04:
                    print("  → Good noise robustness ✓")
                elif row['auc_drop'] < 0.06:
                    print("  → Average noise robustness")
                else:
                    print("  → Poor noise robustness")

        print("\n" + "="*80)
        print(f"\nDetailed results: {args.output_folder}{experiment_name}/results/noise_robustness_summary.csv")

    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
