"""
Quick test script to verify all models can train for one epoch
"""
from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
# from configs.wavelet_configs import *  # Commented out - requires keras/tensorflow
import torch
import sys


def test_single_model(model_config, model_name):
    """Test a single model for one epoch"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        datafolder = 'data/physionet.org/files/ptb-xl/1.0.3/'
        outputfolder = f'mycode/classification/output/test_{model_name}/'

        # Create a modified config with only 1 epoch for quick testing
        import copy
        test_config = copy.deepcopy(model_config)
        test_config['parameters']['epochs'] = 1

        models = [test_config]

        # Test on a single, simple experiment
        e = SCP_Experiment('exp0', 'all', datafolder, outputfolder, models)
        e.prepare()
        e.perform()

        print(f"✓ {model_name} trained successfully!")
        return True

    except Exception as ex:
        print(f"✗ {model_name} FAILED with error:")
        print(f"  {type(ex).__name__}: {ex}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # List of all models to test
    models_to_test = [
        (conf_fastai_xresnet1d101, "XResNet1d101"),
        (conf_fastai_resnet1d_wang, "ResNet1d_wang"),
        (conf_fastai_lstm, "LSTM"),
        (conf_fastai_lstm_bidir, "LSTM_Bidir"),
        (conf_fastai_fcn_wang, "FCN_wang"),
        (conf_fastai_inception1d, "Inception1d"),
        # Note: Wavelet_NN requires keras/tensorflow which is not installed in mamba_test env
        # (conf_wavelet_standard_nn, "Wavelet_NN"),
    ]

    print("Starting model training tests...")
    print(f"Each model will train for 1 epoch on exp0/all")

    results = {}
    for model_config, model_name in models_to_test:
        success = test_single_model(model_config, model_name)
        results[model_name] = success

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for model_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{model_name:20s}: {status}")

    # Return exit code based on results
    all_passed = all(results.values())
    if all_passed:
        print(f"\n✓ All {len(results)} models passed!")
        sys.exit(0)
    else:
        failed_count = sum(1 for v in results.values() if not v)
        print(f"\n✗ {failed_count}/{len(results)} models failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
