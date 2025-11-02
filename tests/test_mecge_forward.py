"""
Test script for MECGE model forward pass.
Tests the model with different configurations (phase, complex, wav features).
"""

import sys
import os
import torch
import yaml

# Add the MECG-E models directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mycode', 'denoising', 'denoising_models', 'my_MECG-E', 'models'))

from MECGE import MECGE


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _run_forward_pass(config_path, feature_type='pha', batch_size=2, signal_length=1000):
    """
    Test the forward pass of MECGE model with both 3D and 4D input formats.

    Note: Function name starts with underscore to avoid pytest collection.
    This function requires parameters and is invoked via main().

    Tests both input formats:
    - 4D format (batch, 1, 1, time): Primary format for ECG denoising pipeline compatibility
    - 3D format (batch, 1, time): Backward compatibility with original implementation

    Args:
        config_path: Path to the configuration YAML file
        feature_type: Type of feature ('pha', 'cpx', 'wav')
        batch_size: Batch size for test
        signal_length: Length of input signal (time dimension)
    """
    print(f"\n{'='*60}")
    print(f"Testing MECGE model with feature type: {feature_type}")
    print(f"{'='*60}")

    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load config
    config = load_config(config_path)
    print(f"\nConfig loaded from: {config_path}")
    print(f"Model parameters:")
    for key, value in config['model'].items():
        print(f"  {key}: {value}")

    # Initialize model
    try:
        model = MECGE(config)
        model = model.to(device)
        model.eval()
        print(f"\n✓ Model initialized successfully and moved to {device}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"\n✗ Model initialization failed: {e}")
        return False

    # =======================================================================
    # Test 1: 4D Input Format (Primary - ECG Pipeline Compatibility)
    # =======================================================================
    print(f"\n{'─'*60}")
    print(f"Test 1: 4D Input Format (batch, 1, 1, time)")
    print(f"{'─'*60}")

    # Create dummy 4D input tensors
    # Shape: [batch_size, 1, 1, time_steps] - matches ECG denoising pipeline
    clean_audio_4d = torch.randn(batch_size, 1, 1, signal_length).to(device)
    noisy_audio_4d = clean_audio_4d + torch.randn_like(clean_audio_4d) * 0.1

    print(f"\nInput shapes (4D format):")
    print(f"  clean_audio: {clean_audio_4d.shape} on {clean_audio_4d.device}")
    print(f"  noisy_audio: {noisy_audio_4d.shape} on {noisy_audio_4d.device}")

    # Test forward pass (training mode) with 4D input
    try:
        model.train()
        loss_4d = model(clean_audio_4d, noisy_audio_4d)
        print(f"\n✓ Training forward pass successful (4D input)")
        print(f"  Loss value: {loss_4d.item():.6f}")
        print(f"  Loss shape: {loss_4d.shape}")

        # Validate loss is scalar
        assert loss_4d.shape == torch.Size([]), f"Expected scalar loss, got shape {loss_4d.shape}"
        print(f"  ✓ Loss is scalar as expected")

        # Test backward pass
        loss_4d.backward()
        print(f"✓ Backward pass successful (4D input)")

    except Exception as e:
        print(f"\n✗ Training forward pass failed (4D input): {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test inference (denoising mode) with 4D input
    try:
        model.eval()
        with torch.no_grad():
            denoised_audio_4d = model.denoising(noisy_audio_4d)
        print(f"\n✓ Inference (denoising) pass successful (4D input)")
        print(f"  Output shape: {denoised_audio_4d.shape}")

        # Validate output shape matches input shape
        assert denoised_audio_4d.shape == noisy_audio_4d.shape, \
            f"Output shape {denoised_audio_4d.shape} doesn't match input shape {noisy_audio_4d.shape}"
        assert denoised_audio_4d.shape == (batch_size, 1, 1, signal_length), \
            f"Expected shape ({batch_size}, 1, 1, {signal_length}), got {denoised_audio_4d.shape}"
        print(f"  ✓ Output shape matches input shape (4D preserved)")

        print(f"  Input signal range: [{noisy_audio_4d.min():.4f}, {noisy_audio_4d.max():.4f}]")
        print(f"  Output signal range: [{denoised_audio_4d.min():.4f}, {denoised_audio_4d.max():.4f}]")

    except Exception as e:
        print(f"\n✗ Inference pass failed (4D input): {e}")
        import traceback
        traceback.print_exc()
        return False

    # =======================================================================
    # Test 2: 3D Input Format (Backward Compatibility)
    # =======================================================================
    print(f"\n{'─'*60}")
    print(f"Test 2: 3D Input Format (batch, 1, time) - Backward Compatibility")
    print(f"{'─'*60}")

    # Create dummy 3D input tensors
    # Shape: [batch_size, 1, time_steps] - original format
    clean_audio_3d = torch.randn(batch_size, 1, signal_length).to(device)
    noisy_audio_3d = clean_audio_3d + torch.randn_like(clean_audio_3d) * 0.1

    print(f"\nInput shapes (3D format):")
    print(f"  clean_audio: {clean_audio_3d.shape} on {clean_audio_3d.device}")
    print(f"  noisy_audio: {noisy_audio_3d.shape} on {noisy_audio_3d.device}")

    # Test forward pass (training mode) with 3D input
    try:
        model.train()
        loss_3d = model(clean_audio_3d, noisy_audio_3d)
        print(f"\n✓ Training forward pass successful (3D input)")
        print(f"  Loss value: {loss_3d.item():.6f}")
        print(f"  Loss shape: {loss_3d.shape}")

        # Validate loss is scalar
        assert loss_3d.shape == torch.Size([]), f"Expected scalar loss, got shape {loss_3d.shape}"
        print(f"  ✓ Loss is scalar as expected")

        # Test backward pass
        loss_3d.backward()
        print(f"✓ Backward pass successful (3D input)")

    except Exception as e:
        print(f"\n✗ Training forward pass failed (3D input): {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test inference (denoising mode) with 3D input
    try:
        model.eval()
        with torch.no_grad():
            denoised_audio_3d = model.denoising(noisy_audio_3d)
        print(f"\n✓ Inference (denoising) pass successful (3D input)")
        print(f"  Output shape: {denoised_audio_3d.shape}")

        # Validate output shape matches input shape
        assert denoised_audio_3d.shape == noisy_audio_3d.shape, \
            f"Output shape {denoised_audio_3d.shape} doesn't match input shape {noisy_audio_3d.shape}"
        assert denoised_audio_3d.shape == (batch_size, 1, signal_length), \
            f"Expected shape ({batch_size}, 1, {signal_length}), got {denoised_audio_3d.shape}"
        print(f"  ✓ Output shape matches input shape (3D preserved)")

        print(f"  Input signal range: [{noisy_audio_3d.min():.4f}, {noisy_audio_3d.max():.4f}]")
        print(f"  Output signal range: [{denoised_audio_3d.min():.4f}, {denoised_audio_3d.max():.4f}]")

    except Exception as e:
        print(f"\n✗ Inference pass failed (3D input): {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n{'='*60}")
    print(f"✓ All tests passed for {feature_type} feature type!")
    print(f"{'='*60}\n")

    return True


def main():
    """Run tests for all feature types."""

    # Get the base directory
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'mycode', 'denoising', 'denoising_models', 'my_MECG-E')
    config_dir = os.path.join(base_dir, 'config')

    # Test configurations
    test_configs = [
        (os.path.join(config_dir, 'MECGE_phase.yaml'), 'pha'),
        (os.path.join(config_dir, 'MECGE_complex.yaml'), 'cpx'),
        (os.path.join(config_dir, 'MECGE_wav.yaml'), 'wav'),
    ]

    results = {}

    print("\n" + "="*60)
    print("MECGE Model Forward Pass Test Suite")
    print("="*60)

    for config_path, feature_type in test_configs:
        if not os.path.exists(config_path):
            print(f"\n⚠ Config file not found: {config_path}")
            results[feature_type] = False
            continue

        try:
            success = _run_forward_pass(config_path, feature_type)
            results[feature_type] = success
        except Exception as e:
            print(f"\n✗ Test failed for {feature_type}: {e}")
            import traceback
            traceback.print_exc()
            results[feature_type] = False

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for feature_type, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{feature_type:10s}: {status}")
    print("="*60)

    # Return exit code
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
