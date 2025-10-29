"""
Test script for Mamba SSM model forward pass.
This script uses the mamba_ssm library to test the forward pass with sample input.
"""

import torch
import torch.nn as nn
import sys

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    MAMBA_ERROR = None
except ImportError as e:
    MAMBA_AVAILABLE = False
    MAMBA_ERROR = str(e)
    print(f"Warning: mamba_ssm import failed with error: {e}")
except Exception as e:
    MAMBA_AVAILABLE = False
    MAMBA_ERROR = str(e)
    print(f"Warning: mamba_ssm import failed with unexpected error: {e}")


def test_mamba_forward():
    """Test the Mamba SSM layer with sample input."""
    if not MAMBA_AVAILABLE:
        print("ERROR: mamba_ssm library could not be imported.")
        print(f"Error details: {MAMBA_ERROR}")
        print("\nPossible causes:")
        print("1. mamba-ssm not installed - Install with: pip install mamba-ssm")
        print("2. CUDA kernel compatibility issues with your GPU")
        print("3. Missing or incompatible dependencies")
        print("\nTrying to get more information...")

        # Try importing submodules to get more specific error
        try:
            import mamba_ssm
            print(f"✓ mamba_ssm package found at: {mamba_ssm.__file__}")
            print(f"  Version: {getattr(mamba_ssm, '__version__', 'unknown')}")
        except Exception as e:
            print(f"✗ Cannot import mamba_ssm package: {e}")

        return

    print("=" * 60)
    print("Testing Mamba SSM Forward Pass")
    print("=" * 60)

    # Check CUDA availability
    print(f"\nCUDA Information:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  Current GPU: {torch.cuda.current_device()}")
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")

        # Check GPU compute capability
        capability = torch.cuda.get_device_capability(0)
        compute_capability = float(f"{capability[0]}.{capability[1]}")
        print(f"  GPU compute capability: {compute_capability}")

        device = torch.device("cuda")
        print(f"  Using device: {device}")
    else:
        print("  WARNING: CUDA is not available. Running on CPU.")
        device = torch.device("cpu")
        print(f"  Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Configuration
    batch_size = 4
    seq_len = 100
    d_model = 64
    d_state = 16
    d_conv = 4
    expand = 2

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  State dimension: {d_state}")
    print(f"  Convolution kernel: {d_conv}")
    print(f"  Expansion factor: {expand}")

    # Create model using mamba_ssm library
    print(f"\nCreating Mamba block from mamba_ssm library...")
    model = Mamba(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create sample input
    print(f"\nCreating sample input...")
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    print(f"  Input shape: {x.shape}")
    print(f"  Input device: {x.device}")
    print(f"  Input mean: {x.mean().item():.4f}")
    print(f"  Input std: {x.std().item():.4f}")

    # Forward pass
    print(f"\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")
    print(f"  Output min: {output.min().item():.4f}")
    print(f"  Output max: {output.max().item():.4f}")

    # Verify output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print(f"\n✓ Output shape is correct!")

    # Test gradient flow
    print(f"\nTesting gradient flow...")
    model.train()
    x_grad = torch.randn(batch_size, seq_len, d_model).to(device)
    x_grad.requires_grad_(True)  # Make it require gradients AFTER moving to device
    output_grad = model(x_grad)
    loss = output_grad.mean()
    loss.backward()

    # Check input gradients
    assert x_grad.grad is not None, "Input gradients should be computed"
    print(f"  Input gradient shape: {x_grad.grad.shape}")
    print(f"  Input gradient mean: {x_grad.grad.mean().item():.6f}")
    print(f"  Input gradient std: {x_grad.grad.std().item():.6f}")

    # Check model parameters have gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"  Model parameters with gradients: {params_with_grad}/{total_params}")
    print(f"  ✓ Gradients computed successfully!")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_mamba_forward()
