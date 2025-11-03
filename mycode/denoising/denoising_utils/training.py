"""
Training module for ECG denoising models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import os


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: dict, model_path: str, device: torch.device) -> dict:
    """
    Train a denoising model.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        model_path: Path to save best model
        device: Device to train on

    Returns:
        Dictionary with training history
    """
    model = model.to(device)

    # Detect if model is MECGE (has denoising method)
    is_mecge = hasattr(model, 'denoising')
    if is_mecge:
        print("Detected MECGE model - using internal loss computation")
    else:
        print("Using standard model interface with external MSE loss")

    # Detect if model is mamba_stft_unet (wrapped or unwrapped)
    # Get the candidate model (unwrap if needed)
    candidate = getattr(model, 'base_model', model)
    is_mamba_stft_unet = False

    # Try to import TinyMambaSTFTUNet for isinstance checking
    TinyMambaSTFTUNet = None
    try:
        import sys
        mamba_stft_unet_path = os.path.join(os.path.dirname(__file__), '../denoising_models/mamba_stft_unet')
        sys.path.insert(0, mamba_stft_unet_path)
        from model import TinyMambaSTFTUNet as _TinyMambaSTFTUNet
        TinyMambaSTFTUNet = _TinyMambaSTFTUNet
    except ImportError:
        pass

    # Perform detection using isinstance if import succeeded, or by class name
    if TinyMambaSTFTUNet is not None:
        is_mamba_stft_unet = isinstance(candidate, TinyMambaSTFTUNet) or isinstance(model, TinyMambaSTFTUNet)
    else:
        is_mamba_stft_unet = candidate.__class__.__name__ == 'TinyMambaSTFTUNet'

    if is_mamba_stft_unet:
        print("Detected TinyMambaSTFTUNet model - using custom L1STFTBandpassLoss")

    # Detect if model is mamba_stft_unet_v2 (wrapped or unwrapped)
    candidate = getattr(model, 'base_model', model)
    is_mamba_stft_unet_v2 = False

    # Try to import TinyMambaSTFTUNetV2 for isinstance checking
    TinyMambaSTFTUNetV2 = None
    try:
        import sys
        mamba_stft_unet_path = os.path.join(os.path.dirname(__file__), '../denoising_models/mamba_stft_unet')
        if mamba_stft_unet_path not in sys.path:
            sys.path.insert(0, mamba_stft_unet_path)
        from model_v2 import TinyMambaSTFTUNetV2 as _TinyMambaSTFTUNetV2
        TinyMambaSTFTUNetV2 = _TinyMambaSTFTUNetV2
    except ImportError:
        pass

    # Perform detection using isinstance if import succeeded, or by class name
    if TinyMambaSTFTUNetV2 is not None:
        is_mamba_stft_unet_v2 = isinstance(candidate, TinyMambaSTFTUNetV2) or isinstance(model, TinyMambaSTFTUNetV2)
    else:
        is_mamba_stft_unet_v2 = candidate.__class__.__name__ == 'TinyMambaSTFTUNetV2'

    if is_mamba_stft_unet_v2:
        print("Detected TinyMambaSTFTUNetV2 model - using EnhancedSTFTLoss with gradient clipping")

    # Conditionally instantiate the appropriate loss function
    if is_mamba_stft_unet_v2:
        # Import the enhanced loss only when needed
        try:
            import sys
            mamba_stft_unet_path = os.path.join(os.path.dirname(__file__), '../denoising_models/mamba_stft_unet')
            if mamba_stft_unet_path not in sys.path:
                sys.path.insert(0, mamba_stft_unet_path)
            from loss_v2 import EnhancedSTFTLoss
            criterion = EnhancedSTFTLoss(sr=250, w_time=1.0, w_mr_stft=1.0, w_phase=1.0, w_complex=1.0, w_consistency=1.0, w_bandpower=0.2)
            print("Using EnhancedSTFTLoss with sr=250, w_time=1.0, w_mr_stft=1.0, w_phase=1.0, w_complex=1.0, w_consistency=1.0, w_bandpower=0.2")
        except ImportError as e:
            print(f"Warning: Could not import EnhancedSTFTLoss, falling back to MSELoss. Error: {e}")
            criterion = nn.MSELoss()
            print("Using standard MSELoss (fallback)")
    elif is_mamba_stft_unet:
        # Import the custom loss only when needed
        try:
            import sys
            mamba_stft_unet_path = os.path.join(os.path.dirname(__file__), '../denoising_models/mamba_stft_unet')
            if mamba_stft_unet_path not in sys.path:
                sys.path.insert(0, mamba_stft_unet_path)
            from loss import L1STFTBandpassLoss
            criterion = L1STFTBandpassLoss(sr=250, w_l1=1.0, w_stft=0.5, w_bp=0.05)
            print("Using L1STFTBandpassLoss with sr=250, w_l1=1.0, w_stft=0.5, w_bp=0.05")
        except ImportError as e:
            print(f"Warning: Could not import L1STFTBandpassLoss, falling back to MSELoss. Error: {e}")
            criterion = nn.MSELoss()
            print("Using standard MSELoss (fallback)")
    else:
        criterion = nn.MSELoss()
        print("Using standard MSELoss")

    # Optimizer
    if config.get('optimizer', 'adam').lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    else:
        raise ValueError(f"Unknown optimizer: {config.get('optimizer')}")

    # Scheduler
    scheduler_config = config.get('scheduler', {})
    if scheduler_config.get('type') == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5)
        )
    else:
        scheduler = None

    # Early stopping
    early_stop_config = config.get('early_stopping', {})
    early_stop_enabled = early_stop_config.get('enabled', False)
    early_stop_patience = early_stop_config.get('patience', 10)
    early_stop_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    epochs = config.get('epochs', 50)

    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for noisy, clean in pbar:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            # Handle MECGE vs standard models
            if is_mecge:
                # MECGE computes loss internally
                loss = model(clean, noisy)
            else:
                # Standard models need external loss computation
                output = model(noisy)
                loss = criterion(output, clean)

            loss.backward()

            # Apply gradient clipping for v2 model only
            if is_mamba_stft_unet_v2:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)

                # Handle MECGE vs standard models
                if is_mecge:
                    # MECGE computes loss internally
                    loss = model(clean, noisy)
                else:
                    # Standard models need external loss computation
                    output = model(noisy)
                    loss = criterion(output, clean)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Update scheduler
        if scheduler:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['lr'].append(float(current_lr))

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_enabled and early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load best model
    model.load_state_dict(torch.load(model_path))

    return history


def predict_with_model(model: nn.Module, test_loader: DataLoader,
                      device: torch.device) -> np.ndarray:
    """
    Generate predictions with trained model.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Predictions array
    """
    model.eval()
    model = model.to(device)

    # Detect if model is MECGE (has denoising method)
    is_mecge = hasattr(model, 'denoising')
    if is_mecge:
        print("Using MECGE denoising method for inference")
    else:
        print("Using standard forward pass for inference")

    predictions = []

    with torch.no_grad():
        for noisy, _ in tqdm(test_loader, desc="Predicting"):
            noisy = noisy.to(device)

            # Handle MECGE vs standard models
            if is_mecge:
                # MECGE uses dedicated denoising method
                output = model.denoising(noisy)
            else:
                # Standard models use forward pass
                output = model(noisy)

            predictions.append(output.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    # Remove all singleton dimensions (handles both 3D and 4D outputs)
    predictions = np.squeeze(predictions)
    # Ensure result is 2D (batch, time)
    if predictions.ndim > 2:
        predictions = predictions.reshape(predictions.shape[0], -1)

    return predictions
