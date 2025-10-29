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
    criterion = nn.MSELoss()

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
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
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

    predictions = []

    with torch.no_grad():
        for noisy, _ in tqdm(test_loader, desc="Predicting"):
            noisy = noisy.to(device)
            output = model(noisy)
            predictions.append(output.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    predictions = predictions.squeeze(1).squeeze(1)  # Remove channel dims

    return predictions
