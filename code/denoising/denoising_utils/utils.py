"""
Utility functions for ECG denoising experiments.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import sys
import os


# ============================================================================
# Metrics
# ============================================================================

def calculate_snr(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio in dB."""
    noise = noisy_signal - clean_signal
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return float('inf')

    return 10 * np.log10(signal_power / noise_power)


def calculate_rmse(clean_signal: np.ndarray, denoised_signal: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((clean_signal - denoised_signal) ** 2))


# ============================================================================
# PyTorch Dataset
# ============================================================================

class ECGDenoisingDataset(Dataset):
    """PyTorch Dataset for ECG denoising task."""

    def __init__(self, noisy_signals: np.ndarray, clean_signals: np.ndarray):
        self.noisy_signals = noisy_signals
        self.clean_signals = clean_signals

    def __len__(self) -> int:
        return len(self.noisy_signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy = self.noisy_signals[idx]
        clean = self.clean_signals[idx]

        # Convert to torch tensors with shape (1, 1, time) for 2D convolutions
        noisy = torch.FloatTensor(noisy).permute(2, 0, 1).unsqueeze(0)
        clean = torch.FloatTensor(clean).permute(2, 0, 1).unsqueeze(0)

        return noisy, clean


class OnlineNoisingDataset(Dataset):
    """PyTorch Dataset with online noise generation for each epoch."""

    def __init__(self, clean_signals: np.ndarray, noise_factory):
        """
        Args:
            clean_signals: Clean ECG signals (N, time, channels)
            noise_factory: NoiseFactory instance for adding noise
        """
        self.clean_signals = clean_signals
        self.noise_factory = noise_factory

    def __len__(self) -> int:
        return len(self.clean_signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.clean_signals[idx:idx+1]  # Shape: (1, time, channels)

        # Add noise online (generates new noise each time)
        # NoiseFactory expects (batch, length, channel) based on axis specification
        noisy = self.noise_factory.add_noise(
            x=clean, batch_axis=0, channel_axis=2, length_axis=1
        )[0]  # Remove batch dimension -> Shape: (time, channels)

        # Convert to torch tensors
        # Input shape: (time, channels) -> Output shape: (1, channels, time)
        noisy = torch.FloatTensor(noisy).permute(1, 0).unsqueeze(0)  # (time, ch) -> (ch, time) -> (1, ch, time)
        clean = torch.FloatTensor(clean[0]).permute(1, 0).unsqueeze(0)  # Same transformation

        return noisy, clean


class Stage2OnlineNoisingDataset(Dataset):
    """Dataset for Stage2 training with online noise generation and Stage1 predictions."""

    def __init__(self, clean_signals: np.ndarray, noise_factory, stage1_model, device):
        """
        Args:
            clean_signals: Clean ECG signals (N, time, channels)
            noise_factory: NoiseFactory instance for adding noise
            stage1_model: Trained Stage1 model (frozen)
            device: torch device
        """
        self.clean_signals = clean_signals
        self.noise_factory = noise_factory
        self.stage1_model = stage1_model
        self.device = device

        # Freeze stage1 model
        self.stage1_model.eval()
        for param in self.stage1_model.parameters():
            param.requires_grad = False

    def __len__(self) -> int:
        return len(self.clean_signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.clean_signals[idx:idx+1]  # Shape: (1, time, channels)

        # Add noise online
        noisy = self.noise_factory.add_noise(
            x=clean, batch_axis=0, channel_axis=2, length_axis=1
        )[0]  # Remove batch dimension -> Shape: (time, channels)

        # Convert to torch tensors
        noisy_tensor = torch.FloatTensor(noisy).permute(1, 0).unsqueeze(0).unsqueeze(0)  # (1, 1, ch, time)
        clean_tensor = torch.FloatTensor(clean[0]).permute(1, 0).unsqueeze(0)  # (1, ch, time)

        # Get Stage1 prediction (no gradients)
        with torch.no_grad():
            stage1_output = self.stage1_model(noisy_tensor.to(self.device)).cpu()  # (1, 1, ch, time)

        # Concatenate noisy and stage1 output as 2-channel input for Stage2
        # Shape: (1, 2, ch, time) where channel dim has [noisy, stage1_pred]
        stage2_input = torch.cat([noisy_tensor, stage1_output], dim=1).squeeze(0)  # (2, ch, time)

        return stage2_input, clean_tensor
def create_dataloaders(noisy_train, clean_train, noisy_val, clean_val,
                      noisy_test, clean_test, batch_size=32, num_workers=4,
                      pin_memory=True):
    """Create train, validation, and test DataLoaders."""
    train_dataset = ECGDenoisingDataset(noisy_train, clean_train)
    val_dataset = ECGDenoisingDataset(noisy_val, clean_val)
    test_dataset = ECGDenoisingDataset(noisy_test, clean_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


def create_online_dataloaders(clean_train, clean_val, clean_test,
                              noise_factory_train, noise_factory_test,
                              batch_size=32, num_workers=0,
                              pin_memory=True):
    """Create dataloaders with online noise generation for training.

    Args:
        clean_train, clean_val, clean_test: Clean ECG data splits
        noise_factory_train: NoiseFactory with mode='train' for training data
        noise_factory_test: NoiseFactory with mode='test' for val/test data
        batch_size, num_workers, pin_memory: DataLoader parameters
    """
    train_dataset = OnlineNoisingDataset(clean_train, noise_factory_train)
    val_dataset = OnlineNoisingDataset(clean_val, noise_factory_test)
    test_dataset = OnlineNoisingDataset(clean_test, noise_factory_test)    # Note: num_workers=0 for online noising to avoid pickling issues with NoiseFactory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


def create_stage2_dataloaders(clean_train, clean_val, clean_test,
                               noise_factory_train, noise_factory_test,
                               stage1_model, device,
                               batch_size=32, pin_memory=True):
    """Create dataloaders for Stage2 training with Stage1 predictions.

    Args:
        clean_train, clean_val, clean_test: Clean ECG data splits
        noise_factory_train: NoiseFactory with mode='train' for training data
        noise_factory_test: NoiseFactory with mode='test' for val/test data
        stage1_model: Trained Stage1 model (frozen)
        device: torch device
        batch_size, pin_memory: DataLoader parameters
    """
    train_dataset = Stage2OnlineNoisingDataset(clean_train, noise_factory_train, stage1_model, device)
    val_dataset = Stage2OnlineNoisingDataset(clean_val, noise_factory_test, stage1_model, device)
    test_dataset = Stage2OnlineNoisingDataset(clean_test, noise_factory_test, stage1_model, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
# ============================================================================
# Model Loading
# ============================================================================

class DenoisingModelWrapper(nn.Module):
    """Wrapper to adapt models for different signal lengths."""

    def __init__(self, base_model: nn.Module, input_length: int = 5000):
        super().__init__()
        self.base_model = base_model
        self.input_length = input_length
        self.target_length = 3600

    def forward(self, x):
        batch_size = x.shape[0]

        if self.input_length != self.target_length:
            x = torch.nn.functional.interpolate(
                x.squeeze(2), size=self.target_length, mode='linear', align_corners=False
            ).unsqueeze(2)

        output = self.base_model(x)

        if self.input_length != self.target_length:
            output = torch.nn.functional.interpolate(
                output.squeeze(2), size=self.input_length, mode='linear', align_corners=False
            ).unsqueeze(2)

        return output


def get_model(model_type: str, input_length: int = 5000,
              pretrained_path: Optional[str] = None, is_stage2: bool = False) -> nn.Module:
    """Get model by name."""
    # Add ECG-processing folder to path (it's in the parent denoising directory)
    ecg_processing_path = os.path.join(os.path.dirname(__file__), '../ECG-processing')
    sys.path.insert(0, ecg_processing_path)

    model_type = model_type.lower()

    if model_type == 'fcn':
        from Stage1_FCN import FCN
        base_model = FCN(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'imunet':
        from Stage1_IMUnet import IMUnet
        base_model = IMUnet(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'unet':
        from Stage1_Unet import UNet
        base_model = UNet(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'stage2' or model_type == 'drnet':
        from Stage2_model3 import DRnet
        base_model = DRnet(in_channels=2)  # 2 channels: noisy + stage1 output
        model = DenoisingModelWrapper(base_model, input_length)
    else:
        raise ValueError(f"Unknown model name: {model_type}")

    if pretrained_path and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded {model_type} with {n_params:,} parameters")

    return model
