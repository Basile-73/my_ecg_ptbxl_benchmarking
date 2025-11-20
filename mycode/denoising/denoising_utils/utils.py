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
import yaml
import importlib.util


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


def run_denoise_inference(model: nn.Module, input_tensor: torch.Tensor,
                          is_stage2: bool = False) -> np.ndarray:
    """
    Run model inference with automatic detection of MECGE vs standard models.

    Args:
        model: The denoising model (MECGE or standard)
        input_tensor: Input tensor on appropriate device
        is_stage2: Whether this is a Stage2 model (with 2-channel input)

    Returns:
        Numpy array of predictions with batch and channel dims squeezed

    Note:
        Stage2 MECGE models are not supported and will fall back to standard forward pass.
    """
    is_mecge = hasattr(model, 'denoising')

    # Stage2 MECGE path is unsupported (would pass invalid shape B,2,1,T)
    if is_stage2 and is_mecge:
        # Fallback to standard forward pass for Stage2 MECGE
        pred = model(input_tensor)
    elif is_mecge:
        # Use MECGE's dedicated inference method for Stage1
        pred = model.denoising(input_tensor)
    else:
        # Standard model forward pass
        pred = model(input_tensor)

    return pred.squeeze().cpu().numpy()


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

        # Convert to torch tensors with shape (1, channels, time) for 2D convolutions
        # Handle 2D inputs: (time, channels) -> (channels, time) -> (1, channels, time)
        noisy = torch.FloatTensor(noisy).permute(1, 0).unsqueeze(0)
        clean = torch.FloatTensor(clean).permute(1, 0).unsqueeze(0)

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
    # Add ECG-processing folder to path (it's in denoising_models/ECG-processing)
    ecg_processing_path = os.path.join(os.path.dirname(__file__), '../denoising_models/ECG-processing')
    sys.path.insert(0, ecg_processing_path)

    # Add my_MECG-E folder to path (it's in denoising_models/my_MECG-E)
    mecge_path = os.path.join(os.path.dirname(__file__), '../denoising_models/my_MECG-E')
    sys.path.insert(0, mecge_path)

    # Add mamba_stft_unet folder to path (it's in denoising_models/mamba_stft_unet)
    mamba_stft_unet_path = os.path.join(os.path.dirname(__file__), '../denoising_models/mamba_stft_unet')
    sys.path.insert(0, mamba_stft_unet_path)

    model_type = model_type.lower()

    if model_type == 'fcn':
        from Stage1_FCN import FCN
        base_model = FCN(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'imunet':
        from Stage1_IMUnet import IMUnet
        base_model = IMUnet(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'imunet_varlen':
        # Variable-length IMUnet with native variable-length support (no wrapper needed)
        # This model dynamically calculates upsample sizes based on input_length,
        # eliminating the need for interpolation and improving efficiency.
        from Stage1_IMUnet_varlen import IMUnet
        model = IMUnet(in_channels=1, input_length=input_length)
        # No DenoisingModelWrapper needed - model handles variable lengths natively
        print(f"  Model: imunet_varlen with native variable-length support (input_length={input_length})")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")
    elif model_type == 'imunet_mamba_varlen':
        # Stage1_2 IMUnet with Mamba-enhanced bottleneck and native variable-length support
        # Uses MambaMerge for context fusion instead of simple 1x1 convolution
        # No interpolation wrapper needed - model handles variable lengths natively
        try:
            from Stage1_2_IMUnet_mamba_merge_bn_big_varlen import IMUnet
            model = IMUnet(in_channels=1, input_length=input_length)
            # No DenoisingModelWrapper needed - model handles variable lengths natively
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Loaded IMUnet_Mamba_varlen with {n_params:,} parameters for input_length={input_length}")
        except ImportError as e:
            raise ImportError(
                "Stage1_2_IMUnet_mamba_merge_bn_big_varlen requires mamba-ssm. "
                "Install with: pip install mamba-ssm"
            ) from e
    elif model_type == 'imunet_mamba_varlen_upconv':
        from Stage1_2_IMUnet_mamba_merge_bn_big_varlen_upconv import IMUnet
        model = IMUnet(in_channels=1, input_length=input_length)
        # No DenoisingModelWrapper needed - model handles variable lengths natively
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Loaded IMUnet_Mamba_varlen_upconv with {n_params:,} parameters for input_length={input_length}")
    elif model_type == 'imunet_early_mamba_varlen':
        # Stage1_4 IMUnet with early-stage Mamba and native variable-length support
        # Uses MambaEarlyLayer in first encoder block to capture global temporal dependencies
        # in raw signals before downsampling. Processes full-length sequences (e.g., 3600 samples)
        # at the earliest stage, unlike Stage1_2 which uses Mamba in the compressed bottleneck.
        # No interpolation wrapper needed - model handles variable lengths natively.
        try:
            from Stage1_4_IMUnet_mamba_merge_early_big_varlen import IMUnet
            model = IMUnet(in_channels=1, input_length=input_length)
            # No DenoisingModelWrapper needed - model handles variable lengths natively
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Loaded IMUnet_EarlyMamba_varlen with {n_params:,} parameters for input_length={input_length}")
        except ImportError as e:
            raise ImportError(
                "Stage1_4_IMUnet_mamba_merge_early_big_varlen requires mamba-ssm and einops. "
                "Install with: pip install mamba-ssm einops"
            ) from e
    elif model_type == 'imunet_early_mamba_varlen_upconv':
        from Stage1_4_IMUnet_mamba_merge_early_big_varlen_upconv import IMUnet
        model = IMUnet(in_channels=1, input_length=input_length)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Loaded IMUnet_EarlyMamba_varlen_upconv with {n_params:,} parameters for input_length={input_length}")
    elif model_type == 'unet':
        from Stage1_Unet import UNet
        base_model = UNet(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'imunet_origin':
        stage_path = os.path.join(ecg_processing_path, "Stage1_1_IMUnet_origin.py")
        spec = importlib.util.spec_from_file_location("stage_imunet_origin", stage_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        IMUnet = module.IMUnet
        base_model = IMUnet(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'imunet_mamba_bn':
        stage_path = os.path.join(ecg_processing_path, "Stage1_2_IMUnet_mamba_merge_bn.py")
        spec = importlib.util.spec_from_file_location("stage_imunet_mamba_bn", stage_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        IMUnet = module.IMUnet
        base_model = IMUnet(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'imunet_mamba_bn_big':
        stage_path = os.path.join(ecg_processing_path, "Stage1_2_IMUnet_mamba_merge_bn_big.py")
        spec = importlib.util.spec_from_file_location("stage_imunet_mamba_bn", stage_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        IMUnet = module.IMUnet
        base_model = IMUnet(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'imunet_mamba_up':
        stage_path = os.path.join(ecg_processing_path, "Stage1_3_IMUnet_mamba_merge_up.py")
        spec = importlib.util.spec_from_file_location("stage_imunet_mamba_up", stage_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        IMUnet = module.IMUnet
        base_model = IMUnet(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'imunet_mamba_early':
        stage_path = os.path.join(ecg_processing_path, "Stage1_4_IMUnet_mamba_merge_early.py")
        spec = importlib.util.spec_from_file_location("stage_imunet_mamba_early", stage_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        IMUnet = module.IMUnet
        base_model = IMUnet(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'imunet_mamba_early_big':
        stage_path = os.path.join(ecg_processing_path, "Stage1_4_IMUnet_mamba_merge_early_big.py")
        spec = importlib.util.spec_from_file_location("stage_imunet_mamba_early", stage_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        IMUnet = module.IMUnet
        base_model = IMUnet(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'imunet_mamba_late':
        stage_path = os.path.join(ecg_processing_path, "Stage1_5_IMUnet_mamba_merge_late.py")
        spec = importlib.util.spec_from_file_location("stage_imunet_mamba_late", stage_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        IMUnet = module.IMUnet
        base_model = IMUnet(in_channels=1)
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'mamba_stft_unet':
        from model import TinyMambaSTFTUNet
        base_model = TinyMambaSTFTUNet()
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'mamba_stft_unet_v2':
        from model_v2 import TinyMambaSTFTUNetV2
        base_model = TinyMambaSTFTUNetV2()
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'stage2' or model_type == 'drnet':
        from Stage2_model3 import DRnet
        base_model = DRnet(in_channels=2)  # 2 channels: noisy + stage1 output
        model = DenoisingModelWrapper(base_model, input_length)
    elif model_type == 'mecge_phase':
        # Load MECGE with phase feature configuration
        config_path = os.path.join(os.path.dirname(__file__), '../denoising_models/my_MECG-E/config/MECGE_phase.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        from models.MECGE import MECGE
        model = MECGE(config)

        if pretrained_path and os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, weights_only=True))

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Loaded {model_type} with {n_params:,} parameters")

        return model
    elif model_type == 'mecge_complex':
        # Load MECGE with complex feature configuration
        config_path = os.path.join(os.path.dirname(__file__), '../denoising_models/my_MECG-E/config/MECGE_complex.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        from models.MECGE import MECGE
        model = MECGE(config)

        if pretrained_path and os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, weights_only=True))

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Loaded {model_type} with {n_params:,} parameters")

        return model
    elif model_type == 'mecge_wav':
        # Load MECGE with waveform feature configuration
        config_path = os.path.join(os.path.dirname(__file__), '../denoising_models/my_MECG-E/config/MECGE_wav.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        from models.MECGE import MECGE
        model = MECGE(config)

        if pretrained_path and os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, weights_only=True))

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Loaded {model_type} with {n_params:,} parameters")

        return model

    # MECGE_varlen models: Identical to MECGE but explicitly marked as variable-length capable
    # for documentation purposes. MECGE is natively variable-length through its STFT-based
    # architecture (torch.stft/istft with config-driven parameters). No code modifications
    # were needed - this is a copy of MECGE.py to document native variable-length support.

    elif model_type == 'mecge_phase_varlen':
        # Load MECGE_varlen with phase feature configuration
        # Natively variable-length via STFT (no modifications needed)
        config_path = os.path.join(os.path.dirname(__file__), '../denoising_models/my_MECG-E/config/MECGE_phase.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        try:
            from models.MECGE_varlen import MECGE
            model = MECGE(config)

            if pretrained_path and os.path.exists(pretrained_path):
                model.load_state_dict(torch.load(pretrained_path, weights_only=True))

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Loaded mecge_phase_varlen with native variable-length support (STFT-based, no modifications needed)")
            print(f"  Parameters: {n_params:,}")
            print(f"  STFT config: n_fft={config['model']['n_fft']}, hop_size={config['model']['hop_size']}, win_size={config['model']['win_size']}")

            return model
        except ImportError as e:
            raise ImportError(
                "MECGE_varlen requires mamba-ssm and einops. "
                "Install with: pip install mamba-ssm einops"
            ) from e

    elif model_type == 'mecge_complex_varlen':
        # Load MECGE_varlen with complex feature configuration
        # Natively variable-length via STFT (no modifications needed)
        config_path = os.path.join(os.path.dirname(__file__), '../denoising_models/my_MECG-E/config/MECGE_complex.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        try:
            from models.MECGE_varlen import MECGE
            model = MECGE(config)

            if pretrained_path and os.path.exists(pretrained_path):
                model.load_state_dict(torch.load(pretrained_path, weights_only=True))

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Loaded mecge_complex_varlen with native variable-length support (STFT-based, no modifications needed)")
            print(f"  Parameters: {n_params:,}")
            print(f"  STFT config: n_fft={config['model']['n_fft']}, hop_size={config['model']['hop_size']}, win_size={config['model']['win_size']}")

            return model
        except ImportError as e:
            raise ImportError(
                "MECGE_varlen requires mamba-ssm and einops. "
                "Install with: pip install mamba-ssm einops"
            ) from e

    elif model_type == 'mecge_wav_varlen':
        # Load MECGE_varlen with waveform feature configuration
        # Natively variable-length via STFT (no modifications needed)
        config_path = os.path.join(os.path.dirname(__file__), '../denoising_models/my_MECG-E/config/MECGE_wav.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        try:
            from models.MECGE_varlen import MECGE
            model = MECGE(config)

            if pretrained_path and os.path.exists(pretrained_path):
                model.load_state_dict(torch.load(pretrained_path, weights_only=True))

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Loaded mecge_wav_varlen with native variable-length support (STFT-based, no modifications needed)")
            print(f"  Parameters: {n_params:,}")
            print(f"  STFT config: n_fft={config['model']['n_fft']}, hop_size={config['model']['hop_size']}, win_size={config['model']['win_size']}")

            return model
        except ImportError as e:
            raise ImportError(
                "MECGE_varlen requires mamba-ssm and einops. "
                "Install with: pip install mamba-ssm einops"
            ) from e

    else:
        raise ValueError(f"Unknown model name: {model_type}")

    if pretrained_path and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, weights_only=True))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded {model_type} with {n_params:,} parameters")

    return model
