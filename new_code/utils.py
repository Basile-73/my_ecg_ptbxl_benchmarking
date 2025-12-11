from torch import nn
from typing import Iterable
from torch.nn import Parameter
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import yaml
import numpy as np
from scipy.signal import butter, filtfilt
from models.Armos.length_wrapper import AnyLengthWrapper

def nested_get(d, path):
    for p in path.split("."):
        d = d[p]
    return d


def get_model(model_type: str, **kwargs):
    if model_type == "imunet":
        from models.IMUnet.Stage1_IMUnet import IMUnet
        sequence_length = kwargs.get('sequence_length')
        return IMUnet(input_length=sequence_length)
    elif model_type == "imunet_mamba":
        from models.IMUnet.Stage1_IMUnet_Mamba import IMUnet
        sequence_length = kwargs.get('sequence_length')
        return IMUnet(input_length=sequence_length)
    elif model_type == "unet":
        from models.UNet.Stage1_UNet import UNet
        sequence_length = kwargs.get('sequence_length')
        return UNet(input_length=sequence_length)
    elif model_type == "unet_mamba":
        from models.UNet.Stage1_UNet_Mamba import UNet
        sequence_length = kwargs.get('sequence_length')
        return UNet(input_length=sequence_length)
    elif model_type == "unet_mamba_bidir":
        from models.UNet.Stage1_UNet_Mamba import UNet
        sequence_length = kwargs.get('sequence_length')
        return UNet(input_length=sequence_length, bidirectional=True)
    elif model_type == "mecge":
        from models.MECGE.MECGE import MECGE
        with open('models/MECGE/config/MECGE_phase.yaml') as f:
            mecge_config = yaml.safe_load(f)
        return MECGE(mecge_config)
    elif model_type == "drnet":
        from models.DRnet.Stage2_model3 import DRnet
        return DRnet()
    elif model_type == "arsene_cnn":
        from models.Armos.arsene_models import CNN_Denoising
        sequence_length = kwargs.get('sequence_length')
        return CNN_Denoising(squence_length=sequence_length)
    elif model_type == "chiang_dae":
        from models.Armos.chiang_dae import FCN_DAE
        return AnyLengthWrapper(FCN_DAE(), factor=32)
    elif model_type == "fotiadou_unet":
        from models.Armos.fotiadou_unet import CNN_encoder_decoder
        return AnyLengthWrapper(CNN_encoder_decoder(), factor=256)
    elif model_type == "ant_drnn":
        from models.Armos.antczak_drrn import DRRN_Denoising
        return DRRN_Denoising()
    else:
        print(f"Model ({model_type}) not found")


def get_loss_function(loss_name: str, **kwargs) -> Module:
    if loss_name == "MSE":
        return nn.MSELoss()
    elif loss_name == "STFT":
        from losses.stft_loss_v2 import EnhancedSTFTLoss
        return EnhancedSTFTLoss()
    else:
        raise ValueError(f"Loss function ({loss_name}) not found")


def get_optimizer(
    optimizer_name: str, model_parameters: Iterable[Parameter], **kwargs
) -> Optimizer:
    if optimizer_name == "Adam":
        return torch.optim.Adam(model_parameters)


def get_scheduler(scheduler_name: str, optimizer: Optimizer, **kwargs) -> _LRScheduler:
    if scheduler_name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer)


def read_config(config_path: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_type = config["model"]["type"]
    model_name = config["model"]["name"]
    simulation_params = config["simulation_params"]
    data_volume = config["data_volume"]
    noise_paths = config["noise_paths"]
    training_config = config["training"]
    return model_type, model_name, simulation_params, data_volume, noise_paths, training_config

def get_sampleset_name(params, n, mode):
    keys = {"means_ai","stds_ai","means_bi","stds_bi"} # keys & values to exclude from file name
    filtered = {k:v for k,v in params.items() if k not in keys}
    parts = [f"{k}_{filtered[k]}" for k in sorted(filtered)]
    return "_".join(parts + [f"n_samples_{n}", f"mode_{mode}"])

def bandpass_filter(data: np.ndarray, fs:int, lowcut: float = 1.0, highcut: float = 45.0,
                    order: int = 2) -> np.ndarray:
    """
    Apply 2nd-order Butterworth bandpass filter to remove baseline wander.
    Following Dias et al. 2024: 1-45 Hz bandpass.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    filtered = np.array([filtfilt(b, a, record) for record in data])
    return filtered

def get_percentiles(a:list, n:int =1000):
    bs = [np.mean(np.random.choice(a, len(a), True)) for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])
