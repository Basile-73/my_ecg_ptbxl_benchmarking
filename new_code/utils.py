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


def get_model(model_name: str, **kwargs):
    if model_name == "imunet":
        from models.Stage1_IMUnet import IMUnet

        return IMUnet()
    else:
        print(f"Model ({model_name}) not found")


def get_loss_function(loss_name: str, **kwargs) -> Module:
    if loss_name == "MSE":
        return nn.MSELoss()


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

    model_name = config["model"]
    simulation_params = config["simulation_params"]
    data_volume = config["data_volume"]
    noise_paths = config["noise_paths"]
    training_config = config["training"]
    return model_name, simulation_params, data_volume, noise_paths, training_config

def get_sampleset_name(simulation_params:dict, n_samples:int, mode:str)-> str:
    s = "_".join(
        f"{k}_{simulation_params[k]}" for k in sorted(simulation_params)
    )
    s = f"{s}_n_samples_{n_samples}"
    s = f"{s}_mode_{mode}"
    return s

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
