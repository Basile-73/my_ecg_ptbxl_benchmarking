from ecg_noise_factory.noise import NoiseFactory
import numpy as np
import torch
import random

nf = NoiseFactory(
    'noise/data',
    100,
    'noise/configs/default.yaml',
    'train',
    seed=42
)

clean = np.ones([5,1,1000])

noisy_1 = nf.add_noise(
    clean, 0, 1, 2
)

noisy_2 = nf.add_noise(
    clean, 0, 1, 2
)

print(np.array_equal(noisy_1, noisy_2))

nf = NoiseFactory(
    'noise/data',
    100,
    'noise/configs/default.yaml',
    'train',
    seed=42
)

noisy_1_1 = nf.add_noise(
    clean, 0, 1, 2
)

print(np.array_equal(noisy_1, noisy_1_1))

simulation_params = {
    "duration": 10,
    "sampling_rate": 100,
    "heart_rate": [60, 80],
    "heart_rate_std": 5,
    "lfhfratio": 0.001,
    "means_ai": [1.2, -5, 30, -7.5, 0.75],
    "stds_ai": [0.6, 0.2, 0.0, 1, 0.35],
    "means_bi": [0.25, 0.1, 0.1, 0.1, 0.4],
    "stds_bi": [0.1, 0.1, 0.0, 0.0, 0.0],
}


from dataset import SyntheticEcgDataset

nf_1 = NoiseFactory(
    'noise/data',
    100,
    'noise/configs/default.yaml',
    'train',
    seed=42
)

nf_2 = NoiseFactory(
    'noise/data',
    100,
    'noise/configs/default.yaml',
    'train',
    seed=42
)

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# model 1 epoch 2
dataset_1 = SyntheticEcgDataset(
    simulation_params=simulation_params,
    n_samples=3,
    noise_factory=nf_1,
    save_clean_samples=True
)

noisy_1 , clean_1 = dataset_1[0]

# model 2 epoch 2
dataset_2 = SyntheticEcgDataset(
    simulation_params=simulation_params,
    n_samples=3,
    noise_factory=nf_2,
)
noisy_2 , clean_2 = dataset_2[0]


# are they the same?
torch.equal(clean_1, clean_2) # retuns true
torch.equal(noisy_1, noisy_2) # retuns false
