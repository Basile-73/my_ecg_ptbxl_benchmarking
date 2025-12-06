import neurokit2 as nk
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from ecg_noise_factory.noise import NoiseFactory
from typing import Optional
import os
import torch
from utils import get_sampleset_name
from utils import bandpass_filter


class SyntheticEcgDataset(Dataset):
    def __init__(
        self,
        simulation_params: dict,
        n_samples: int,
        noise_factory: NoiseFactory,
        median: Optional[float] = None,
        iqr: Optional[float] = None,
        save_clean_samples: bool = False,
    ):
        assert simulation_params["sampling_rate"] == noise_factory.sampling_rate
        assert not (
            noise_factory.mode != "train" and (median is None or iqr is None)
        ), "scaler required when mode != train"

        self.simulation_params = simulation_params
        self.n_samples = n_samples
        self.noise_factory = noise_factory
        self.median = median
        self.iqr = iqr
        self.sample_set_name = get_sampleset_name(
            simulation_params, n_samples, noise_factory.mode
        )
        self.save_clean_samples = save_clean_samples
        self.samples = None

        already_generated = os.path.exists(f"data/{self.sample_set_name}")
        if already_generated:
            print(f"Loading existing data from: data/{self.sample_set_name}")
            self.samples = np.loadtxt(f"data/{self.sample_set_name}", delimiter=",")
        else:
            self.samples = self._generate_samples()
            if self.save_clean_samples:
                np.savetxt(f"data/{self.sample_set_name}", self.samples, delimiter=",")


        if self.median is None or self.iqr is None:
            self._get_scaler_stats()
            scaler_stats = np.array([self.median, self.iqr])
            np.savetxt(f"data/{self.sample_set_name}_scaler_stats", scaler_stats, delimiter=",")


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        clean_not_normalized = self.samples[idx, :]
        clean = (clean_not_normalized - self.median) / self.iqr
        noisy = self._add_noise_to_single_record(clean)
        clean = torch.tensor(clean, dtype=torch.float32)
        noisy = torch.tensor(noisy, dtype=torch.float32)
        return (
            noisy.unsqueeze(0).unsqueeze(0), # X
            clean.unsqueeze(0).unsqueeze(0), # y
        )  # [channel = 1, height = 1, length = self.simulation_params[duration] * self.simulation_params[sampling_rate]]


    def _generate_samples(self):
        ecg_list = []
        for i in tqdm(range(self.n_samples), desc="Generating sythetic samples"):
            ecg = self._generate_single_sample(i)
            ecg_list.append(ecg)
        filtered = bandpass_filter(np.array(ecg_list), self.simulation_params["sampling_rate"])
        return filtered

    def _generate_single_sample(self, i):
        mode = self.noise_factory.mode
        base = {"train": 0, "test": 1_000_000, "eval": 2_000_000}.get(self.noise_factory.mode, 0)
        seed = base + i

        cutoff_duration = 5
        cutoff = self.simulation_params["sampling_rate"] * cutoff_duration

        effective_simulation_params = self.simulation_params.copy()
        effective_simulation_params['duration'] += cutoff_duration
        effective_simulation_params['random_state'] = seed

        ecg = nk.ecg_simulate(**effective_simulation_params, noise = 0)
        ecg = ecg[cutoff:]
        return ecg


    def _add_noise_to_single_record(self, clean):
        clean_3d = clean.reshape(1, 1, -1)
        noisy_3d = self.noise_factory.add_noise(clean_3d, 0, 1, 2)
        noisy = noisy_3d.reshape(-1)
        return noisy

    def _get_scaler_stats(self):
        vals = self.samples.ravel()
        self.median = np.median(vals)
        self.iqr = np.percentile(vals, 75) - np.percentile(vals, 25)


# # Example Usage
# sim_params = {
#     "duration": 10,
#     "sampling_rate": 360,
#     "heart_rate": 70,
#     "heart_rate_std": 5,
#     "lfhfratio": 0.001
# }

# n_samples = 5

# train_factory = NoiseFactory(
#     data_path="noise/data",
#     sampling_rate=360,
#     config_path="noise/configs/synthetic.yaml",
#     mode="train",
#     seed=42,
# )
# train_set = SyntheticEcgDataset(
#     sim_params, n_samples, train_factory, save_clean_samples=False
# )

# example_i = 3

# import matplotlib.pyplot as plt
# clean, noisy = train_set[example_i]
# clean_np = clean.reshape(-1)
# t = np.arange(len(clean_np)) / train_set.simulation_params["sampling_rate"]

# width = train_set.simulation_params["duration"]
# plt.figure(figsize=(width, 3))
# plt.plot(t, clean_np, color="green", label="ground truth")
