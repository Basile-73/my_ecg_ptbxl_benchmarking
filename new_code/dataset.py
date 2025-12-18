import neurokit2 as nk
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from ecg_noise_factory.noise import NoiseFactory
from typing import Optional
import os
import torch
from utils.getters import get_sampleset_name, get_sampleset_name_mitbh_arr
from utils.getters import bandpass_filter
import time
import glob
import wfdb


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
        np.random.seed(42)
        assert simulation_params["sampling_rate"] == noise_factory.sampling_rate
        assert not (
            noise_factory.mode != "train" and (median is None or iqr is None)
        ), "scaler required when mode != train"

        self.simulation_params = simulation_params
        self.n_samples = n_samples
        self.noise_factory = noise_factory
        self.median = median
        self.iqr = iqr
        self.sample_set_name = self._get_sampleset_name()
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

    def _get_sampleset_name(self):
        return get_sampleset_name(self.simulation_params, self.n_samples, self.noise_factory.mode)

    def _generate_samples(self):
        ecg_list = []
        for i in tqdm(range(self.n_samples), desc="Generating sythetic samples"):
            ecg = self._generate_single_sample(i)
            ecg_list.append(ecg)
        filtered = bandpass_filter(np.array(ecg_list), self.simulation_params["sampling_rate"])
        return filtered

    def _get_effective_params_and_cutoff(self, i):
        effective_simulation_params = self.simulation_params.copy()
        for k in ["means_ai", "stds_ai", "means_bi", "stds_bi"]:
            effective_simulation_params.pop(k, None)

        hr_interval = self.simulation_params["heart_rate"]
        effective_simulation_params["heart_rate"] = np.random.randint(low=hr_interval[0], high=hr_interval[1])

        means_ai = np.array(self.simulation_params["means_ai"])
        stds_ai = np.array(self.simulation_params["stds_ai"])
        effective_ai = np.random.uniform(means_ai-stds_ai, means_ai+stds_ai)
        effective_simulation_params["ai"] = effective_ai

        means_bi = np.array(self.simulation_params["means_bi"])
        stds_bi = np.array(self.simulation_params["stds_bi"])
        effective_bi = np.random.uniform(means_bi-stds_bi, means_bi+stds_bi)
        effective_simulation_params["bi"] = effective_bi

        base = {"train": 0, "test": 1_000_000, "eval": 2_000_000}.get(self.noise_factory.mode, 0)
        effective_seed = base + i
        effective_simulation_params["random_state"] = effective_seed

        cutoff_duration = 5
        cutoff = self.simulation_params["sampling_rate"] * cutoff_duration
        effective_duration = self.simulation_params["duration"] + cutoff_duration
        effective_simulation_params["duration"] = effective_duration
        return effective_simulation_params, cutoff

    def _generate_single_sample(self, i):
        effective_simulation_params, cutoff = self._get_effective_params_and_cutoff(i)
        for _ in range(1000):
            try:
                ecg = nk.ecg_simulate(**effective_simulation_params, noise = 0)
                return ecg[cutoff:]
            except Exception:
                time.sleep(0.001)
        raise RuntimeError("Failed to generate ECG after 1000 retries")

    def _add_noise_to_single_record(self, clean):
        clean_3d = clean.reshape(1, 1, -1)
        noisy_3d = self.noise_factory.add_noise(clean_3d, 0, 1, 2)
        noisy = noisy_3d.reshape(-1)
        return noisy

    def _get_scaler_stats(self):
        vals = self.samples.ravel()
        self.median = np.median(vals)
        self.iqr = np.percentile(vals, 75) - np.percentile(vals, 25)

class LengthExperimentDataset(SyntheticEcgDataset):
    def __init__(
        self,
        simulation_params: dict,
        n_samples: int,
        noise_factory: NoiseFactory,
        split_length: int,
        median: Optional[float] = None,
        iqr: Optional[float] = None,
        save_clean_samples: bool = False,
    ):
        super().__init__(
            simulation_params,
            n_samples,
            noise_factory,
            median,
            iqr,
            save_clean_samples,
        )
        assert self.samples.shape[1] % split_length == 0, f"Signal length {self.samples.shape[1]} not divisible by split_length {split_length}"
        self.split_length = split_length

        if self.samples.shape[1] != split_length:
            self.samples = self.samples.reshape(-1, split_length)
            self.n_samples = self.samples.shape[0]

class MITBihArrDataset(LengthExperimentDataset):
    def __init__(
            self,
            n_samples: int,
            noise_factory: NoiseFactory,
            duration: int,
            split_length: int,
            data_path: str,
            median: Optional[float] = None,
            iqr: Optional[float] = None,
            save_clean_samples: bool = False
    ):
        dummy_simulation_params = {
            "sampling_rate": noise_factory.sampling_rate,
            "duration": duration,
        }
        self.duration = duration
        self.path = data_path

        super().__init__(
            simulation_params=dummy_simulation_params,
            n_samples=n_samples,
            noise_factory=noise_factory,
            split_length=split_length,
            median=median,
            iqr=iqr,
            save_clean_samples=save_clean_samples,
        )

    def _get_sampleset_name(self):
        return get_sampleset_name_mitbh_arr(self.duration, self.n_samples)

    def _generate_samples(self):
        fs = 360
        win = int(self.duration * fs)
        records = sorted({os.path.splitext(f)[0] for f in glob.glob(f"{self.path}/*.dat")})

        total = 0
        info = []
        for r in records:
            sig, _ = wfdb.rdsamp(r)
            w = len(sig) // win
            total += w
            info.append((r, w))

        if self.n_samples > total:
            raise ValueError("Requested more samples than available")

        X, remaining = [], self.n_samples
        for r, w in info:
            sig, _ = wfdb.rdsamp(r)
            sig = sig[:,0]
            take = min(w, remaining)
            for i in range(take):
                X.append(sig[i*win:(i+1)*win])
            remaining -= take
            if remaining == 0:
                break

        X = np.stack(X)
        return bandpass_filter(X, fs)

# Example Usage

n_samples = 5

train_factory = NoiseFactory(
    data_path="noise/data",
    sampling_rate=360,
    config_path="noise/configs/synthetic.yaml",
    mode="train",
    seed=42,
)

train_set = MITBihArrDataset(
    n_samples=n_samples,
    noise_factory=train_factory,
    duration=40,
    split_length=(10*360),
    data_path= 'data/mitdb_arr/physionet.org/files/mitdb/1.0.0/x_mitdb',
    median=None,
    iqr=None,
    save_clean_samples=False
)
train_set.samples.shape

# # Example Usage
# sim_params = {
#     "duration": 10,
#     "sampling_rate": 360,
#     "heart_rate": [60, 80],
#     "heart_rate_std": 5,
#     "lfhfratio": 0.001,
#     "means_ai" : [1.2, -5, 30, -7.5, 0.75],
#     "stds_ai" : [0.6, 0.2, 0.0, 1, 0.35],
#     "means_bi" : [0.25, 0.1, 0.1, 0.1, 0.4],
#     "stds_bi" : [0.1, 0.1, 0.0, 0.0, 0.0],
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

# train_set.samples.shape

# train_set = LengthExperimentDataset(
#     sim_params, n_samples, train_factory, split_length=(5*360), save_clean_samples=False
# )

# retrieve first element and check shape
# noisy, clean = train_set[0]
# noisy.shape, clean.shape

# train_set.samples.shape


import matplotlib.pyplot as plt

for example_i in range(5):
    noisy, clean = train_set[example_i]
    clean_np = clean.reshape(-1)
    t = np.arange(len(clean_np)) / train_set.simulation_params["sampling_rate"]

    width = train_set.simulation_params["duration"]
    plt.figure(figsize=(width, 3))
    plt.plot(t, clean_np, color="green", label="ground truth")
