from pathlib import Path
import torch
from ecg_noise_factory.noise import NoiseFactory
from dataset import SyntheticEcgDataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from functools import cached_property
import pandas as pd
from utils import get_percentiles


from utils import get_model, read_config, get_sampleset_name

class Evaluator:
    def __init__(self, config_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name, simulation_params, data_volume, noise_paths, training_config = (
            read_config(config_path)
        )

        self.simulation_params = simulation_params
        self.model = get_model(model_name)
        self.model
        self.model.load_state_dict(
            torch.load(f"model_weights/best_{model_name}.pth")
        )


        self.eval_noise_factory = NoiseFactory(
            noise_paths["data_path"],
            simulation_params["sampling_rate"],
            noise_paths["config_path"],
            mode="eval",
            seed=42,
        )

        self.train_sample_set_name = get_sampleset_name(
            simulation_params,
            data_volume["n_samples_train"],
            "train"
        )

        scaler_stats = np.loadtxt(f'data/{self.train_sample_set_name}_scaler_stats')

        self.eval_dataset = SyntheticEcgDataset(
            simulation_params,
            data_volume["n_samples_test"],
            self.eval_noise_factory,
            median=scaler_stats[0],
            iqr=scaler_stats[1],
            save_clean_samples=data_volume["save_clean_samples"],
        )

        self.eval_data_loader = DataLoader(self.eval_dataset, training_config["batch_size"])


    def plot_examples(self, idx):
        noisy, clean = self.eval_dataset[idx]

        noisy = noisy.to(self.device)
        self.model = self.model.to(self.device)

        with torch.no_grad():
            denoised = self.model(noisy.unsqueeze(0))
            denoised = denoised.squeeze(0)

        clean = clean.reshape(-1).detach().cpu().numpy()
        noisy = noisy.reshape(-1).detach().cpu().numpy()
        denoised = denoised.reshape(-1).detach().cpu().numpy()


        sampling_frequency = self.simulation_params["sampling_rate"]
        t = np.arange(len(clean)) / sampling_frequency

        width = self.simulation_params["duration"]

        plt.figure(figsize=(width, 3))
        plt.plot(t, noisy, color="grey", label="noisy")
        plt.plot(t, clean, color="green", label="ground truth")
        plt.axhline(0, linestyle=":", color="lightgreen")
        plt.plot(t, denoised, color="blue", label="prediction")
        plt.legend()
        plt.xlabel("Time (s)")
        plt.show()

    @cached_property
    def results(self):
        rmses, snrs = [], []

        self.model.eval()
        for noisy, clean in self.eval_data_loader:
            noisy = noisy.to(self.device)
            denoised = self.model(noisy).detach().cpu().numpy().reshape(len(clean), -1)
            clean = clean.numpy().reshape(len(clean), -1)

            err = clean - denoised
            rmses.extend(np.sqrt((err**2).mean(axis=1)))
            snrs.extend(10*np.log10((clean**2).mean(axis=1)/(err**2).mean(axis=1)))

        ci_rmses = get_percentiles(rmses)
        ci_snrs = get_percentiles(snrs)

        return pd.DataFrame({
            "metric": ["RMSE", "SNR"],
            "mean": [np.mean(rmses), np.mean(snrs)],
            "ci_low": [ci_rmses[0], ci_snrs[0]],
            "ci_high": [ci_rmses[1], ci_snrs[1]],
        })

# # example usage
# config_path = Path('configs/train_config.yaml')
# evaluator = Evaluator(config_path)
# evaluator.plot_examples(6)
# evaluator.results
