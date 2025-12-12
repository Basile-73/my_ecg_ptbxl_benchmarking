from pathlib import Path
import torch
from ecg_noise_factory.noise import NoiseFactory
from dataset import SyntheticEcgDataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from functools import cached_property
import pandas as pd
from utils.getters import get_percentiles


from utils.getters import get_model, read_config, get_sampleset_name

class Evaluator:
    def __init__(self, config_path: Path, experiment_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_type, model_name, simulation_params, data_volume, noise_paths, training_config = (
            read_config(config_path)
        )

        self.experiment_name = experiment_name
        self.simulation_params = simulation_params
        self.duration = simulation_params["duration"]
        self.sequence_length = simulation_params["duration"] * simulation_params["sampling_rate"]
        self.model_name = model_name
        self.model = get_model(model_type, sequence_length=self.sequence_length)
        weights_name = f"{experiment_name}_" if experiment_name else ""
        state = torch.load(f"model_weights/{weights_name}best_{self.simulation_params['duration']}s_{model_name}.pth", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)


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

    def save_results(self):
        experiment_string = f"{self.experiment_name}" if self.experiment_name else ""
        folder = Path("outputs") / experiment_string
        folder.mkdir(parents=True, exist_ok=False)

        file_path = folder / f"{self.duration}s_{self.model_name}.csv"
        self.results.to_csv(file_path, sep=",")


class Stage2Evaluator(Evaluator):
    def __init__(self, config_path: Path, stage1_type, stage1_weights_path, experiment_name=None):
        super().__init__(config_path, experiment_name)
        self.stage1_model = get_model(stage1_type, sequence_length = self.sequence_length)
        self.stage1_model.load_state_dict(torch.load(stage1_weights_path))
        self.stage1_model.eval()
        self.stage1_model.to(self.device)

    def plot_examples(self, idx):
        noisy, clean = self.eval_dataset[idx]

        noisy = noisy.to(self.device)
        self.model = self.model.to(self.device)

        with torch.no_grad():
            pred_1 = self.stage1_model(noisy.unsqueeze(0))
            input_stage_2 = torch.cat((noisy.unsqueeze(0), pred_1), dim=1)
            denoised = self.model(input_stage_2)
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
            pred_1 = self.stage1_model(noisy)
            input_stage_2 = torch.cat((noisy, pred_1), dim=1)
            denoised = self.model(input_stage_2).detach().cpu().numpy().reshape(len(clean), -1)
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
# evaluator.save_results()
