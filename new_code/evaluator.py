from pathlib import Path
import torch
from ecg_noise_factory.noise import NoiseFactory
from dataset import LengthExperimentDataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from functools import cached_property
import pandas as pd
from utils.getters import get_percentiles, get_data_set
import yaml


from utils.getters import (
    get_model,
    read_config,
    get_sampleset_name,
    get_sampleset_name_mitbh_arr,
    get_sampleset_name_mitbh_sin,
    get_sampleset_name_european_st_t,
    get_sampleset_name_ptbxl,
)

class Evaluator:
    def __init__(self, config_path: Path, experiment_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config, model_type, model_name, simulation_params, split_length, data_volume, noise_paths, training_config = (
            read_config(config_path)
        )

        self.split_length = split_length
        self.experiment_name = experiment_name
        self.simulation_params = simulation_params
        self.duration = simulation_params["duration"]
        self.sequence_length = split_length
        self.model_config = model_config


        self.eval_noise_factory = NoiseFactory(
            noise_paths["data_path"],
            simulation_params["sampling_rate"],
            noise_paths["config_path"],
            mode="eval",
            seed=42,
        )

        self.data_volumne = data_volume
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.dataset_type = config["dataset"]
        self.ptb_xl_params = config.get("ptb_xl_params")
        self.train_sample_set_name = self._get_sampleset_name()

        scaler_stats = np.loadtxt(f'data/{self.train_sample_set_name}_scaler_stats')

        self.eval_dataset = get_data_set(
            config_path=config_path,
            mode="eval",
            noise_factory=self.eval_noise_factory,
            median=scaler_stats[0],
            iqr=scaler_stats[1],
        )

        self.eval_data_loader = DataLoader(self.eval_dataset, training_config["batch_size"])

        self.model_name = model_name
        self.model = get_model(model_type, sequence_length=self.split_length, model_config=self.model_config)
        weights_name = f"{experiment_name}_" if experiment_name else ""
        weights_name = f"{weights_name}{self.eval_dataset.dataset_type}_"
        state = torch.load(f"model_weights/{weights_name}best_{self.split_length}_{model_name}.pth", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def _get_sampleset_name(self):
        if self.dataset_type == 'synthetic':
            return get_sampleset_name(
                self.simulation_params,
                self.data_volumne['n_samples_train'],
                'train')
        elif self.dataset_type == 'mitbih_arrhythmia':
            return get_sampleset_name_mitbh_arr(
                self.duration,
                self.data_volumne['n_samples_train'],
                'train'
            )
        elif self.dataset_type == 'mitbih_sinus':
            return get_sampleset_name_mitbh_sin(
                self.duration,
                self.data_volumne['n_samples_train'],
                'train'
            )
        elif self.dataset_type == 'european_st_t':
            return get_sampleset_name_european_st_t(
                self.duration,
                self.data_volumne['n_samples_train'],
                'train'
            )
        elif self.dataset_type == 'ptb_xl':
            if not self.ptb_xl_params:
                raise ValueError("ptb_xl_params missing from config")
            folds = list(range(1, 9))[: self.data_volumne['n_folds_train']]
            return get_sampleset_name_ptbxl(
                split_length=self.split_length,
                folds=folds,
                original_fs=self.ptb_xl_params['original_sampling_rate'],
                mode='train',
                lead_index=self.ptb_xl_params.get('lead_index', 0)
            )
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not recognized")


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
        rmses, snrs, pccs = [], [], []

        self.model.eval()
        for noisy, clean in self.eval_data_loader:
            noisy = noisy.to(self.device)
            denoised = self.model(noisy).detach().cpu().numpy().reshape(len(clean), -1)
            clean = clean.numpy().reshape(len(clean), -1)

            err = clean - denoised
            rmses.extend(np.sqrt((err**2).mean(axis=1)))
            snrs.extend(10*np.log10((clean**2).mean(axis=1)/(err**2).mean(axis=1)))

            # Calculate PCC for each sample
            for i in range(len(clean)):
                pcc = np.corrcoef(clean[i], denoised[i])[0, 1]
                pccs.append(pcc)

        ci_rmses = get_percentiles(rmses)
        ci_snrs = get_percentiles(snrs)
        ci_pccs = get_percentiles(pccs)

        return pd.DataFrame({
            "metric": ["RMSE", "SNR", "PCC"],
            "mean": [np.mean(rmses), np.mean(snrs), np.mean(pccs)],
            "ci_low": [ci_rmses[0], ci_snrs[0], ci_pccs[0]],
            "ci_high": [ci_rmses[1], ci_snrs[1], ci_pccs[1]],
        })

    def save_results(self):
        experiment_string = f"{self.experiment_name}" if self.experiment_name else ""
        folder = Path("outputs") / experiment_string / self.eval_dataset.dataset_type / f"{self.split_length}_{self.model_name}"
        folder.mkdir(parents=True, exist_ok=False)

        file_path = folder / "results.csv"
        self.results.to_csv(file_path, sep=",")


class Stage2Evaluator(Evaluator):
    def __init__(self, config_path: Path, stage1_type, stage1_weights_path, experiment_name=None):
        super().__init__(config_path, experiment_name)
        self.stage1_model = get_model(stage1_type, sequence_length=self.sequence_length, model_config=self.model_config)
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
        rmses, snrs, pccs = [], [], []

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

            # Calculate PCC for each sample
            for i in range(len(clean)):
                pcc = np.corrcoef(clean[i], denoised[i])[0, 1]
                pccs.append(pcc)

        ci_rmses = get_percentiles(rmses)
        ci_snrs = get_percentiles(snrs)
        ci_pccs = get_percentiles(pccs)

        return pd.DataFrame({
            "metric": ["RMSE", "SNR", "PCC"],
            "mean": [np.mean(rmses), np.mean(snrs), np.mean(pccs)],
            "ci_low": [ci_rmses[0], ci_snrs[0], ci_pccs[0]],
            "ci_high": [ci_rmses[1], ci_snrs[1], ci_pccs[1]],
        })

# # example usage
# import yaml
# from pathlib import Path
# experiment_name = 'multitrain_5'
# config_path = Path('outputs/train_runs/2025-12-14_14-34-49/multitrain_5_best_1800_drnet_unet_1.yaml')
# with open(config_path) as f:
#     config = yaml.safe_load(f)
# stage1_type = config["model"]["stage_1_type"]
# print(f"Stage 1 type: {stage1_type}")
# stage1_weights_path = config["model"]["stage_1_weights_path"]
# print(f"Stage 1 weights path: {stage1_weights_path}")
# evaluator = Stage2Evaluator(
#     config_path=config_path,
#     stage1_type=stage1_type,
#     stage1_weights_path=stage1_weights_path,
#     experiment_name=experiment_name
# )
# evaluator.plot_examples(6)
# evaluator.results
# evaluator.save_results()
