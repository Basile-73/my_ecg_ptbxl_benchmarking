import argparse
import os, yaml
from pathlib import Path
import itertools
from trainer import SimpleTrainer
from evaluator import Evaluator
import numpy as np
import pandas as pd
from utils import nested_get


class CombinationExperiment:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.data_configs = self._get_configs('data_configs')
        self.model_configs = self._get_configs('model_configs')
        self.train_configs = self._get_configs('train_configs')
        self.configs = self._assemble_configs()
        self.config_paths = self._get_config_paths()
        self.results_summary = None

    def _get_configs(self, folder: str):
        configs = []
        path = Path(f"experiments/{self.exp_name}/{folder}")
        for file in os.listdir(path):
            if file.endswith(".yaml"):
                with open(os.path.join(path, file)) as f:
                    configs.append(yaml.safe_load(f))
        return configs

    def _assemble_configs(self):
        configs = []
        for data_config, model_config, train_config in itertools.product(self.data_configs, self.model_configs, self.train_configs):
            config = {**model_config, **data_config, **train_config}
            configs.append(config)
        return configs

    def _get_config_paths(self, output_folder: str = "outputs"):
        config_paths = []
        for config in self.configs:
            model_name = config["model"]
            duration = config["simulation_params"]["duration"]
            folder_name = f"{output_folder}/{self.exp_name}/{duration}s_{model_name}"
            config_paths.append(os.path.join(folder_name, 'config.yaml'))
        return config_paths

    def _create_folders_and_save_configs(self, output_folder: str = "outputs"):
        for config, config_path in zip(self.configs, self.config_paths):
            folder_path = Path(config_path).parent
            os.makedirs(folder_path)
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f)

    def summarize_results(self, keys: list[str]):
        folders = [Path(f).parent for f in self.config_paths]
        all_results = pd.DataFrame()

        for folder in folders:
            results = pd.read_csv(os.path.join(folder, "results.csv"), index_col = 0)
            out = results.set_index('metric').stack().to_frame().T # single line, double index

            with open(os.path.join(folder, "config.yaml")) as f:
                config = yaml.safe_load(f)
            for key in keys:
                value = nested_get(config, key)
                out.insert(0, key, value)

            all_results= pd.concat([all_results, out])
        return all_results.sort_values(by=keys)

    def run(self):
        for config, config_path in zip(self.configs, self.config_paths):
            print(F"Training model {config['model']} on sequence length {config['simulation_params']['duration']}s")
            trainer = SimpleTrainer(Path(config_path))
            trainer.train()

            loss_histories = [trainer.train_loss_history, trainer.test_loss_history]
            for loss_history, name in zip(loss_histories, ['train', 'test']):
                arr = np.array([
                    float(l.detach() if hasattr(l, "detach") else l)
                    for l in loss_history
                ]) # TODO: find out why mix of tensors and floats here
                np.save(os.path.join(Path(config_path).parent, f'{name}.npy'), arr)

            print(F"Evaulating model {config['model']} on sequence length {config['simulation_params']['duration']}s")
            evaluator = Evaluator(Path(config_path))
            results = evaluator.results
            restuls_path = os.path.join(Path(config_path).parent, 'results.csv')
            results.to_csv(restuls_path, sep=",")

        self.results_summary = self.summarize_results(keys=['model', 'simulation_params.duration'])
        self.results_summary.to_csv(os.path.join(Path(f"outputs/{self.exp_name}"), 'summary.csv'), sep = ",")

# # example usage
# exp_name = "length_mamba"
# experiment = CombinationExperiment(exp_name)
# experiment.config_paths
# experiment._create_folders_and_save_configs()
# experiment.run()


def main(exp_name:str):
    experiment = CombinationExperiment(exp_name)
    experiment._create_folders_and_save_configs()
    experiment.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        help="Must match experiments folder name and becomes output folder name"
    )
    args = parser.parse_args()
    exp_name = args.exp_name
    main(exp_name)
