import argparse
import os, yaml
from pathlib import Path
import itertools
from trainer import SimpleTrainer
from evaluator import Evaluator


class CombinationExperiment:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.data_configs = self._get_configs('data_configs')
        self.model_configs = self._get_configs('model_configs')
        self.train_configs = self._get_configs('train_configs')
        self.configs = self._assemble_configs()
        self.config_paths = self._get_config_paths()

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

    def run(self):
        for config, config_path in zip(self.configs, self.config_paths):
            print(F"Training model {config['model']} on sequence length {config['simulation_params']['duration']}s")
            trainer = SimpleTrainer(Path(config_path))
            trainer.train()

            print(F"Evaulating model {config['model']} on sequence length {config['simulation_params']['duration']}s")
            evaluator = Evaluator(Path(config_path))
            results = evaluator.results
            restuls_path = os.path.join(Path(config_path).parent, 'restults.csv')
            results.to_csv(restuls_path, sep=",")

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
