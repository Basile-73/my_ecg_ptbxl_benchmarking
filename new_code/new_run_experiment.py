import argparse
import os, yaml
from pathlib import Path
import itertools
from trainer import SimpleTrainer
from evaluator import Evaluator
import numpy as np
import pandas as pd
from utils.getters import nested_get
from utils.read_configs import get_configs, assemble_configs, get_config_paths


class CombinationExperiment2:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.data_configs = get_configs(f'experiments/{exp_name}/data_configs')
        self.model_configs = get_configs(f'experiments/{exp_name}/model_configs')

        self.experiment_config = yaml.safe_load(open(f'experiments/{exp_name}/experiment_config.yaml'))
        self.traits = self.experiment_config.get('traits', '[model.name]')

        configs = assemble_configs(self.data_configs, self.model_configs)
        config_paths = get_config_paths(self.configs, self.traits, exp_name)
        config_tuples = list(zip(self.configs, self.config_paths))
        config_tuples.sort(key=lambda x: x[0]["simulation_params"]["duration"])
        self.configs, self.config_paths = zip(*config_tuples)


        self.results_summary = None

        def _get_trainer(self, config):
            return SimpleTrainer(config)

        def run(self):
            for config, config_path in zip(self.configs, self.config_paths):
                trainer = self._get_trainer(config) # ! In the simple trainer replace the Synthetic ECG Dataset but maintain backward compatibility
