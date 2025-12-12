import sys
from pathlib import Path

import os, yaml
import itertools
import numpy as np
import pandas as pd

try:
    from .getters import nested_get
except ImportError:
    from getters import nested_get

def nested_get(d, path):
    for p in path.split("."):
        d = d[p]
    return d

def get_configs(path: Path):
        configs = []
        for file in sorted(os.listdir(path)):
            if file.endswith(".yaml"):
                with open(os.path.join(path, file)) as f:
                    configs.append(yaml.safe_load(f))
        return configs

def assemble_configs(data_configs: list[dict], model_configs: list[dict]):
        configs = []
        for data_config, model_config in itertools.product(data_configs, model_configs):
            config = {**model_config, **data_config}
            configs.append(config)
        return configs

def get_config_paths(configs, traits: list[str], experiment_name: str, output_folder: str = "outputs"):
        config_paths = []
        for config in configs:
            config_path_list = []
            for trait in traits:
                value = nested_get(config, trait)
                config_path_list.append(str(value))
            folder_name = "_".join(config_path_list)
            config_path = f'{output_folder}/{experiment_name}/{folder_name}/config.yaml'
            config_paths.append(config_path)
        return config_paths

# example usage of get config path
# read a config file
# with open('../configs/train_config.yaml') as f:
#     config = yaml.safe_load(f)

# fields = ['model.name', 'simulation_params.duration']
# experiment_name = 'template_experiment'
# config_path = get_config_paths([config], fields, experiment_name)[0]
# print(config_path)

with open('../experiments/template2/experiment_config.yaml') as f:
    config = yaml.safe_load(f)
