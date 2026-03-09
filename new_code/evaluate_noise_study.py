
# 1. load the test config
import yaml
import numpy as np
config_name = 'high_range_smooth'
test_config = yaml.safe_load(open(f'experiments/noise_study/{config_name}.yaml'))

# 2. create noise configs and save them in noise/configs/temp
all_noise_levels = {}
resolution = test_config['noise']['steps']
plausible_range = test_config['noise'].get('plausible_range', None)  # backward compat

for noise_type in ['em', 'bw', 'ma', 'AWGN']:
     noise_range = [test_config['noise']['range'][noise_type][0], test_config['noise']['range'][noise_type][1]]
     noise_levels = np.linspace(noise_range[0], noise_range[1], resolution)
     all_noise_levels[noise_type] = noise_levels
     print(noise_levels)

     noise_config = {'SNR': {'em': None, 'ma': None, 'bw': None, 'AWGN': None}}

     for i in range(resolution):
          noise_config['SNR'][noise_type] = float(noise_levels[i])
          yaml.dump(noise_config, open(f'noise/configs/temp/{noise_type}/{i}.yaml', 'w'))

# Combined noise levels: linspace over plausible_range (or fall back to original range)
combined_noise_levels = {}
for noise_type in ['em', 'bw', 'ma', 'AWGN']:
    if plausible_range is not None:
        lo, hi = plausible_range[noise_type][0], plausible_range[noise_type][1]
    else:
        lo, hi = test_config['noise']['range'][noise_type][0], test_config['noise']['range'][noise_type][1]
    combined_noise_levels[noise_type] = np.linspace(lo, hi, resolution)

for i in range(resolution):
    noise_config = {'SNR': {'em': None, 'ma': None, 'bw': None, 'AWGN': None}}
    for noise_type in ['em', 'bw', 'ma', 'AWGN']:
        noise_config['SNR'][noise_type] = float(combined_noise_levels[noise_type][i])
    yaml.dump(noise_config, open(f'noise/configs/temp/combined/{i}.yaml', 'w'))


def _combined_snr_db(snr_db_dict):
    """Compute combined SNR (dB) for independent noise sources.

    Uses: SNR_combined = 1 / sum(1 / SNR_linear_i) for each active noise type.
    Returns None if all sources are None/zero.
    """
    inv_sum = 0.0
    for snr_db in snr_db_dict.values():
        if snr_db is not None:
            snr_lin = 10 ** (snr_db / 10.0)
            if snr_lin > 0:
                inv_sum += 1.0 / snr_lin
    if inv_sum == 0:
        return None
    return float(10.0 * np.log10(1.0 / inv_sum))


# 3. build and filter all model configs
import itertools
from utils.read_configs import get_configs
from datetime import datetime
from copy import deepcopy

final_res = []
for experiment in test_config['experiments']:
    experiment_name = experiment.split('/')[-1]
    model_configs = get_configs(f'{experiment}/model_configs')
    data_configs = get_configs(f'{experiment}/data_configs')

    configs = [] # One config per model
    for model_config, data_config in itertools.product(model_configs, data_configs):
            candidate_config = {**model_config, **data_config}
            if (candidate_config['model']['type'] in test_config['filters']['model_types'] and
                candidate_config['dataset'] in test_config['filters']['datasets']):
                configs.append(candidate_config)

# 4. create eval configs and save them in outputs/temp
    all_configs = []
    # One config per model, noise_type and resolution
    for base_config in configs:
        for noise_type in ['em', 'bw', 'ma', 'AWGN']:
            for i in range(resolution):
                config = deepcopy(base_config)
                config['noise_paths']['config_path'] = f'noise/configs/temp/{noise_type}/{i}.yaml'
                config['_snr_value'] = float(all_noise_levels[noise_type][i])
                all_configs.append(config)

        # combined: linspace over plausible_range (or original range if not set)
        for i in range(resolution):
            config = deepcopy(base_config)
            config['noise_paths']['config_path'] = f'noise/configs/temp/combined/{i}.yaml'
            snr_vals = {nt: float(combined_noise_levels[nt][i]) for nt in ['em', 'bw', 'ma', 'AWGN']}
            config['_snr_value'] = _combined_snr_db(snr_vals)
            all_configs.append(config)

# 5. Evaluate for each config and append results
    from train_multiple import group_configs, get_model_weights_name, save_config
    import traceback
    from evaluator import Evaluator, Stage2Evaluator
    stage_1_configs, stage_2_configs, mamba_configs = group_configs(all_configs)

    for config in [*stage_1_configs, *mamba_configs]:
         print(f'Evaluating model: {config["model"]["type"]} with noise config: {config["noise_paths"]}')
         config_path =f'outputs/temp/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}.yaml'
         yaml.dump(config, open(config_path, 'w'))
         try:
            evaluator = Evaluator(
                config_path=config_path,
                experiment_name=experiment_name,
            )
# 6. Format results
            res = evaluator.results
            res['model_type'] = config['model']['type']
            res['model_name'] = config['model']['name']
            res['dataset'] = config['dataset']
            res['noise_type'] = config['noise_paths']['config_path'].split('/')[-2]
            res['noise_level'] = config['noise_paths']['config_path'].split('/')[-1].split('.')[0]
            res['snr_value'] = config.get('_snr_value')  # actual SNR in dB (None if unavailable)
            res['experiment_name'] = experiment_name
            final_res.append(res)
         except Exception as e:
            print(f"Error occurred while evaluating {config['model']['type']}: {str(e)}")
            traceback.print_exc()
            continue  # Continue to the next configuration

    for config in stage_2_configs:
         print(f'Evaluating model: {config["model"]["type"]} with noise config: {config["noise_paths"]}')
         config_path =f'outputs/temp/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}.yaml'
         yaml.dump(config, open(config_path, 'w'))
         try:
            evaluator = Stage2Evaluator(
                config_path=config_path,
                stage1_type=config["model"]["stage_1_type"],
                stage1_weights_path=config["model"]["stage_1_weights_path"],
                experiment_name=experiment_name,
            )
            res = evaluator.results
            res['model_type'] = config['model']['type']
            res['model_name'] = config['model']['name']
            res['dataset'] = config['dataset']
            res['noise_type'] = config['noise_paths']['config_path'].split('/')[-2]
            res['noise_level'] = config['noise_paths']['config_path'].split('/')[-1].split('.')[0]
            res['snr_value'] = config.get('_snr_value')
            res['experiment_name'] = experiment_name
            final_res.append(res)
         except Exception as e:
            print(f"Error occurred while evaluating {config['model']['type']}: {str(e)}")
            traceback.print_exc()
            continue

# 7. Create logic to store the results
import pandas as pd
df = pd.concat(final_res, ignore_index=True)
df.to_csv(f'outputs/noise_study_results_{config_name}.csv', index=False)

# 8. Delete all temporary files (but keep the folders)
import shutil
from pathlib import Path

shutil.rmtree('noise/configs/temp', ignore_errors=True)
[Path(f'noise/configs/temp/{x}').mkdir(parents=True) for x in ['em', 'bw', 'ma', 'AWGN', 'combined']]

shutil.rmtree('outputs/temp', ignore_errors=True)
Path('outputs/temp').mkdir(parents=True)

print("DONE")
