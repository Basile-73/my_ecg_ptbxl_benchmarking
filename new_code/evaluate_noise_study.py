
# 1. load the test config
import yaml
config_name = 'ptb_xl'
test_config = yaml.safe_load(open(f'experiments/noise_study/{config_name}.yaml'))

# 2. create noise configs and save them in noise/configs/temp
import numpy as np
all_noise_levels = {}
resolution = test_config['noise']['steps']
for noise_type in ['em', 'bw', 'ma', 'AWGN']:
     noise_range = [test_config['noise']['range'][noise_type][0], test_config['noise']['range'][noise_type][1]]
     noise_levels = np.linspace(noise_range[0], noise_range[1], resolution)
     all_noise_levels[noise_type] = noise_levels
     print(noise_levels)

     noise_config = {'SNR': {'em': None, 'ma': None, 'bw': None, 'AWGN': None}}

     for i in range(resolution):
          noise_config['SNR'][noise_type] = float(noise_levels[i])
          yaml.dump(noise_config, open(f'noise/configs/temp/{noise_type}/{i}.yaml', 'w'))

for i in range(resolution):
    noise_config = {'SNR': {'em': None, 'ma': None, 'bw': None, 'AWGN': None}}
    for noise_type in ['em', 'bw', 'ma', 'AWGN']:
        noise_config['SNR'][noise_type] = float(all_noise_levels[noise_type][i])
    yaml.dump(noise_config, open(f'noise/configs/temp/combined/{i}.yaml', 'w'))



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
        for noise_type in ['em', 'bw', 'ma', 'AWGN', 'combined']:
            for i in range(resolution):
                config = deepcopy(base_config)
                config['noise_paths']['config_path'] = f'noise/configs/temp/{noise_type}/{i}.yaml'
                all_configs.append(config)

# 5. Evaluate for each config and append results
    from train_multiple import group_configs, get_model_weights_name, save_config
    import traceback
    from evaluator import Evaluator
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
            # append results here
         except Exception as e:
            print(f"Error occurred while evaluating {config['model']['type']}: {str(e)}")
            traceback.print_exc()
            continue  # Continue to the next configuration
# 6. Format results
         res = evaluator.results
         res['model_type'] = config['model']['type']
         res['model_name'] = config['model']['name']
         res['dataset'] = config['dataset']
         res['noise_type'] = config['noise_paths']['config_path'].split('/')[-2]
         res['noise_level'] = config['noise_paths']['config_path'].split('/')[-1].split('.')[0]
         res['experiment_name'] = experiment_name
         final_res.append(res)

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
