from train_multiple import get_model_weights_name, group_configs, save_config
from utils.read_configs import get_configs
import itertools
from evaluator import Evaluator, Stage2Evaluator
from datetime import datetime

def main(experiment_name):
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_configs = get_configs(f'experiments/{experiment_name}/model_configs')
    data_configs = get_configs(f'experiments/{experiment_name}/data_configs')

    configs = []
    for model_config, data_config in itertools.product(model_configs, data_configs):
        configs.append({**model_config, **data_config})

    stage_1_configs, stage_2_configs, mamba_configs = group_configs(configs)

    for config in [*stage_1_configs, *mamba_configs]:
        model_name = get_model_weights_name(config, experiment_name)
        print(f'Evaluating model: {model_name}')
        config_path = save_config(config, now_str, model_name)
        try:
            evaluator = Evaluator(
                config_path=config_path,
                experiment_name=experiment_name,
            )
            evaluator.save_results()
        except Exception as e:
            print(f"Error occurred while evaluating {model_name}: {str(e)}")
            continue  # Continue to the next configuration

    for config in stage_2_configs:
        model_name = get_model_weights_name(config, experiment_name)
        print(f'Evaluating model: {model_name}')
        config_path = save_config(config, now_str, model_name)
        stage1_type = config["model"]["stage_1_type"]
        stage1_weights_path = config["model"]["stage_1_weights_path"]

        try:
            evaluator = Stage2Evaluator(
                config_path=config_path,
                stage1_type=stage1_type,
                stage1_weights_path=stage1_weights_path,
                experiment_name=experiment_name,
            )
            evaluator.save_results()
        except Exception as e:
            print(f"Error occurred while evaluating {model_name}: {str(e)}")
            continue

    print("DONE")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        required=True,
        help="Must match experiments folder name and becomes output folder name"
    )
    args = parser.parse_args()
    exp_name = args.exp_name
    main(exp_name)
