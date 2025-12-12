from trainer import SimpleTrainer, Stage2Trainer, MambaTrainer
from utils.read_configs import get_configs, assemble_configs, get_config_paths
import itertools
from datetime import datetime
import yaml, os

def get_model_weights_name(config, experiment_name):
        return f"{experiment_name}_best_{config['split_length']}_{config['model']['name']}"

def check_paths(experiment_name):
    model_configs = get_configs(f'experiments/{experiment_name}/model_configs')
    data_configs = get_configs(f'experiments/{experiment_name}/data_configs')

    for model_config, data_config in itertools.product(model_configs, data_configs):
        config = {**model_config, **data_config}
        print(get_model_weights_name(config, experiment_name) + '.pth')

# check_paths('multitrain')

def group_configs(configs: list[str]):
     stage_1_configs = [config for config in configs if (
         config['model']['is_stage_2']==False and
         config['model']['is_mamba']==False)]

     stage_2_configs = [config for config in configs if (
         config['model']['is_stage_2']==True and
         config['model']['is_mamba']==False)]

     mamba_configs = [config for config in configs if (
         config['model']['is_stage_2']==False and
         config['model']['is_mamba']==True)]

     return stage_1_configs, stage_2_configs, mamba_configs

def save_config(config, now_str, model_name):
    os.makedirs(f"outputs/train_runs/{now_str}", exist_ok=True)
    path = f"outputs/train_runs/{now_str}/{model_name}.yaml"
    with open(path, "w"):
        yaml.dump(config, open(path, "w"))
    return path


def main(experiment_name):
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_configs = get_configs(f'experiments/{experiment_name}/model_configs')
    data_configs = get_configs(f'experiments/{experiment_name}/data_configs')

    configs = []
    for model_config, data_config in itertools.product(model_configs, data_configs):
        configs.append({**model_config, **data_config})

    stage_1_configs, stage_2_configs, mamba_configs = group_configs(configs)

    for config in stage_1_configs:
        model_name = get_model_weights_name(config, experiment_name)
        print(f'Training model: {model_name}')
        config_path = save_config(config, now_str, model_name)
        trainer = SimpleTrainer(
             config_path=config_path,
             experiment_name = experiment_name
        )
        trainer.train()

    for config in stage_2_configs:
        model_name = get_model_weights_name(config, experiment_name)
        print(f'Training model: {model_name}')
        config_path = save_config(config, now_str, model_name)

        stage_1_type = config['model']['stage_1_type']
        stage_1_weights_path = config['model']['stage_1_weights_path']
        trainer = Stage2Trainer(
            config_path = config_path,
            stage1_type=stage_1_type,
            stage1_weights_path=stage_1_weights_path,
            experiment_name=experiment_name
        )
        trainer.train()

    for config in mamba_configs:
        model_name = get_model_weights_name(config, experiment_name)
        print(f'Training model: {model_name}')
        config_path = save_config(config, now_str, model_name)

        pre_trained_weights_path = config['model']['pre_trained_weights_path']
        trainer = MambaTrainer(
            config_path=config_path,
            experiment_name=experiment_name,
            pre_trained_weights_path=pre_trained_weights_path
        )
        trainer.train()


if __name__ == "__main__":
    main('multitrain')
