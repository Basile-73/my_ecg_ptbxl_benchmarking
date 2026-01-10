import wandb

def initialize_wandb(experiment_name: str, run_name: str, config: dict):
    run = wandb.init(
        entity="basile-73-eth-zurich",
        project="new_code",
        name=run_name,
        tags=[experiment_name],
        config=config
    )
    return run

def concat_all_configs(configs: list[dict]):
    combined_config = {}
    for config in configs:
        combined_config.update(config)
    return combined_config
