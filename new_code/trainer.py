from utils.getters import (
    read_config,
    get_loss_function,
    get_model,
    get_optimizer,
    get_scheduler,
    get_data_set,
)
from pathlib import Path
from ecg_noise_factory.noise import NoiseFactory
from dataset import LengthExperimentDataset, SyntheticEcgDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import random
import numpy as np


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleTrainer:
    def __init__(self, config_path: Path, seed=42,
                 experiment_name=None,
                 pre_trained_weights_path=None,
                 load_weights=True):
        # Set seed for reproducibility
        set_seed(seed)
        self.experiment_name = experiment_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config, model_type, model_name, simulation_params, split_length, data_volume, noise_paths, training_config = (
            read_config(config_path)
        )
        self.split_length = split_length
        self.simulation_params = simulation_params
        self.data_volume = data_volume
        self.noise_paths = noise_paths
        self.training_config = training_config
        self.model_type = model_type
        self.model_config = model_config

        self.train_noise_factory = NoiseFactory(
            noise_paths["data_path"],
            simulation_params["sampling_rate"],
            noise_paths["config_path"],
            mode="train",
            seed=42,
        )
        self.test_noise_factory = NoiseFactory(
            noise_paths["data_path"],
            simulation_params["sampling_rate"],
            noise_paths["config_path"],
            mode="test",
            seed=42,
        )

        self.train_dataset = get_data_set(config_path=config_path, mode='train', noise_factory=self.train_noise_factory)
        self.test_dataset = get_data_set(config_path=config_path,
                                         mode='test',
                                         noise_factory=self.test_noise_factory,
                                         median=self.train_dataset.median,
                                         iqr=self.train_dataset.iqr)

        # Create worker init function for DataLoader reproducibility
        def worker_init_fn(worker_id):
            np.random.seed(seed + worker_id)
            random.seed(seed + worker_id)

        self.train_data_loader = DataLoader(
            self.train_dataset,
            training_config["batch_size"],
            worker_init_fn=worker_init_fn
        )
        self.test_data_loader = DataLoader(
            self.test_dataset,
            training_config["batch_size"],
            worker_init_fn=worker_init_fn
        )

        self.model_name = model_name
        self.sequence_length = split_length
        self.model = get_model(model_type, sequence_length=self.sequence_length, model_config=self.model_config)

        if pre_trained_weights_path and load_weights:
            self.model.load_state_dict(torch.load(pre_trained_weights_path))

        self.loss_fn = get_loss_function(training_config["loss_function"])
        self.optimizer = get_optimizer(
            training_config["optimizer"], self.model.parameters()
        )
        self.scheduler = get_scheduler(training_config["scheduler"], self.optimizer)
        self.patience = training_config["early_stopping_patience"]
        self.epochs = training_config["epochs"]

        self.train_loss_history = None
        self.test_loss_history = None

        self.is_mecge = (self.model_type == "mecge")

    def _train_loop(self):
        self.model.train()
        for batch, (X, y) in tqdm(
            enumerate(self.train_data_loader), total=len(self.train_data_loader)
        ):
            X, y = X.to(self.device), y.to(self.device)
            if not self.is_mecge:
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
            else:
                loss =self.model.get_loss(y, X) # MECGE expects (clean, noisy)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def _test_loop(self):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in tqdm(self.test_data_loader, total=len(self.test_data_loader)):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss = test_loss / len(self.test_data_loader)
        return test_loss, correct

    def train(self):
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        best, wait, patience = 1e9, 0, self.patience
        train_loss_history = []
        test_loss_history = []
        for t in range(self.epochs):
            print(f"-------------------------------\nEpoch {t + 1}")
            train_loss = self._train_loop()
            test_loss, _ = self._test_loop()
            print(
                f"Train loss: {train_loss}, Test loss: {test_loss}\n-------------------------------"
            )
            self.scheduler.step(test_loss)

            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)

            if test_loss < best:
                best, wait = test_loss, 0
                weights_name = f"{self.experiment_name}_" if self.experiment_name else ""
                weights_name = f"{weights_name}{self.train_dataset.dataset_type}_"
                torch.save(
                    self.model.state_dict(), f"model_weights/{weights_name}best_{self.split_length}_{self.model_name}.pth"
                )
            else:
                wait += 1
                if wait > patience:
                    print(f"Early stop at epoch {t}")
                    break

        weights_name = f"{self.experiment_name}_" if self.experiment_name else ""
        weights_name = f"{weights_name}{self.train_dataset.dataset_type}_"
        self.model.load_state_dict(
            torch.load(f"model_weights/{weights_name}best_{self.split_length}_{self.model_name}.pth")
        )
        self.train_loss_history = train_loss_history
        self.test_loss_history = test_loss_history
        print("Done!")


class MambaTrainer(SimpleTrainer):
    def __init__(self, config_path, seed=42, experiment_name=None,
                 pre_trained_weights_path=None):

        super().__init__(config_path, seed, experiment_name, pre_trained_weights_path, load_weights=False)

        if pre_trained_weights_path:
            missing, unexpected = self.model.load_state_dict(
                torch.load(pre_trained_weights_path), strict=False
            )
            print("missing:", missing)
            print("unexpected:", unexpected)
            if self.model_config.get("train_mamba_only", False):
                for name, p in self.model.named_parameters():
                    p.requires_grad = name.startswith("mamba_layer")

        self.optimizer = get_optimizer(
            self.training_config["optimizer"],
            filter(lambda p: p.requires_grad, self.model.parameters())
        )

class Stage2Trainer(SimpleTrainer):
    def __init__(self, config_path, stage1_type, stage1_weights_path, seed=42, experiment_name=None,
                 pre_trained_weights_path=None):
        super().__init__(config_path, seed, experiment_name, pre_trained_weights_path, load_weights=False)

        self.stage1_model = get_model(stage1_type, sequence_length=self.sequence_length, model_config=self.model_config)
        self.stage1_model.load_state_dict(torch.load(stage1_weights_path))
        self.stage1_model.eval()
        self.stage1_model.to(self.device)

    def _train_loop(self):
        self.model.train()
        for batch, (X, y) in tqdm(
            enumerate(self.train_data_loader), total=len(self.train_data_loader)
        ):
            X, y = X.to(self.device), y.to(self.device)
            pred_1 = self.stage1_model(X)
            input_stage_2 = torch.cat((X, pred_1), dim=1)
            pred = self.model(input_stage_2)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def _test_loop(self):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in tqdm(self.test_data_loader, total=len(self.test_data_loader)):
                X, y = X.to(self.device), y.to(self.device)
                pred_1 = self.stage1_model(X)
                input_stage_2 = torch.cat((X, pred_1), dim=1)
                pred = self.model(input_stage_2)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss = test_loss / len(self.test_data_loader)
        return test_loss, correct

# # example usage

# trainer = SimpleTrainer(
#     config_path=Path("configs/train_config.yaml"),
#     experiment_name="DELETE_ME",
#     pre_trained_weights_path=None
# )
# trainer.train()

# trainer = MambaTrainer(
#     config_path=Path("configs/train_config.yaml"),
#     experiment_name="unet",
#     pre_trained_weights_path='model_weights/unet_best_30s_unet_1.pth'
# )
# trainer.train()

# from evaluator import Evaluator
# evaluator = Evaluator(
#     config_path=Path("configs/train_config.yaml"),
#     experiment_name="unet"
# )
# evaluator.results

# path = Path(f"outputs/AAA_mamba_comparison/unet_drnet_patience_30.csv")
# evaluator.results.to_csv(path, sep=',')

# trainer = Stage2Trainer(
#     config_path=Path("configs/train_config.yaml"),
#     stage1_type='unet',
#     stage1_weights_path= 'model_weights/unet_best_30s_unet_1.pth',
#     experiment_name="stage_2",
# )
# trainer.train()

# from evaluator import Stage2Evaluator
# evaluator = Stage2Evaluator(
#     config_path=Path("configs/train_config.yaml"),
#     stage1_type='unet',
#     stage1_weights_path= 'model_weights/unet_best_30s_unet_1.pth',
#     experiment_name="stage_2"
# )
# evaluator.results
