import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from utils.getters import (
    read_config,
    get_loss_function,
    get_model,
    get_optimizer,
    get_scheduler,
    get_data_set,
)
from utils.wandb_utils import initialize_wandb, concat_all_configs
from pathlib import Path
from ecg_noise_factory.noise import NoiseFactory
from dataset import LengthExperimentDataset, SyntheticEcgDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import random
import numpy as np
import math


class ExponentialMovingAverage:
    """Maintains exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        """
        Initialize EMA with model reference and decay rate.

        Args:
            model: PyTorch model to track
            decay: EMA decay rate (default: 0.999)
        """
        self.decay = decay
        self.model = model
        self.shadow = {}  # Store EMA parameters
        self.backup = {}  # Temporary storage during evaluation

        # Initialize shadow parameters by deep copying all model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights after optimizer step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Apply EMA formula: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        """Temporarily replace model weights with EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        """Restore original model weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].to(param.device)
        self.backup = {}

    def state_dict(self):
        """Return shadow parameters for saving."""
        return self.shadow


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    # Set CUBLAS workspace config before CUDA initialization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforce deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Disable TF32 for full precision
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class SimpleTrainer:
    def __init__(self, config_path: Path, seed=42,
                 experiment_name=None,
                 pre_trained_weights_path=None,
                 load_weights=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config, model_type, model_name, simulation_params, split_length, data_volume, noise_paths, training_config = (
            read_config(config_path)
        )

        seed = model_config.get("seed", seed)
        set_seed(seed)
        self.experiment_name = experiment_name

        self.split_length = split_length
        self.simulation_params = simulation_params
        self.data_volume = data_volume
        self.noise_paths = noise_paths
        self.training_config = training_config
        self.model_type = model_type
        self.model_config = model_config
        self.track_wandb = bool(self.training_config.get("wandb", False))

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
            print(f"Loading weights from {pre_trained_weights_path}")
            self.model.load_state_dict(torch.load(pre_trained_weights_path))

        self.loss_fn = get_loss_function(training_config["loss_function"])
        self.optimizer = get_optimizer(
            training_config["optimizer"], self.model.parameters(), **training_config
        )
        self.scheduler = get_scheduler(training_config["scheduler"], self.optimizer, **training_config)
        self.patience = training_config["early_stopping_patience"]
        self.epochs = training_config["epochs"]

        # Store EMA config for later initialization (after model.to(device))
        self.use_ema = self.training_config.get("ema", False)
        self.ema_decay = self.training_config.get("ema_decay", 0.999)
        self.ema = None  # Will be initialized in train() after model.to(device)

        self.train_loss_history = None
        self.test_loss_history = None

        self.is_mecge = (self.model_type == "mecge")

    def _train_loop(self):
        self.model.train()
        train_loss = 0
        for batch, (X, y) in tqdm(
            enumerate(self.train_data_loader), total=len(self.train_data_loader)
        ):
            X, y = X.to(self.device), y.to(self.device)
            if not self.is_mecge:
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
            else:
                loss =self.model.get_loss(y, X) # MECGE expects (clean, noisy)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update EMA weights after optimizer step
            if self.use_ema:
                self.ema.update()
        train_loss = train_loss / len(self.train_data_loader)
        return train_loss

    def _test_loop(self):
        self.model.eval()

        # Apply EMA weights before evaluation
        if self.use_ema:
            self.ema.apply_shadow()

        test_loss = 0
        total_squared_error = 0.0
        total_signal_power = 0.0
        total_numel = 0
        with torch.no_grad():
            for X, y in tqdm(self.test_data_loader, total=len(self.test_data_loader)):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                error = y - pred
                total_squared_error += torch.sum(error**2).item()
                total_signal_power += torch.sum(y**2).item()
                total_numel += error.numel()

        test_loss = test_loss / len(self.test_data_loader)

        # Guard against division by zero when computing metrics
        eps = 1e-12
        mse = total_squared_error / max(total_numel, 1)
        test_RMSE = math.sqrt(mse)
        test_SNR = 10 * math.log10((total_signal_power + eps) / (total_squared_error + eps))

        # Restore original weights after evaluation
        if self.use_ema:
            self.ema.restore()

        return test_loss, test_RMSE, test_SNR

    def train(self):
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Initialize EMA after model is on correct device
        if self.use_ema:
            self.ema = ExponentialMovingAverage(self.model, decay=self.ema_decay)

        best, wait, patience = 1e9, 0, self.patience
        train_loss_history = []
        test_loss_history = []
        test_SNR_history = []
        test_RMSE_history = []

        wandb_run = None
        if self.track_wandb:
            wandb_config = concat_all_configs([
                self.model_config,
                self.simulation_params,
                self.training_config,
                self.noise_paths,
                {"data_volume": self.data_volume, "split_length": self.split_length}
            ])

            wandb_run = initialize_wandb(
                experiment_name=self.experiment_name if self.experiment_name else "experiment not specified",
                dataset_name=self.train_dataset.dataset_type,
                run_name=self.model_name,
                config=wandb_config
            )

        for t in range(self.epochs):
            print(f"-------------------------------\nEpoch {t + 1}")
            train_loss = self._train_loop()
            test_loss, test_RMSE, test_SNR = self._test_loop()

            print(
                f"Train loss: {train_loss}, Test loss: {test_loss}\n-------------------------------"
                f"\nTest RMSE: {test_RMSE}, Test SNR: {test_SNR}"
            )
            # Handle different scheduler types
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(test_loss)
            else:
                self.scheduler.step()

            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            test_SNR_history.append(test_SNR)
            test_RMSE_history.append(test_RMSE)

            if wandb_run:
                wandb_run.log({
                    "train/loss": train_loss,
                    "test/loss": test_loss,
                    "test/RMSE": test_RMSE,
                    "test/SNR": test_SNR,
                    "epoch": t + 1
                })


            if test_loss < best:
                best, wait = test_loss, 0
                weights_name = f"{self.experiment_name}_" if self.experiment_name else ""
                weights_name = f"{weights_name}{self.train_dataset.dataset_type}_"

                # Save EMA weights if enabled, otherwise save regular weights
                if self.use_ema:
                    # Temporarily apply EMA weights to model, save complete state_dict, then restore
                    self.ema.apply_shadow()
                    torch.save(
                        self.model.state_dict(), f"model_weights/{weights_name}best_{self.split_length}_{self.model_name}.pth"
                    )
                    self.ema.restore()
                else:
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

        # Load weights and reinitialize EMA shadow if enabled
        self.model.load_state_dict(
            torch.load(f"model_weights/{weights_name}best_{self.split_length}_{self.model_name}.pth")
        )
        if self.use_ema:
            # Reinitialize EMA shadow from loaded model parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema.shadow[name] = param.data.clone().to(param.device)
        self.train_loss_history = train_loss_history
        self.test_loss_history = test_loss_history
        print("Done!")
        if wandb_run:
            wandb_run.finish()


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
                    p.requires_grad = name in missing

        self.optimizer = get_optimizer(
            self.training_config["optimizer"],
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.training_config
        )

class Stage2Trainer(SimpleTrainer):
    def __init__(self, config_path, stage1_type, stage1_weights_path, seed=42, experiment_name=None,
                 pre_trained_weights_path=None):
        super().__init__(config_path, seed, experiment_name, pre_trained_weights_path, load_weights=False)

        self.stage1_model = get_model(stage1_type, sequence_length=self.sequence_length, model_config=self.model_config)
        print(f"Loading stage 1 weights from {stage1_weights_path}")
        self.stage1_model.load_state_dict(torch.load(stage1_weights_path))
        self.stage1_model.eval()
        self.stage1_model.to(self.device)

    def _train_loop(self):
        train_loss = 0
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

            # Update EMA weights after optimizer step
            if self.use_ema:
                self.ema.update()

            train_loss += loss.item()
        train_loss = train_loss / len(self.train_data_loader)
        return train_loss

    def _test_loop(self):
        self.model.eval()

        # Apply EMA weights before evaluation
        if self.use_ema:
            self.ema.apply_shadow()

        test_loss = 0
        total_squared_error = 0.0
        total_signal_power = 0.0
        total_numel = 0
        with torch.no_grad():
            for X, y in tqdm(self.test_data_loader, total=len(self.test_data_loader)):
                X, y = X.to(self.device), y.to(self.device)
                pred_1 = self.stage1_model(X)
                input_stage_2 = torch.cat((X, pred_1), dim=1)
                pred = self.model(input_stage_2)
                test_loss += self.loss_fn(pred, y).item()

                error = y - pred
                total_squared_error += torch.sum(error**2).item()
                total_signal_power += torch.sum(y**2).item()
                total_numel += error.numel()
        test_loss = test_loss / len(self.test_data_loader)

        eps = 1e-12
        mse = total_squared_error / max(total_numel, 1)
        test_RMSE = math.sqrt(mse)
        test_SNR = 10 * math.log10((total_signal_power + eps) / (total_squared_error + eps))


        # Restore original weights after evaluation
        if self.use_ema:
            self.ema.restore()

        return test_loss, test_RMSE, test_SNR

# # example usage

# trainer = SimpleTrainer(
#     config_path=Path("configs/train_config.yaml"),
#     seed=42,
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
