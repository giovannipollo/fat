"""Trainer class for training and evaluating models.

Provides a complete training loop with support for:

- Progress bars with tqdm
- Checkpoint saving/loading
- Learning rate warmup
- Mixed precision training (AMP)
- TensorBoard logging
- Validation and test evaluation
- Fault injection for fault-aware training (FAT)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Dict, Iterator, Optional, Tuple, Union

from .optimizer import OptimizerFactory
from .scheduler import SchedulerFactory, SchedulerType
from .experiment import ExperimentManager
from .logging import MetricsLogger
from .loss import LossFactory
from .fault_injection import (
    FaultInjectionConfig,
    ActivationFaultInjector,
    WeightFaultInjector,
    FaultStatistics,
)


class Trainer:
    """Trainer class for training and evaluating models.

    Orchestrates the training process using modular components:

    - OptimizerFactory: Creates optimizers from config
    - SchedulerFactory: Creates schedulers with warmup support
    - ExperimentManager: Handles checkpoints and experiment organization
    - MetricsLogger: Handles TensorBoard and console logging
    - ActivationFaultInjector: Optional fault injection for fault-aware training

    Features:
        - Progress bars with tqdm
        - Checkpoint saving/loading with meaningful experiment names
        - Learning rate warmup
        - Mixed precision training (AMP)
        - TensorBoard logging
        - Config saving for experiment reproducibility
        - Fault injection for fault-aware training (FAT)

    Example:
        ```python
        trainer = Trainer(model, train_loader, test_loader, config, device)
        trainer.train()
        ```

    Attributes:
        model: The model being trained.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation data.
        test_loader: DataLoader for test data.
        config: Full configuration dictionary.
        device: Compute device (cuda/mps/cpu).
        criterion: Loss function.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        use_amp: Whether AMP is enabled.
        experiment: Experiment manager for checkpoints.
        logger: Metrics logger for TensorBoard and console.
        fault_injector: Optional fault injector for FAT.
        fault_config: Fault injection configuration.
        fault_statistics: Fault injection statistics tracker.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        test_loader: DataLoader[Any],
        config: Dict[str, Any],
        device: torch.device,
        val_loader: Optional[DataLoader[Any]] = None,
        local_rank: Optional[int] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The model to train (will be moved to device).
            train_loader: DataLoader for training data.
            test_loader: DataLoader for test data.
            config: Full configuration dictionary.
            device: Compute device (cuda/mps/cpu).
            val_loader: Optional DataLoader for validation data.
            local_rank: Local rank for distributed training (optional).
        """
        self.local_rank: Optional[int] = local_rank
        self.is_distributed: bool = local_rank is not None

        self.model: nn.Module = model.to(device)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[local_rank])
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            self.rank = 0
            self.world_size = 1

        self.train_loader: DataLoader[Any] = train_loader
        self.val_loader: Optional[DataLoader[Any]] = val_loader
        self.test_loader: DataLoader[Any] = test_loader
        self.config: Dict[str, Any] = config
        self.device: torch.device = device

        brevitas_config: Dict[str, Any] = config.get("brevitas", {})
        if brevitas_config.get("ignore_missing_keys", False):
            try:
                from brevitas import config as brev_config

                brev_config.IGNORE_MISSING_KEYS = True
                if self.rank == 0 or not self.is_distributed:
                    print(
                        "Brevitas IGNORE_MISSING_KEYS enabled for float-to-quant loading"
                    )
            except ImportError:
                pass

        # Training state
        self.start_epoch: int = 0
        self.best_acc: float = 0.0
        self.total_epochs: int = config["training"]["epochs"]

        # Setup loss function
        self.criterion: nn.Module = LossFactory.create(config)

        # Setup optimizer and scheduler using factories
        self.optimizer: torch.optim.Optimizer = OptimizerFactory.create(
            self.model, config
        )

        self.scheduler: SchedulerType = SchedulerFactory.create(self.optimizer, config)

        # Mixed precision training (AMP)
        amp_config: Dict[str, Any] = config.get("amp", {})
        self.use_amp: bool = amp_config.get("enabled", False) and device.type == "cuda"
        self.scaler: Optional[GradScaler] = GradScaler() if self.use_amp else None

        # Setup experiment manager for checkpoints
        self.experiment: ExperimentManager = ExperimentManager.from_config(config)

        # Setup metrics logger
        self.logger: MetricsLogger = MetricsLogger.from_config(
            config,
            tensorboard_dir=self.experiment.get_tensorboard_dir(),
            experiment_dir=self.experiment.get_experiment_dir(),
        )

        # Progress bar settings
        progress_config: Dict[str, Any] = config.get("progress", {})
        self.show_progress: bool = progress_config.get("enabled", True)

        # Setup fault injection (if configured)
        self.act_fault_injector: Optional[ActivationFaultInjector] = None
        self.act_fault_config: Optional[FaultInjectionConfig] = None
        self.act_fault_statistics: Optional[FaultStatistics] = None

        self.weight_fault_injector: Optional[WeightFaultInjector] = None
        self.weight_fault_config: Optional[FaultInjectionConfig] = None
        self.weight_fault_statistics: Optional[FaultStatistics] = None

        self._setup_fault_injection(config)

        # Resume from checkpoint if specified (full training state)
        checkpoint_config: Dict[str, Any] = config.get("checkpoint", {})
        resume_path: Optional[str] = checkpoint_config.get("resume")
        if resume_path:
            self.start_epoch, self.best_acc = self.experiment.load_checkpoint(
                resume_path,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.device,
            )

        # Load model weights only (no training state) if specified
        load_weights_path: Optional[str] = checkpoint_config.get("load_weights")
        if load_weights_path:
            print("\n" + "=" * 60)
            print("Loading pretrained weights (no training state)")
            print("=" * 60)
            self._load_weights_only(load_weights_path)
            self.best_acc = 0.0

    def _setup_fault_injection(self, config: Dict[str, Any]) -> None:
        """Setup fault injection if configured.

        Supports both activation and weight fault injection independently.
        Each can be enabled/disabled separately in the config.

        Args:
            config: Full configuration dictionary.
        """
        # Setup activation fault injection
        act_config = config.get("activation_fault_injection", {})
        if act_config.get("enabled", False):
            self.act_fault_config = FaultInjectionConfig.from_dict(act_config)
            self.act_fault_injector = ActivationFaultInjector()
            self.model = self.act_fault_injector.inject(
                self.model, self.act_fault_config
            )

            if self.act_fault_config.verbose:
                # Print model architecture with injection layers
                for name, module in self.model.named_modules():
                    print(f"{name}: {module}")

            # Setup statistics tracking if enabled
            if self.act_fault_config.track_statistics:
                num_layers = self.act_fault_injector.get_num_layers(self.model)
                self.act_fault_statistics = FaultStatistics(num_layers=num_layers)
                self.act_fault_injector.set_statistics(
                    self.model, self.act_fault_statistics
                )

            # Log setup
            if self.act_fault_config.verbose:
                num_layers = self.act_fault_injector.get_num_layers(self.model)
                print(
                    f"Activation fault injection enabled: {num_layers} injection layers added"
                )
                print(f"  Probability: {self.act_fault_config.probability}%")
                print(f"  Injection type: {self.act_fault_config.injection_type}")
                print(f"  Apply during: {self.act_fault_config.apply_during}")

        # Setup weight fault injection
        weight_config = config.get("weight_fault_injection", {})
        if weight_config.get("enabled", False):
            self.weight_fault_config = FaultInjectionConfig.from_dict(weight_config)
            self.weight_fault_injector = WeightFaultInjector()
            self.model = self.weight_fault_injector.inject(
                self.model, self.weight_fault_config
            )

            if self.weight_fault_config.verbose:
                # Print model architecture with weight hooks
                for name, module in self.model.named_modules():
                    print(f"{name}: {module}")

            # Setup statistics tracking if enabled
            if self.weight_fault_config.track_statistics:
                num_layers = self.weight_fault_injector.get_num_layers(self.model)
                self.weight_fault_statistics = FaultStatistics(num_layers=num_layers)
                self.weight_fault_injector.set_statistics(
                    self.model, self.weight_fault_statistics
                )

            # Log setup
            if self.weight_fault_config.verbose:
                num_layers = self.weight_fault_injector.get_num_layers(self.model)
                print(
                    f"Weight fault injection enabled: {num_layers} injection hooks added"
                )
                print(f"  Probability: {self.weight_fault_config.probability}%")
                print(f"  Injection type: {self.weight_fault_config.injection_type}")
                print(f"  Apply during: {self.weight_fault_config.apply_during}")

    @property
    def has_validation(self) -> bool:
        """Check if validation loader is available.

        Returns:
            True if validation data is available.
        """
        return self.val_loader is not None

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number (for progress bar display).

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Set epoch for distributed sampler
        if self.is_distributed and hasattr(self.train_loader, "sampler"):
            self.train_loader.sampler.set_epoch(epoch)

        # Setup activation fault injection for this epoch
        if self.act_fault_injector is not None and self.act_fault_config is not None:
            # Set mode based on apply_during config
            apply_during = self.act_fault_config.apply_during
            if apply_during in ("train", "both"):
                self.act_fault_injector.set_enabled(self.model, True)
            else:
                self.act_fault_injector.set_enabled(self.model, False)

        # Setup weight fault injection for this epoch
        if (
            self.weight_fault_injector is not None
            and self.weight_fault_config is not None
        ):
            apply_during = self.weight_fault_config.apply_during
            if apply_during in ("train", "both"):
                self.weight_fault_injector.set_enabled(self.model, True)
            else:
                self.weight_fault_injector.set_enabled(self.model, False)

        # Create progress bar
        if self.show_progress and self.rank == 0:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            pbar = self.train_loader

        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.use_amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            if hasattr(self.model, "clip_weights"):
                self.model.clip_weights(-1, 1)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            if self.show_progress and self.rank == 0:
                current_acc = 100.0 * correct / total
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.2f}%"}
                )

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader[Any], desc: str = "Evaluating"
    ) -> Tuple[float, float]:
        """Evaluate model on a given dataloader.

        Args:
            loader: DataLoader to evaluate on.
            desc: Description for progress bar.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # Setup fault injection for evaluation
        if self.act_fault_injector is not None and self.act_fault_config is not None:
            apply_during = self.act_fault_config.apply_during
            if apply_during in ("eval", "both"):
                self.act_fault_injector.set_enabled(self.model, True)
            else:
                self.act_fault_injector.set_enabled(self.model, False)

        # Setup weight fault injection for evaluation
        if (
            self.weight_fault_injector is not None
            and self.weight_fault_config is not None
        ):
            apply_during = self.weight_fault_config.apply_during
            if apply_during in ("eval", "both"):
                self.weight_fault_injector.set_enabled(self.model, True)
            else:
                self.weight_fault_injector.set_enabled(self.model, False)

        # Create progress bar for evaluation
        if self.show_progress and self.rank == 0:
            pbar = tqdm(
                loader,
                desc=desc,
                leave=False,
                dynamic_ncols=True,
            )
        else:
            pbar = loader

        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Use AMP for inference too (optional, for consistency)
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """Evaluate model on validation set.

        Returns:
            Tuple of (average_loss, accuracy).

        Raises:
            ValueError: If no validation set is available.
        """
        if self.val_loader is None:
            raise ValueError("No validation set available")
        return self.evaluate(self.val_loader, desc="Validating")

    def test(self) -> Tuple[float, float]:
        """Evaluate model on test set.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        return self.evaluate(self.test_loader, desc="Testing")

    def train(self) -> None:
        """Run the full training loop

        Executes training for the configured number of epochs,
        handling validation/test evaluation, checkpointing, and logging.
        """
        epochs: int = self.total_epochs
        model_name: str = self.config["model"]["name"]
        eval_name: str = "Val" if self.has_validation else "Test"

        # Test evaluation frequency (only when using validation set)
        training_config: Dict[str, Any] = self.config.get("training", {})
        test_frequency: int = training_config.get("test_frequency", 10)

        # Log training start
        if self.rank == 0:
            self.logger.log_training_start(
                model_name=model_name,
                epochs=epochs,
                device=str(self.device),
                has_validation=self.has_validation,
                use_amp=self.use_amp,
                experiment_dir=self.experiment.get_experiment_dir(),
            )

            # Log fault injection info
            if (
                self.act_fault_injector is not None
                and self.act_fault_config is not None
            ):
                num_layers = self.act_fault_injector.get_num_layers(self.model)
                print(
                    f"Activation fault injection: {num_layers} layers, "
                    f"prob={self.act_fault_config.probability}%, "
                    f"mode=full_model, "
                    f"type={self.act_fault_config.injection_type}"
                )

            if (
                self.weight_fault_injector is not None
                and self.weight_fault_config is not None
            ):
                num_layers = self.weight_fault_injector.get_num_layers(self.model)
                print(
                    f"Weight fault injection: {num_layers} hooks, "
                    f"prob={self.weight_fault_config.probability}%, "
                    f"mode=full_model, "
                    f"type={self.weight_fault_config.injection_type}"
                )

        for epoch in range(self.start_epoch, epochs):
            train_loss, train_acc = self.train_epoch(epoch)

            if self.scheduler is not None:
                lr = self.scheduler.get_last_lr()[0]
                self.scheduler.step()
            else:
                lr = self.config["optimizer"]["learning_rate"]
            # Evaluate on validation or test set
            if self.has_validation:
                eval_loss, eval_acc = self.validate()
            else:
                eval_loss, eval_acc = self.test()

            # Check if this is the best model
            is_best = eval_acc > self.best_acc
            if is_best:
                self.best_acc = eval_acc

            # Evaluate on test set when:
            # 1. We find a new best model (when using validation), OR
            # 2. It's a periodic test evaluation epoch (when using validation)
            test_loss, test_acc = None, None
            if self.has_validation:
                is_periodic_test = test_frequency > 0 and (
                    (epoch + 1) % test_frequency == 0 or epoch == epochs - 1
                )
                if is_best or is_periodic_test:
                    test_loss, test_acc = self.test()

            # Log epoch metrics
            if self.rank == 0:
                self.logger.log_epoch(
                    epoch=epoch,
                    total_epochs=epochs,
                    lr=lr,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    eval_loss=eval_loss,
                    eval_acc=eval_acc,
                    eval_name=eval_name,
                    test_loss=test_loss,
                    test_acc=test_acc,
                )

            # Save checkpoint (only on rank 0)
            if self.rank == 0 and self.experiment.should_save(epoch, is_best):
                model_to_save = self.model.module if self.is_distributed else self.model
                self.experiment.save_checkpoint(
                    epoch=epoch,
                    model=model_to_save,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    best_acc=self.best_acc,
                    current_acc=eval_acc,
                    scaler=self.scaler,
                    is_best=is_best,
                    test_acc=test_acc if is_best and self.has_validation else None,
                    act_fault_injector=self.act_fault_injector,
                    weight_fault_injector=self.weight_fault_injector,
                    act_fault_config=self.act_fault_config,
                    weight_fault_config=self.weight_fault_config,
                )

                # Log best model to file
                if is_best:
                    self.logger.log_best_model(
                        epoch=epoch,
                        val_acc=eval_acc,
                        test_acc=test_acc if self.has_validation else None,
                    )

        # Final evaluation on test set (if we used validation during training)
        if self.has_validation and self.rank == 0:
            print("\nRunning final evaluation on test set...")
            final_test_loss, final_test_acc = self.test()
            self.logger.log_final_test(final_test_loss, final_test_acc, epochs)

        # Print fault injection statistics
        if self.rank == 0:
            if self.act_fault_statistics is not None:
                print("\n" + "=" * 60)
                print("Activation Fault Injection Statistics:")
                print("=" * 60)
                self.act_fault_statistics.print_report()

                # Save statistics to experiment directory
                experiment_dir = self.experiment.get_experiment_dir()
                if experiment_dir is not None:
                    import os

                    stats_path = os.path.join(
                        experiment_dir, "activation_fault_injection_stats.json"
                    )
                    self.act_fault_statistics.save_to_file(stats_path)
                    print(
                        f"Activation fault injection statistics saved to: {stats_path}"
                    )

            if self.weight_fault_statistics is not None:
                print("\n" + "=" * 60)
                print("Weight Fault Injection Statistics:")
                print("=" * 60)
                self.weight_fault_statistics.print_report()

                # Save statistics to experiment directory
                experiment_dir = self.experiment.get_experiment_dir()
                if experiment_dir is not None:
                    import os

                    stats_path = os.path.join(
                        experiment_dir, "weight_fault_injection_stats.json"
                    )
                    self.weight_fault_statistics.save_to_file(stats_path)
                    print(f"Weight fault injection statistics saved to: {stats_path}")

            # Log completion
            self.logger.log_training_complete(self.best_acc, eval_name)

            experiment_dir = self.experiment.get_experiment_dir()
            if experiment_dir is not None:
                print(f"Results saved to: {experiment_dir}")

            # Close logger
            self.logger.close()

    def _load_weights_only(self, checkpoint_path: str) -> None:
        """Load only model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        from pathlib import Path

        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.is_absolute():
            if self.experiment.experiment_dir is not None:
                ckpt_file = self.experiment.experiment_dir / checkpoint_path
            else:
                raise FileNotFoundError(
                    f"Cannot resolve relative path: {checkpoint_path}"
                )

        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        print(f"Loading weights from: {ckpt_file}")

        ckpt = torch.load(str(ckpt_file), map_location="cpu")

        model_to_load = self.model.module if self.is_distributed else self.model
        missing_keys, unexpected_keys = model_to_load.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )

        if missing_keys:
            print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(
                f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}..."
            )

        self.model = self.model.to(self.device)
        print(f"  Loaded model weights (optimizer/scheduler not restored)")
