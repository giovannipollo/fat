"""Trainer class for training and evaluating models.

Provides a complete training loop with support for:

- Progress bars with tqdm
- Checkpoint saving/loading
- Learning rate warmup
- Mixed precision training (AMP)
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
from pathlib import Path
from brevitas import config as brev_config

from .optimizer import OptimizerFactory
from .scheduler import SchedulerFactory, SchedulerType
from .experiment import ExperimentManager
from .logging import MetricsLogger
from .loss import LossFactory
from .fault_injection import (
    FaultInjectionConfig,
    ActivationFaultInjector,
    WeightFaultInjector,
)


class Trainer:
    """Trainer class for training and evaluating models.

    Orchestrates the training process using modular components:

    - OptimizerFactory: Creates optimizers from config
    - SchedulerFactory: Creates schedulers with warmup support
    - ExperimentManager: Handles checkpoints and experiment organization
    - MetricsLogger: Handles console and file logging
    - ActivationFaultInjector: Optional fault injection for fault-aware training

    Features:
        - Progress bars with tqdm
        - Checkpoint saving/loading with meaningful experiment names
        - Learning rate warmup
        - Mixed precision training (AMP)
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
        logger: Metrics logger for console and file.
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
            brev_config.IGNORE_MISSING_KEYS = True
            if self.rank == 0 or not self.is_distributed:
                print("Brevitas IGNORE_MISSING_KEYS enabled for float-to-quant loading")

        # Training state
        self.start_epoch: int = 0
        self.best_val_acc: float = 0.0
        self.best_test_acc: float = 0.0
        self.total_epochs: int = config["training"]["epochs"]

        # Setup loss function
        self.criterion: nn.Module = LossFactory.create(config)

        self.optimizer: torch.optim.Optimizer = OptimizerFactory.create(
            self.model, config
        )

        _act_fault_warmup: int = config.get("activation_fault_injection", {}).get(
            "warmup_epochs", 0
        )
        _weight_fault_warmup: int = config.get("weight_fault_injection", {}).get(
            "warmup_epochs", 0
        )
        _fault_warmup: int = max(_act_fault_warmup, _weight_fault_warmup)

        self.scheduler: SchedulerType = SchedulerFactory.create(
            self.optimizer, config, fault_warmup_epochs=_fault_warmup
        )

        # Mixed precision training (AMP)
        amp_config: Dict[str, Any] = config.get("amp", {})
        self.use_amp: bool = amp_config.get("enabled", False) and device.type == "cuda"
        self.scaler: Optional[GradScaler] = GradScaler() if self.use_amp else None

        # Setup experiment manager for checkpoints
        self.experiment: ExperimentManager = ExperimentManager.from_config(config)

        # Setup metrics logger
        self.logger: MetricsLogger = MetricsLogger.from_config(
            config,
            experiment_dir=self.experiment.get_experiment_dir(),
        )

        # Progress bar settings
        progress_config: Dict[str, Any] = config.get("progress", {})
        self.show_progress: bool = progress_config.get("enabled", True)

        # Setup fault injection (if configured)
        self.act_fault_injector: Optional[ActivationFaultInjector] = None
        self.act_fault_config: Optional[FaultInjectionConfig] = None

        self.weight_fault_injector: Optional[WeightFaultInjector] = None
        self.weight_fault_config: Optional[FaultInjectionConfig] = None

        checkpoint_config: Dict[str, Any] = config.get("checkpoint", {})

        # Load model weights only (no training state) if specified
        load_weights_path: Optional[str] = checkpoint_config.get("load_weights")
        if load_weights_path:
            print("\n" + "=" * 60)
            print("Loading pretrained weights")
            print("=" * 60)
            model_to_load = self.model.module if self.is_distributed else self.model
            self.experiment.load_weights(
                load_weights_path, model_to_load, self.device, strict=False
            )

        self._setup_fault_injection(config)

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

            # Log setup
            if self.weight_fault_config.verbose:
                num_layers = self.weight_fault_injector.get_num_layers(self.model)
                print(
                    f"Weight fault injection enabled: {num_layers} injection hooks added"
                )
                print(f"  Probability: {self.weight_fault_config.probability}%")
                print(f"  Injection type: {self.weight_fault_config.injection_type}")
                print(f"  Apply during: {self.weight_fault_config.apply_during}")

    def _apply_fault_warmup(self, epoch: int) -> None:
        """Update fault injection probabilities for the current warmup epoch.

        Called once per epoch from the main training loop. When warmup_epochs > 0,
        the effective probability is computed by FaultInjectionConfig.get_warmup_probability()
        and pushed to all injection layers via update_probability(). When warmup is
        complete or not configured, this method is a no-op.

        Args:
            epoch: Current epoch index (0-based).
        """
        if (
            self.act_fault_injector is not None
            and self.act_fault_config is not None
            and self.act_fault_config.warmup_epochs > 0
        ):
            p = self.act_fault_config.get_warmup_probability(epoch)
            self.act_fault_injector.update_probability(self.model, p)

        if (
            self.weight_fault_injector is not None
            and self.weight_fault_config is not None
            and self.weight_fault_config.warmup_epochs > 0
        ):
            p = self.weight_fault_config.get_warmup_probability(epoch)
            self.weight_fault_injector.update_probability(self.model, p)

    def _fault_warmup_epochs(self) -> int:
        """Return the number of epochs over which fault probability is being warmed up.

        This is the maximum warmup_epochs across both active injector configs.
        Returns 0 if no fault warmup is configured.
        """
        warmup = 0
        if self.act_fault_config is not None:
            warmup = max(warmup, self.act_fault_config.warmup_epochs)
        if self.weight_fault_config is not None:
            warmup = max(warmup, self.weight_fault_config.warmup_epochs)
        return warmup

    @property
    def has_validation(self) -> bool:
        """Check if validation loader is available.

        Returns:
            True if validation data is available.
        """
        return self.val_loader is not None

    def train_epoch(
        self, epoch: int, is_faulty_epoch: bool = True
    ) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number (for progress bar display).
            is_faulty_epoch: Whether this epoch should have fault injection enabled.

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
            apply_during = self.act_fault_config.apply_during
            inject_this_epoch = is_faulty_epoch and apply_during in ("train", "both")
            if self.act_fault_config.step_interval == 1:
                self.act_fault_injector.set_enabled(self.model, inject_this_epoch)
            else:
                self.act_fault_injector.set_enabled(self.model, False)

        # Setup weight fault injection for this epoch
        if (
            self.weight_fault_injector is not None
            and self.weight_fault_config is not None
        ):
            apply_during = self.weight_fault_config.apply_during
            inject_this_epoch = is_faulty_epoch and apply_during in ("train", "both")
            if self.weight_fault_config.step_interval == 1:
                self.weight_fault_injector.set_enabled(self.model, inject_this_epoch)
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

        for step, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Per-step fault injection gate (only relevant when step_interval > 1)
            if is_faulty_epoch:
                if (
                    self.act_fault_injector is not None
                    and self.act_fault_config is not None
                ):
                    if self.act_fault_config.step_interval > 1:
                        apply_during = self.act_fault_config.apply_during
                        inject_this_step = apply_during in (
                            "train",
                            "both",
                        ) and self.act_fault_config.is_faulty_step(step)
                        self.act_fault_injector.set_enabled(
                            self.model, inject_this_step
                        )

                if (
                    self.weight_fault_injector is not None
                    and self.weight_fault_config is not None
                ):
                    if self.weight_fault_config.step_interval > 1:
                        apply_during = self.weight_fault_config.apply_during
                        inject_this_step = apply_during in (
                            "train",
                            "both",
                        ) and self.weight_fault_config.is_faulty_step(step)
                        self.weight_fault_injector.set_enabled(
                            self.model, inject_this_step
                        )

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
        return self.evaluate(self.val_loader, desc="Validation")

    def test(self) -> Tuple[float, float]:
        """Evaluate model on test set.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        if self.test_loader is None:
            raise ValueError("No test set available")
        return self.evaluate(self.test_loader, desc="Testing")

    def train(self) -> None:
        """Run the full training loop

        Executes training for the configured number of epochs,
        handling validation/test evaluation, checkpointing, and logging.
        """
        epochs: int = self.total_epochs
        model_name: str = self.config["model"]["name"]

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
                warmup_info = ""
                if self.act_fault_config.warmup_epochs > 0:
                    warmup_info = (
                        f", warmup_epochs={self.act_fault_config.warmup_epochs}, "
                        f"warmup_schedule={self.act_fault_config.warmup_schedule}"
                    )
                print(
                    f"Activation fault injection: {num_layers} layers, "
                    f"prob={self.act_fault_config.probability}%{warmup_info}, "
                    f"epoch_interval={self.act_fault_config.epoch_interval}, "
                    f"step_interval={self.act_fault_config.step_interval}, "
                    f"type={self.act_fault_config.injection_type}"
                )

            if (
                self.weight_fault_injector is not None
                and self.weight_fault_config is not None
            ):
                num_layers = self.weight_fault_injector.get_num_layers(self.model)
                warmup_info = ""
                if self.weight_fault_config.warmup_epochs > 0:
                    warmup_info = (
                        f", warmup_epochs={self.weight_fault_config.warmup_epochs}, "
                        f"warmup_schedule={self.weight_fault_config.warmup_schedule}"
                    )
                print(
                    f"Weight fault injection: {num_layers} hooks, "
                    f"prob={self.weight_fault_config.probability}%{warmup_info}, "
                    f"epoch_interval={self.weight_fault_config.epoch_interval}, "
                    f"step_interval={self.weight_fault_config.step_interval}, "
                    f"type={self.weight_fault_config.injection_type}"
                )

        for epoch in range(self.start_epoch, epochs):
            self._apply_fault_warmup(epoch)

            act_faulty_epoch = (
                self.act_fault_injector is not None
                and self.act_fault_config is not None
                and self.act_fault_config.should_inject_during_training()
                and self.act_fault_config.is_faulty_epoch(epoch)
            )
            weight_faulty_epoch = (
                self.weight_fault_injector is not None
                and self.weight_fault_config is not None
                and self.weight_fault_config.should_inject_during_training()
                and self.weight_fault_config.is_faulty_epoch(epoch)
            )
            is_faulty_epoch = act_faulty_epoch or weight_faulty_epoch

            train_loss, train_acc = self.train_epoch(
                epoch=epoch, is_faulty_epoch=is_faulty_epoch
            )

            fault_warmup_epochs = self._fault_warmup_epochs()
            in_fault_warmup = epoch < fault_warmup_epochs

            if self.scheduler is not None:
                lr = self.scheduler.get_last_lr()[0]
                if not in_fault_warmup:
                    self.scheduler.step()
            else:
                lr = self.config["optimizer"]["learning_rate"]

            # Evaluate on validation and/or test set
            val_loss: Optional[float] = None
            val_acc: Optional[float] = None
            test_loss: Optional[float] = None
            test_acc: Optional[float] = None

            if self.has_validation:
                val_loss, val_acc = self.validate()
                # Check if this is the best model
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc

                # Check if we should run test evaluation this epoch based on test_frequency
                is_periodic_test = False
                if test_frequency > 0 and (epoch + 1) % test_frequency == 0:
                    is_periodic_test = True

                # Check if this is the last epoch (always test at the end)
                is_last_epoch = False
                if epoch == epochs - 1:
                    is_last_epoch = True

                # If this is the best model so far, or if it's a periodic test epoch, or if it's the last epoch, run test evaluation
                if is_best or is_periodic_test or is_last_epoch:
                    test_loss, test_acc = self.test()
                    # Track best test accuracy
                    if test_acc > self.best_test_acc:
                        self.best_test_acc = test_acc
            else:
                # No validation set - use test set for monitoring
                test_loss, test_acc = self.test()
                # Check if this is the best model
                is_best = test_acc > self.best_test_acc
                if is_best:
                    self.best_test_acc = test_acc

            # Log epoch metrics
            if self.rank == 0:
                self.logger.log_epoch(
                    epoch=epoch,
                    total_epochs=epochs,
                    lr=lr,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    test_loss=test_loss,
                    test_acc=test_acc,
                    is_best=is_best,
                )

            # Save checkpoint
            if self.rank == 0 and self.experiment.should_save(
                epoch=epoch, is_best=is_best
            ):
                model_to_save = self.model.module if self.is_distributed else self.model
                self.experiment.save_checkpoint(
                    epoch=epoch,
                    model=model_to_save,
                    val_acc=val_acc,
                    best_val_acc=self.best_val_acc,
                    test_acc=test_acc,
                    best_test_acc=self.best_test_acc,
                    is_best=is_best,
                    act_fault_injector=self.act_fault_injector,
                    weight_fault_injector=self.weight_fault_injector,
                    act_fault_config=self.act_fault_config,
                    weight_fault_config=self.weight_fault_config,
                )

                # Log best model to file
                if is_best:
                    self.logger.log_best_model(
                        epoch=epoch,
                        val_acc=val_acc,
                        test_acc=test_acc,
                    )

        # Final evaluation on test set (if we used validation during training)
        if self.has_validation and self.rank == 0:
            print("\nRunning final evaluation on test set...")
            final_test_loss, final_test_acc = self.test()
            self.logger.log_final_test(final_test_loss, final_test_acc, epochs)

        # Log completion
        if self.rank == 0:
            if self.has_validation:
                monitor_name = "validation"
                best_acc = self.best_val_acc
            else:
                monitor_name = "test"
                best_acc = self.best_test_acc
            self.logger.log_training_complete(
                best_acc=best_acc, monitor_name=monitor_name
            )

            experiment_dir = self.experiment.get_experiment_dir()
            if experiment_dir is not None:
                print(f"Results saved to: {experiment_dir}")

            # Close logger
            self.logger.close()
