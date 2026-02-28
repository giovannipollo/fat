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
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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
from .phase_config import PhaseConfig, parse_phases


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

        # Parse phases (phases key is required)
        self.phases: List[PhaseConfig] = parse_phases(config)
        self.total_epochs: int = sum(p.epochs for p in self.phases)

        # Per-phase setup deferred to _setup_phase()
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[SchedulerType] = None

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

        self.weight_fault_injector: Optional[WeightFaultInjector] = None
        self.weight_fault_config: Optional[FaultInjectionConfig] = None

        checkpoint_config: Dict[str, Any] = config.get("checkpoint", {})

        # Load model weights only (no training state) if specified
        load_weights_path: Optional[str] = checkpoint_config.get("load_weights")
        if load_weights_path:
            print("\n" + "=" * 60)
            print("Loading pretrained weights (no training state)")
            print("=" * 60)
            self._load_weights_only(load_weights_path)
            self.best_acc = 0.0

        # Resume from checkpoint if specified (full training state)
        # Note: optimizer/scheduler are created per-phase, so resume happens
        # after we set up the appropriate phase in train()
        self._resume_path: Optional[str] = checkpoint_config.get("resume")

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

    def _teardown_fault_injection(self) -> None:
        """Remove any active fault injection from the model."""
        if self.act_fault_injector is not None:
            self.model = self.act_fault_injector.remove(self.model)
            self.act_fault_injector = None
            self.act_fault_config = None

        if self.weight_fault_injector is not None:
            self.model = self.weight_fault_injector.remove(self.model)
            self.weight_fault_injector = None
            self.weight_fault_config = None

    def _setup_phase(self, phase: PhaseConfig) -> None:
        """Configure optimizer, scheduler, loss, and fault injection for a phase."""
        flat = phase.to_flat_config(self.config)

        self.criterion = LossFactory.create(flat)

        self.optimizer = OptimizerFactory.create(self.model, flat)

        act_warmup = phase.activation_fault_injection.get("warmup_epochs", 0)
        wgt_warmup = phase.weight_fault_injection.get("warmup_epochs", 0)
        fault_warmup = max(act_warmup, wgt_warmup)

        self.scheduler = SchedulerFactory.create(
            self.optimizer, flat, fault_warmup_epochs=fault_warmup
        )

        self._teardown_fault_injection()
        self._setup_fault_injection(flat)

    def _log_phase_start(self, phase: PhaseConfig) -> None:
        """Log the start of a training phase."""
        print("\n" + "=" * 60)
        print(f"Phase: {phase.name} (index={phase.phase_index})")
        print(f"  Epochs: {phase.epochs} (global {phase.global_epoch_offset} - {phase.global_epoch_offset + phase.epochs - 1})")
        print(f"  Batch size: {phase.batch_size}")
        print(f"  Optimizer: {phase.optimizer.get('name', 'unknown')} (lr={phase.optimizer.get('learning_rate')})")
        print(f"  Scheduler: {phase.scheduler.get('name', 'unknown')}")
        print(f"  Loss: {phase.loss.get('name', 'unknown')}")
        if phase.has_fault_injection:
            print(f"  Fault injection: enabled")
        print("=" * 60)

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
        return self.evaluate(self.val_loader, desc="Validating")

    def test(self) -> Tuple[float, float]:
        """Evaluate model on test set.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        return self.evaluate(self.test_loader, desc="Testing")

    def train(self) -> None:
        """Run the full training loop across all phases."""
        epochs: int = self.total_epochs
        model_name: str = self.config["model"]["name"]
        eval_name: str = "Val" if self.has_validation else "Test"

        test_frequency: int = 10

        if self.rank == 0:
            self.logger.log_training_start(
                model_name=model_name,
                epochs=epochs,
                device=str(self.device),
                has_validation=self.has_validation,
                use_amp=self.use_amp,
                experiment_dir=self.experiment.get_experiment_dir(),
            )
            print(f"\nTraining will run {len(self.phases)} phase(s):")
            for p in self.phases:
                print(f"  - {p.name}: {p.epochs} epochs")

        global_epoch = self.start_epoch

        for phase in self.phases:
            phase_end = phase.global_epoch_offset + phase.epochs
            if global_epoch >= phase_end:
                continue

            self._setup_phase(phase)

            if self.rank == 0:
                self._log_phase_start(phase)

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

            if self._resume_path is not None and global_epoch == self.start_epoch:
                self.start_epoch, self.best_acc = self.experiment.load_checkpoint(
                    self._resume_path,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    self.device,
                )
                global_epoch = self.start_epoch
                self._resume_path = None

            local_start = max(0, global_epoch - phase.global_epoch_offset)

            for local_epoch in range(local_start, phase.epochs):
                global_epoch = phase.global_epoch_offset + local_epoch

                self._apply_fault_warmup(local_epoch)

                act_faulty_epoch = (
                    self.act_fault_injector is not None
                    and self.act_fault_config is not None
                    and self.act_fault_config.should_inject_during_training()
                    and self.act_fault_config.is_faulty_epoch(local_epoch)
                )
                weight_faulty_epoch = (
                    self.weight_fault_injector is not None
                    and self.weight_fault_config is not None
                    and self.weight_fault_config.should_inject_during_training()
                    and self.weight_fault_config.is_faulty_epoch(local_epoch)
                )
                is_faulty_epoch = act_faulty_epoch or weight_faulty_epoch

                train_loss, train_acc = self.train_epoch(
                    epoch=global_epoch, is_faulty_epoch=is_faulty_epoch
                )

                fault_warmup_epochs = self._fault_warmup_epochs()
                in_fault_warmup = local_epoch < fault_warmup_epochs

                if self.scheduler is not None:
                    lr = self.scheduler.get_last_lr()[0]
                    if not in_fault_warmup:
                        self.scheduler.step()
                else:
                    lr = phase.optimizer.get("learning_rate", 0.0)

                if self.has_validation:
                    eval_loss, eval_acc = self.validate()
                else:
                    eval_loss, eval_acc = self.test()

                is_best = eval_acc > self.best_acc
                if is_best:
                    self.best_acc = eval_acc

                test_loss, test_acc = None, None
                if self.has_validation:
                    is_periodic_test = test_frequency > 0 and (
                        (global_epoch + 1) % test_frequency == 0 or global_epoch == epochs - 1
                    )
                    if is_best or is_periodic_test:
                        test_loss, test_acc = self.test()

                if self.rank == 0:
                    self.logger.log_epoch(
                        epoch=global_epoch,
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

                if self.rank == 0 and self.experiment.should_save(global_epoch, is_best):
                    model_to_save = self.model.module if self.is_distributed else self.model
                    self.experiment.save_checkpoint(
                        epoch=global_epoch,
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
                        phase_index=phase.phase_index,
                        phase_name=phase.name,
                        phase_local_epoch=local_epoch,
                        total_phases=len(self.phases),
                    )

                    if is_best:
                        self.logger.log_best_model(
                            epoch=global_epoch,
                            val_acc=eval_acc,
                            test_acc=test_acc if self.has_validation else None,
                        )

            self._teardown_fault_injection()

        if self.has_validation and self.rank == 0:
            print("\nRunning final evaluation on test set...")
            final_test_loss, final_test_acc = self.test()
            self.logger.log_final_test(final_test_loss, final_test_acc, epochs)

        if self.rank == 0:
            self.logger.log_training_complete(self.best_acc, eval_name)

            experiment_dir = self.experiment.get_experiment_dir()
            if experiment_dir is not None:
                print(f"Results saved to: {experiment_dir}")

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
