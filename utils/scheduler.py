"""Learning rate scheduler factory and warmup wrapper.

Provides factory pattern for creating PyTorch LR schedulers
with optional linear warmup support.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch.optim as optim


class WarmupScheduler:
    """Learning rate warmup scheduler that wraps another scheduler.

    Implements linear warmup for the first N epochs, then
    delegates to the main scheduler. Compatible with PyTorch's
    scheduler interface (step(), get_last_lr(), state_dict(), etc.).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        main_scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    ):
        """Initialize warmup scheduler.

        Args:
            optimizer: The optimizer being scheduled.
            warmup_epochs: Number of warmup epochs.
            main_scheduler: Optional scheduler to use after warmup.
        """
        self.optimizer = optimizer
        self.warmup_epochs: int = warmup_epochs
        self.main_scheduler: Optional[optim.lr_scheduler.LRScheduler] = main_scheduler

        # Store initial learning rates (target LR after warmup)
        self.base_lrs: List[float] = [group["lr"] for group in optimizer.param_groups]
        
        # Set initial LR to first warmup value and start at epoch 1
        if warmup_epochs > 0:
            self.current_epoch: int = 1
            initial_warmup_factor = 1.0 / warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.base_lrs[i] * initial_warmup_factor
        else:
            self.current_epoch: int = 0

    def step(self) -> None:
        """Advance the scheduler by one epoch.

        During warmup, linearly scales LR from base_lr/warmup_epochs to base_lr.
        After warmup, delegates to the main scheduler if present.
        """
        if self.current_epoch < self.warmup_epochs:
            # Increment epoch counter
            self.current_epoch += 1
            # Linear warmup
            warmup_factor: float = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.base_lrs[i] * warmup_factor
        elif self.current_epoch == self.warmup_epochs:
            # Transition from warmup to main scheduler
            self.current_epoch += 1
            # Ensure LR is at base value before starting main scheduler
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.base_lrs[i]
        else:
            # After warmup, use main scheduler
            self.current_epoch += 1
            if self.main_scheduler is not None:
                self.main_scheduler.step()

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate.

        Returns:
            List of learning rates for each parameter group.
        """
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state as a dictionary.

        Returns:
            State dictionary for checkpointing.
        """
        state: Dict[str, Any] = {
            "current_epoch": self.current_epoch,
            "base_lrs": self.base_lrs,
        }
        if self.main_scheduler is not None:
            state["main_scheduler"] = self.main_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from dictionary.

        Args:
            state_dict: State dictionary from state_dict().
        """
        self.current_epoch = state_dict["current_epoch"]
        self.base_lrs = state_dict["base_lrs"]
        if self.main_scheduler is not None and "main_scheduler" in state_dict:
            self.main_scheduler.load_state_dict(state_dict["main_scheduler"])


SchedulerType = Union[
    optim.lr_scheduler.LRScheduler,
    WarmupScheduler,
    None,
]
"""Type alias for scheduler types (LRScheduler, WarmupScheduler, or None)."""


class SchedulerFactory:
    """Factory class for creating learning rate schedulers from configuration.

    Supports various PyTorch schedulers with optional warmup wrapper.

    Supported Schedulers:
        - cosine: Cosine annealing
        - step: Step decay
        - multistep: Multi-step decay
        - exponential: Exponential decay
        - plateau: Reduce on plateau
        - none: No scheduling
    """

    SCHEDULERS: Dict[str, Optional[type]] = {
        "cosine": optim.lr_scheduler.CosineAnnealingLR,
        "step": optim.lr_scheduler.StepLR,
        "multistep": optim.lr_scheduler.MultiStepLR,
        "exponential": optim.lr_scheduler.ExponentialLR,
        "plateau": optim.lr_scheduler.ReduceLROnPlateau,
        "none": None,
    }
    """Registry of available scheduler classes."""

    @classmethod
    def create(
        cls,
        optimizer: optim.Optimizer,
        config: Dict[str, Any],
        total_epochs: Optional[int] = None,
        fault_warmup_epochs: int = 0,
    ) -> SchedulerType:
        """Create a learning rate scheduler from configuration.

        Args:
            optimizer: The optimizer to schedule.
            config: Full configuration dictionary.
            total_epochs: Optional explicit epoch count. If provided, this is used instead of computing from config.
            fault_warmup_epochs: Number of epochs the LR scheduler will be frozen
                due to fault probability warmup. Subtracted from the cosine T_max
                default so the curve spans exactly the epochs the scheduler runs.

        Returns:
            Configured scheduler (possibly wrapped with warmup), or None.

        Raises:
            ValueError: If the scheduler name is unknown.
        """
        sched_config: Dict[str, Any] = config.get("scheduler", {})
        sched_name: str = sched_config.get("name", "cosine").lower()
        warmup_epochs: int = sched_config.get("warmup_epochs", 0)

        # Determine total epochs: use explicit value if provided, otherwise compute from config
        computed_epochs: int
        if total_epochs is not None:
            computed_epochs = total_epochs
        else:
            computed_epochs = config["training"]["epochs"]

        if sched_name not in cls.SCHEDULERS:
            available: List[str] = list(cls.SCHEDULERS.keys())
            raise ValueError(f"Unknown scheduler: {sched_name}. Available: {available}")

        # Create main scheduler
        main_scheduler = cls._create_main_scheduler(
            optimizer=optimizer,
            name=sched_name,
            sched_config=sched_config,
            total_epochs=computed_epochs,
            warmup_epochs=warmup_epochs,
            fault_warmup_epochs=fault_warmup_epochs,
        )

        # Wrap with warmup if requested
        if warmup_epochs > 0:
            return WarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,
                main_scheduler=main_scheduler,
            )

        return main_scheduler

    @classmethod
    def _create_main_scheduler(
        cls,
        optimizer: optim.Optimizer,
        name: str,
        sched_config: Dict[str, Any],
        total_epochs: int,
        warmup_epochs: int,
        fault_warmup_epochs: int = 0,
    ) -> Optional[optim.lr_scheduler.LRScheduler]:
        """Create the main scheduler without warmup wrapper.

        Args:
            optimizer: The optimizer to schedule.
            name: Scheduler name.
            sched_config: Scheduler configuration section.
            total_epochs: Total training epochs.
            warmup_epochs: Number of warmup epochs (for T_max adjustment).
            fault_warmup_epochs: Number of fault warmup epochs (for T_max adjustment).

        Returns:
            Configured scheduler or None.
        """
        if name == "none":
            return None

        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=sched_config.get(
                    "T_max",
                    total_epochs - warmup_epochs - fault_warmup_epochs,
                ),
                eta_min=sched_config.get("eta_min", 0),
            )

        if name == "step":
            return optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=sched_config.get("step_size", 30),
                gamma=sched_config.get("gamma", 0.1),
            )

        if name == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=sched_config.get("milestones", [30, 60, 90]),
                gamma=sched_config.get("gamma", 0.1),
            )

        if name == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=sched_config.get("gamma", 0.95),
            )

        if name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=sched_config.get("mode", "max"),
                factor=sched_config.get("factor", 0.1),
                patience=sched_config.get("patience", 10),
                min_lr=sched_config.get("min_lr", 1e-6),
            )

        # This shouldn't happen due to the check in create(), but just in case
        raise ValueError(f"Unknown scheduler: {name}")

    @classmethod
    def available_schedulers(cls) -> List[str]:
        """Get list of available scheduler names.

        Returns:
            List of supported scheduler name strings.
        """
        return list(cls.SCHEDULERS.keys())
