"""Scheduler factory for creating learning rate schedulers from configuration."""

import torch.optim as optim
from typing import Optional, Union


class WarmupScheduler:
    """Learning rate warmup scheduler that wraps another scheduler."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        main_scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.main_scheduler = main_scheduler
        self.current_epoch = 0

        # Store initial learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        """Advance the scheduler by one epoch."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.base_lrs[i] * warmup_factor
        elif self.main_scheduler is not None:
            self.main_scheduler.step()

    def get_last_lr(self) -> list:
        """Return last computed learning rate by the scheduler."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Return the state of the scheduler as a dict."""
        state = {
            "current_epoch": self.current_epoch,
            "base_lrs": self.base_lrs,
        }
        if self.main_scheduler is not None:
            state["main_scheduler"] = self.main_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict):
        """Load the scheduler state."""
        self.current_epoch = state_dict["current_epoch"]
        self.base_lrs = state_dict["base_lrs"]
        if self.main_scheduler is not None and "main_scheduler" in state_dict:
            self.main_scheduler.load_state_dict(state_dict["main_scheduler"])


# Type alias for scheduler types
SchedulerType = Union[
    optim.lr_scheduler.LRScheduler,
    WarmupScheduler,
    None,
]


class SchedulerFactory:
    """Factory class for creating learning rate schedulers from configuration."""

    SCHEDULERS = {
        "cosine": optim.lr_scheduler.CosineAnnealingLR,
        "step": optim.lr_scheduler.StepLR,
        "multistep": optim.lr_scheduler.MultiStepLR,
        "exponential": optim.lr_scheduler.ExponentialLR,
        "plateau": optim.lr_scheduler.ReduceLROnPlateau,
        "none": None,
    }

    @classmethod
    def create(
        cls,
        optimizer: optim.Optimizer,
        config: dict,
    ) -> SchedulerType:
        """
        Create a learning rate scheduler based on configuration.

        Args:
            optimizer: The optimizer to schedule
            config: Full configuration dictionary

        Returns:
            The configured scheduler (possibly wrapped with warmup), or None

        Raises:
            ValueError: If the scheduler name is unknown
        """
        sched_config = config.get("scheduler", {})
        sched_name = sched_config.get("name", "cosine").lower()
        warmup_epochs = sched_config.get("warmup_epochs", 0)
        total_epochs = config["training"]["epochs"]

        if sched_name not in cls.SCHEDULERS:
            available = list(cls.SCHEDULERS.keys())
            raise ValueError(f"Unknown scheduler: {sched_name}. Available: {available}")

        # Create main scheduler
        main_scheduler = cls._create_main_scheduler(
            optimizer, sched_name, sched_config, total_epochs, warmup_epochs
        )

        # Wrap with warmup if requested
        if warmup_epochs > 0:
            return WarmupScheduler(optimizer, warmup_epochs, main_scheduler)

        return main_scheduler

    @classmethod
    def _create_main_scheduler(
        cls,
        optimizer: optim.Optimizer,
        name: str,
        sched_config: dict,
        total_epochs: int,
        warmup_epochs: int,
    ) -> Optional[optim.lr_scheduler.LRScheduler]:
        """Create the main scheduler (without warmup wrapper)."""
        if name == "none":
            return None

        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_config.get("T_max", total_epochs - warmup_epochs),
                eta_min=sched_config.get("eta_min", 0),
            )

        if name == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_config.get("step_size", 30),
                gamma=sched_config.get("gamma", 0.1),
            )

        if name == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=sched_config.get("milestones", [30, 60, 90]),
                gamma=sched_config.get("gamma", 0.1),
            )

        if name == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=sched_config.get("gamma", 0.95),
            )

        if name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=sched_config.get("mode", "max"),
                factor=sched_config.get("factor", 0.1),
                patience=sched_config.get("patience", 10),
                min_lr=sched_config.get("min_lr", 1e-6),
            )

        # This shouldn't happen due to the check in create(), but just in case
        raise ValueError(f"Unknown scheduler: {name}")

    @classmethod
    def available_schedulers(cls) -> list[str]:
        """Return list of available scheduler names."""
        return list(cls.SCHEDULERS.keys())
