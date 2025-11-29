"""!
@file utils/scheduler.py
@brief Learning rate scheduler factory and warmup wrapper.

@details Provides factory pattern for creating PyTorch LR schedulers
with optional linear warmup support.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch.optim as optim


class WarmupScheduler:
    """!
    @brief Learning rate warmup scheduler that wraps another scheduler.
    
    @details Implements linear warmup for the first N epochs, then
    delegates to the main scheduler. Compatible with PyTorch's
    scheduler interface (step(), get_last_lr(), state_dict(), etc.).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        main_scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    ):
        """!
        @brief Initialize warmup scheduler.
        
        @param optimizer The optimizer being scheduled
        @param warmup_epochs Number of warmup epochs
        @param main_scheduler Optional scheduler to use after warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs: int = warmup_epochs
        self.main_scheduler: Optional[optim.lr_scheduler.LRScheduler] = main_scheduler
        self.current_epoch: int = 0

        # Store initial learning rates
        self.base_lrs: List[float] = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        """!
        @brief Advance the scheduler by one epoch.
        
        @details During warmup, linearly scales LR from 0 to base_lr.
        After warmup, delegates to the main scheduler if present.
        """
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            warmup_factor: float = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.base_lrs[i] * warmup_factor
        elif self.main_scheduler is not None:
            self.main_scheduler.step()

    def get_last_lr(self) -> List[float]:
        """!
        @brief Return last computed learning rate.
        
        @return List of learning rates for each parameter group
        """
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """!
        @brief Return scheduler state as a dictionary.
        
        @return State dictionary for checkpointing
        """
        state: Dict[str, Any] = {
            "current_epoch": self.current_epoch,
            "base_lrs": self.base_lrs,
        }
        if self.main_scheduler is not None:
            state["main_scheduler"] = self.main_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """!
        @brief Load scheduler state from dictionary.
        
        @param state_dict State dictionary from state_dict()
        """
        self.current_epoch = state_dict["current_epoch"]
        self.base_lrs = state_dict["base_lrs"]
        if self.main_scheduler is not None and "main_scheduler" in state_dict:
            self.main_scheduler.load_state_dict(state_dict["main_scheduler"])


## @var SchedulerType
#  @brief Type alias for scheduler types (LRScheduler, WarmupScheduler, or None)
SchedulerType = Union[
    optim.lr_scheduler.LRScheduler,
    WarmupScheduler,
    None,
]


class SchedulerFactory:
    """!
    @brief Factory class for creating learning rate schedulers from configuration.
    
    @details Supports various PyTorch schedulers with optional warmup wrapper.
    
    @par Supported Schedulers
    - cosine: Cosine annealing
    - step: Step decay
    - multistep: Multi-step decay
    - exponential: Exponential decay
    - plateau: Reduce on plateau
    - none: No scheduling
    """

    ## @var SCHEDULERS
    #  @brief Registry of available scheduler classes
    SCHEDULERS: Dict[str, Optional[type]] = {
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
        config: Dict[str, Any],
    ) -> SchedulerType:
        """!
        @brief Create a learning rate scheduler from configuration.
        
        @param optimizer The optimizer to schedule
        @param config Full configuration dictionary
        @return Configured scheduler (possibly wrapped with warmup), or None
        @throws ValueError If the scheduler name is unknown
        """
        sched_config: Dict[str, Any] = config.get("scheduler", {})
        sched_name: str = sched_config.get("name", "cosine").lower()
        warmup_epochs: int = sched_config.get("warmup_epochs", 0)
        total_epochs: int = config["training"]["epochs"]

        if sched_name not in cls.SCHEDULERS:
            available: List[str] = list(cls.SCHEDULERS.keys())
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
        sched_config: Dict[str, Any],
        total_epochs: int,
        warmup_epochs: int,
    ) -> Optional[optim.lr_scheduler.LRScheduler]:
        """!
        @brief Create the main scheduler without warmup wrapper.
        
        @param optimizer The optimizer to schedule
        @param name Scheduler name
        @param sched_config Scheduler configuration section
        @param total_epochs Total training epochs
        @param warmup_epochs Number of warmup epochs (for T_max adjustment)
        @return Configured scheduler or None
        """
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
    def available_schedulers(cls) -> List[str]:
        """!
        @brief Get list of available scheduler names.
        
        @return List of supported scheduler name strings
        """
        return list(cls.SCHEDULERS.keys())
