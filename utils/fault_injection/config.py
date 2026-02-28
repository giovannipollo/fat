"""Configuration for fault injection.

Provides a dataclass for managing fault injection parameters with
validation and YAML configuration support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math


@dataclass
class FaultInjectionConfig:
    """Configuration for fault injection (activations or weights).

    Attributes:
        enabled: Master switch for fault injection.
        target_type: Type of injection - "activation" or "weight".
        probability: Injection probability as percentage (0-100).
        injection_type: Type of fault - "random", "lsb_flip", "msb_flip", "full_flip".
        apply_during: When to inject - "train", "eval", or "both".
        target_layers: List of layer types to inject (context depends on target_type).
        track_statistics: Enable statistics tracking (RMSE, cosine similarity).
        verbose: Print injection details.
        epoch_interval: Inject faults only every N epochs (1 = every epoch, default).
        step_interval: Inject faults only every N steps within a faulty epoch
                       (1 = every step, default).
        warmup_epochs: Number of epochs over which to ramp the injection probability
                       from 0 to `probability`. 0 = no warmup (default).
        warmup_schedule: Shape of the warmup ramp. "linear" (default) or "cosine".
                         Only meaningful when warmup_epochs > 0.

    Example:
        ```python
        config = FaultInjectionConfig(
            enabled=True,
            target_type="activation",
            probability=5.0,
            injection_type="random",
            epoch_interval=2,
            step_interval=4,
        )

        # Or from YAML dict
        config = FaultInjectionConfig.from_dict(yaml_config["fault_injection"])
        ```
    """

    enabled: bool = False
    target_type: str = "activation"
    probability: float = 0.0
    injection_type: str = "random"
    apply_during: str = "eval"
    target_layers: List[str] = field(default_factory=list)
    track_statistics: bool = False
    verbose: bool = False
    epoch_interval: int = 1
    step_interval: int = 1
    warmup_epochs: int = 0
    warmup_schedule: str = "linear"

    _VALID_TARGET_TYPES: List[str] = field(
        default_factory=lambda: ["activation", "weight"],
        repr=False,
    )
    _VALID_INJECTION_TYPES: List[str] = field(
        default_factory=lambda: ["random", "lsb_flip", "msb_flip", "full_flip"],
        repr=False,
    )
    _VALID_APPLY_DURING: List[str] = field(
        default_factory=lambda: ["train", "eval", "both"],
        repr=False,
    )
    _VALID_ACTIVATION_LAYERS: List[str] = field(
        default_factory=lambda: [
            "QuantIdentity",
            "QuantReLU",
            "QuantHardTanh",
            "QuantConv2d",
            "QuantLinear",
        ],
        repr=False,
    )
    _VALID_WEIGHT_LAYERS: List[str] = field(
        default_factory=lambda: ["QuantConv2d", "QuantLinear"],
        repr=False,
    )
    _VALID_WARMUP_SCHEDULES: List[str] = field(
        default_factory=lambda: ["linear", "cosine"],
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate configuration and set defaults."""
        if not self.target_layers:
            if self.target_type == "activation":
                self.target_layers = [
                    "QuantIdentity",
                    "QuantReLU",
                    "QuantHardTanh",
                    "QuantConv2d",
                ]
            elif self.target_type == "weight":
                self.target_layers = ["QuantConv2d", "QuantLinear"]

        self.validate()

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FaultInjectionConfig":
        """Create configuration from dictionary.

        Args:
            config: Dictionary containing fault injection parameters.
                Typically from YAML configuration file.

        Returns:
            FaultInjectionConfig instance.

        Example:
            ```python
            yaml_config = {
                "enabled": True,
                "target_type": "activation",
                "probability": 5.0,
                "injection_type": "random",
            }
            config = FaultInjectionConfig.from_dict(yaml_config)
            ```
        """
        target_type = config.get("target_type", "activation")

        if "target_layers" not in config:
            if target_type == "activation":
                default_layers = [
                    "QuantIdentity",
                    "QuantReLU",
                    "QuantHardTanh",
                    "QuantConv2d",
                ]
            else:
                default_layers = ["QuantConv2d", "QuantLinear"]
        else:
            default_layers = None

        return cls(
            enabled=config.get("enabled", False),
            target_type=target_type,
            probability=config.get("probability", 0.0),
            injection_type=config.get("injection_type", "random"),
            apply_during=config.get("apply_during", "eval"),
            target_layers=config.get("target_layers", default_layers),
            track_statistics=config.get("track_statistics", False),
            verbose=config.get("verbose", False),
            epoch_interval=config.get("epoch_interval", 1),
            step_interval=config.get("step_interval", 1),
            warmup_epochs=config.get("warmup_epochs", 0),
            warmup_schedule=config.get("warmup_schedule", "linear"),
        )

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.probability < 0.0 or self.probability > 100.0:
            raise ValueError(
                f"probability must be between 0 and 100, got {self.probability}"
            )

        if self.target_type not in self._VALID_TARGET_TYPES:
            raise ValueError(
                f"target_type must be one of {self._VALID_TARGET_TYPES}, "
                f"got '{self.target_type}'"
            )

        if self.injection_type not in self._VALID_INJECTION_TYPES:
            raise ValueError(
                f"injection_type must be one of {self._VALID_INJECTION_TYPES}, "
                f"got '{self.injection_type}'"
            )

        if self.apply_during not in self._VALID_APPLY_DURING:
            raise ValueError(
                f"apply_during must be one of {self._VALID_APPLY_DURING}, "
                f"got '{self.apply_during}'"
            )

        valid_layers = (
            self._VALID_ACTIVATION_LAYERS
            if self.target_type == "activation"
            else self._VALID_WEIGHT_LAYERS
        )
        for layer in self.target_layers:
            if layer not in valid_layers:
                raise ValueError(
                    f"For target_type='{self.target_type}', target_layers must be one of {valid_layers}, "
                    f"got '{layer}'"
                )

        if not isinstance(self.epoch_interval, int) or self.epoch_interval < 1:
            raise ValueError(
                f"epoch_interval must be a positive integer >= 1, got {self.epoch_interval}"
            )

        if not isinstance(self.step_interval, int) or self.step_interval < 1:
            raise ValueError(
                f"step_interval must be a positive integer >= 1, got {self.step_interval}"
            )

        if not isinstance(self.warmup_epochs, int) or self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be a non-negative integer, got {self.warmup_epochs}"
            )

        if self.warmup_schedule not in self._VALID_WARMUP_SCHEDULES:
            raise ValueError(
                f"warmup_schedule must be one of {self._VALID_WARMUP_SCHEDULES}, "
                f"got '{self.warmup_schedule}'"
            )

    def should_inject_during_training(self) -> bool:
        """Check if injection should occur during training.

        Returns:
            True if injection is enabled for training phase.
        """
        return self.enabled and self.apply_during in ("train", "both")

    def should_inject_during_eval(self) -> bool:
        """Check if injection should occur during evaluation.

        Returns:
            True if injection is enabled for evaluation phase.
        """
        return self.enabled and self.apply_during in ("eval", "both")

    def is_faulty_epoch(self, epoch: int) -> bool:
        """Return True if fault injection should be active during this epoch.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            True when (epoch % epoch_interval == 0).
        """
        return epoch % self.epoch_interval == 0

    def is_faulty_step(self, step: int) -> bool:
        """Return True if fault injection should fire on this step.

        Args:
            step: Current step index within the epoch (0-based).

        Returns:
            True when (step % step_interval == 0).
        """
        return step % self.step_interval == 0

    def get_warmup_probability(self, epoch: int) -> float:
        """Return the effective injection probability for the given epoch.

        During the warmup window [0, warmup_epochs), the probability is
        interpolated from 0 to self.probability. At epoch >= warmup_epochs
        the full target probability is returned.

        When warmup_epochs == 0, always returns self.probability unchanged.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Effective probability value in [0, self.probability].

        Example:
            >>> cfg = FaultInjectionConfig(probability=10.0, warmup_epochs=5,
            ...                            warmup_schedule="linear")
            >>> [cfg.get_warmup_probability(e) for e in range(7)]
            [2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 10.0]
        """
        if self.warmup_epochs == 0 or epoch >= self.warmup_epochs:
            return self.probability

        progress: float = (epoch + 1) / self.warmup_epochs

        if self.warmup_schedule == "linear":
            factor = progress
        else:
            factor = (1.0 - math.cos(math.pi * progress)) / 2.0

        return self.probability * factor

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "enabled": self.enabled,
            "target_type": self.target_type,
            "probability": self.probability,
            "injection_type": self.injection_type,
            "apply_during": self.apply_during,
            "target_layers": self.target_layers,
            "track_statistics": self.track_statistics,
            "verbose": self.verbose,
            "epoch_interval": self.epoch_interval,
            "step_interval": self.step_interval,
            "warmup_epochs": self.warmup_epochs,
            "warmup_schedule": self.warmup_schedule,
        }
