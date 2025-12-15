"""Configuration for fault injection.

Provides a dataclass for managing fault injection parameters with
validation and YAML configuration support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FaultInjectionConfig:
    """Configuration for fault injection.

    Attributes:
        enabled: Master switch for fault injection.
        probability: Injection probability as percentage (0-100).
        mode: Injection mode - "full_model" or "layer".
        injection_layer: Specific layer index for "layer" mode (-1 for random).
        injection_type: Type of fault - "random", "lsb_flip", "msb_flip", "full_flip".
        apply_during: When to inject - "train", "eval", or "both".
        epoch_interval: Re-generate fault mask every N epochs (training only).
        step_interval: Probability of injection per batch step (0-1).
        seed: Random seed for reproducible fault patterns.
        track_statistics: Enable statistics tracking (RMSE, cosine similarity).
        verbose: Print injection details.
        hw_mask: Use hardware-aware periodic fault pattern instead of random.
        frequency_value: Hardware parallelism factor for hw_mask (e.g., 1024 for
            1024 parallel MAC units). Determines the period of the fault pattern.
        gradient_mode: How gradients flow through faulty positions during backprop.
            - "ste": Straight-Through Estimator - gradients flow through all positions
            - "zero_faulty": Zero gradients at faulty positions, flow through clean

    Example:
        ```python
        config = FaultInjectionConfig(
            enabled=True,
            probability=5.0,
            mode="full_model",
            injection_type="random",
        )

        # Hardware-aware mask for realistic FPGA/ASIC simulation
        config = FaultInjectionConfig(
            enabled=True,
            probability=5.0,
            hw_mask=True,
            frequency_value=1024,
        )

        # Zero gradients at faulty positions (original fat/ behavior)
        config = FaultInjectionConfig(
            enabled=True,
            probability=5.0,
            gradient_mode="zero_faulty",
        )

        # Or from YAML dict
        config = FaultInjectionConfig.from_dict(yaml_config["fault_injection"])
        ```
    """

    enabled: bool = False
    probability: float = 0.0
    mode: str = "full_model"
    injection_layer: int = -1
    injection_type: str = "random"
    apply_during: str = "eval"
    epoch_interval: int = 1
    step_interval: float = 0.5
    seed: Optional[int] = None
    track_statistics: bool = False
    verbose: bool = False
    hw_mask: bool = False
    frequency_value: int = 1024
    gradient_mode: str = "ste"

    # Valid values for validation
    _VALID_MODES: List[str] = field(
        default_factory=lambda: ["full_model", "layer"],
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
    _VALID_GRADIENT_MODES: List[str] = field(
        default_factory=lambda: ["ste", "zero_faulty"],
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
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
                "probability": 5.0,
                "mode": "full_model",
            }
            config = FaultInjectionConfig.from_dict(yaml_config)
            ```
        """
        return cls(
            enabled=config.get("enabled", False),
            probability=config.get("probability", 0.0),
            mode=config.get("mode", "full_model"),
            injection_layer=config.get("injection_layer", -1),
            injection_type=config.get("injection_type", "random"),
            apply_during=config.get("apply_during", "eval"),
            epoch_interval=config.get("epoch_interval", 1),
            step_interval=config.get("step_interval", 0.5),
            seed=config.get("seed", None),
            track_statistics=config.get("track_statistics", False),
            verbose=config.get("verbose", False),
            hw_mask=config.get("hw_mask", False),
            frequency_value=config.get("frequency_value", 1024),
            gradient_mode=config.get("gradient_mode", "ste"),
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

        if self.mode not in self._VALID_MODES:
            raise ValueError(
                f"mode must be one of {self._VALID_MODES}, got '{self.mode}'"
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

        if self.epoch_interval < 1:
            raise ValueError(
                f"epoch_interval must be >= 1, got {self.epoch_interval}"
            )

        if self.step_interval < 0.0 or self.step_interval > 1.0:
            raise ValueError(
                f"step_interval must be between 0 and 1, got {self.step_interval}"
            )

        if self.frequency_value < 1:
            raise ValueError(
                f"frequency_value must be >= 1, got {self.frequency_value}"
            )

        if self.gradient_mode not in self._VALID_GRADIENT_MODES:
            raise ValueError(
                f"gradient_mode must be one of {self._VALID_GRADIENT_MODES}, "
                f"got '{self.gradient_mode}'"
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

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "enabled": self.enabled,
            "probability": self.probability,
            "mode": self.mode,
            "injection_layer": self.injection_layer,
            "injection_type": self.injection_type,
            "apply_during": self.apply_during,
            "epoch_interval": self.epoch_interval,
            "step_interval": self.step_interval,
            "seed": self.seed,
            "track_statistics": self.track_statistics,
            "verbose": self.verbose,
            "hw_mask": self.hw_mask,
            "frequency_value": self.frequency_value,
            "gradient_mode": self.gradient_mode,
        }
