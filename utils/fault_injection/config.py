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
        injection_type: Type of fault - "random", "lsb_flip", "msb_flip", "full_flip".
        apply_during: When to inject - "train", "eval", or "both".
        target_layers: List of layer types to inject after (e.g., ["QuantConv2d", "QuantReLU"]).
        track_statistics: Enable statistics tracking (RMSE, cosine similarity).
        verbose: Print injection details.

    Example:
        ```python
        config = FaultInjectionConfig(
            enabled=True,
            probability=5.0,
            injection_type="random",
        )

        # Or from YAML dict
        config = FaultInjectionConfig.from_dict(yaml_config["fault_injection"])
        ```
    """

    enabled: bool = False
    probability: float = 0.0
    injection_type: str = "random"
    apply_during: str = "eval"
    target_layers: List[str] = field(
        default_factory=lambda: ["QuantIdentity", "QuantReLU", "QuantHardTanh", "QuantConv2d"]
    )
    track_statistics: bool = False
    verbose: bool = False

    # Valid values for validation
    _VALID_INJECTION_TYPES: List[str] = field(
        default_factory=lambda: ["random", "lsb_flip", "msb_flip", "full_flip"],
        repr=False,
    )
    _VALID_APPLY_DURING: List[str] = field(
        default_factory=lambda: ["train", "eval", "both"],
        repr=False,
    )
    _VALID_TARGET_LAYERS: List[str] = field(
        default_factory=lambda: [
            "QuantIdentity",
            "QuantReLU",
            "QuantHardTanh",
            "QuantConv2d",
            "QuantLinear",
        ],
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
                "injection_type": "random",
            }
            config = FaultInjectionConfig.from_dict(yaml_config)
            ```
        """
        return cls(
            enabled=config.get("enabled", False),
            probability=config.get("probability", 0.0),
            injection_type=config.get("injection_type", "random"),
            apply_during=config.get("apply_during", "eval"),
            target_layers=config.get(
                "target_layers",
                ["QuantIdentity", "QuantReLU", "QuantHardTanh", "QuantConv2d"],
            ),
            track_statistics=config.get("track_statistics", False),
            verbose=config.get("verbose", False),
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

        for layer in self.target_layers:
            if layer not in self._VALID_TARGET_LAYERS:
                raise ValueError(
                    f"target_layers must be one of {self._VALID_TARGET_LAYERS}, "
                    f"got '{layer}'"
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
            "injection_type": self.injection_type,
            "apply_during": self.apply_during,
            "target_layers": self.target_layers,
            "track_statistics": self.track_statistics,
            "verbose": self.verbose,
        }
