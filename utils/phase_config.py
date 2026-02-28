"""Phase configuration dataclass for multi-phase training.

Provides PhaseConfig for representing a single training phase, and parse_phases
for converting a config dict into a list of PhaseConfig objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PhaseConfig:
    """Configuration for a single training phase.

    Every field that controls training behavior is required. Phases do not
    inherit from each other or from top-level config keys.
    """

    name: str
    phase_index: int
    epochs: int
    batch_size: int
    loss: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    activation_fault_injection: Dict[str, Any] = field(default_factory=dict)
    weight_fault_injection: Dict[str, Any] = field(default_factory=dict)
    global_epoch_offset: int = 0

    @classmethod
    def from_dict(
        cls,
        phase_dict: Dict[str, Any],
        phase_index: int,
        global_epoch_offset: int = 0,
    ) -> "PhaseConfig":
        """Create a PhaseConfig from a raw YAML dict entry.

        Args:
            phase_dict: Raw phase configuration from YAML.
            phase_index: 0-based position in the phases list.
            global_epoch_offset: Cumulative epoch count from prior phases.

        Returns:
            PhaseConfig instance.

        Raises:
            KeyError: If required fields are missing.
        """
        training = phase_dict.get("training", {})
        return cls(
            name=phase_dict["name"],
            phase_index=phase_index,
            epochs=training["epochs"],
            batch_size=training["batch_size"],
            loss=phase_dict["loss"],
            optimizer=phase_dict["optimizer"],
            scheduler=phase_dict["scheduler"],
            activation_fault_injection=phase_dict.get("activation_fault_injection", {}),
            weight_fault_injection=phase_dict.get("weight_fault_injection", {}),
            global_epoch_offset=global_epoch_offset,
        )

    def validate(self) -> None:
        """Validate that required fields exist and values are in valid ranges.

        Raises:
            ValueError: If any required field is missing or invalid.
        """
        if not self.name or not self.name.strip():
            raise ValueError(f"Phase {self.phase_index}: 'name' is required")
        if self.epochs <= 0:
            raise ValueError(
                f"Phase '{self.name}': epochs must be positive, got {self.epochs}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"Phase '{self.name}': batch_size must be positive, got {self.batch_size}"
            )
        if not self.loss:
            raise ValueError(f"Phase '{self.name}': 'loss' section is required")
        if "name" not in self.loss:
            raise ValueError(f"Phase '{self.name}': 'loss.name' is required")
        if not self.optimizer:
            raise ValueError(f"Phase '{self.name}': 'optimizer' section is required")
        if "name" not in self.optimizer:
            raise ValueError(f"Phase '{self.name}': 'optimizer.name' is required")
        if "learning_rate" not in self.optimizer:
            raise ValueError(
                f"Phase '{self.name}': 'optimizer.learning_rate' is required"
            )
        if not self.scheduler:
            raise ValueError(f"Phase '{self.name}': 'scheduler' section is required")
        if "name" not in self.scheduler:
            raise ValueError(f"Phase '{self.name}': 'scheduler.name' is required")

    def to_flat_config(self, global_config: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a flat config dict compatible with existing factories.

        Args:
            global_config: Global config dict with shared settings (seed, model, etc.)

        Returns:
            Flat config dict merging global settings with phase-specific values.
        """
        flat = dict(global_config)
        flat["training"] = {"epochs": self.epochs, "batch_size": self.batch_size}
        flat["optimizer"] = self.optimizer
        flat["scheduler"] = self.scheduler
        flat["loss"] = self.loss
        flat["activation_fault_injection"] = self.activation_fault_injection
        flat["weight_fault_injection"] = self.weight_fault_injection
        return flat

    @property
    def has_fault_injection(self) -> bool:
        """Check if this phase has any fault injection enabled."""
        act = self.activation_fault_injection.get("enabled", False)
        wgt = self.weight_fault_injection.get("enabled", False)
        return act or wgt

    @property
    def total_epochs_before(self) -> int:
        """Return total number of epochs from all prior phases."""
        return self.global_epoch_offset


def parse_phases(config: Dict[str, Any]) -> List[PhaseConfig]:
    """Parse the 'phases' key from a config dict into PhaseConfig objects.

    Args:
        config: Full configuration dictionary.

    Returns:
        List of PhaseConfig objects in order.

    Raises:
        ValueError: If 'phases' key is missing, empty, or contains invalid phases.
    """
    if "phases" not in config:
        raise ValueError(
            "Config must contain a 'phases' key. "
            "Flat configs without 'phases' are not supported. "
            "See the migration guide in docs/plan/13-backward-compat.md."
        )

    raw_phases = config["phases"]
    if not isinstance(raw_phases, list) or len(raw_phases) == 0:
        raise ValueError("'phases' must be a non-empty list")

    phases: List[PhaseConfig] = []
    offset = 0
    for i, raw in enumerate(raw_phases):
        phase = PhaseConfig.from_dict(
            phase_dict=raw, phase_index=i, global_epoch_offset=offset
        )
        phase.validate()
        phases.append(phase)
        offset += phase.epochs

    return phases
