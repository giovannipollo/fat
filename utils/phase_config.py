"""Phase configuration dataclass for multi-phase training.

Provides a dataclass for managing multi-phase training configurations with
validation and YAML configuration support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PhaseConfig:
    """Configuration for a single training phase.

    Represents one phase of multi-phase training with optional parameter
    overrides. Supports overriding optimizer, scheduler, fault injection,
    and other training parameters on a per-phase basis.

    Attributes:
        name: Human-readable phase name (e.g., 'standard_training',
            'fault_aware_finetuning').
        epochs: Number of epochs to run this phase.
        start_epoch: Absolute starting epoch number (computed by PhaseManager).
        end_epoch: Absolute ending epoch number (computed by PhaseManager).
        phase_idx: Zero-based phase index (computed by PhaseManager).
        training: Override training parameters (batch_size, test_frequency, etc.).
        optimizer: Override optimizer configuration.
        scheduler: Override scheduler configuration.
        activation_fault_injection: Override activation fault injection configuration.
        weight_fault_injection: Override weight fault injection configuration.
        load_checkpoint: Checkpoint to load before this phase starts. Can be
            absolute path, relative to experiment dir, or special values like
            "best" or "latest". None means continue from previous phase.
        freeze_layers: List of layer name patterns to freeze during this phase.
            Supports wildcards.
        metadata: Custom metadata for this phase.

    Example:
        ```python
        # Create phase config from YAML dict
        config = PhaseConfig.from_dict({
            "name": "fault_aware_finetuning",
            "epochs": 50,
            "optimizer": {"learning_rate": 0.01},
            "activation_fault_injection": {"enabled": True, "probability": 5.0}
        })
        ```
    """

    # Required fields
    name: str
    epochs: int

    # Computed fields (set by PhaseManager)
    start_epoch: int = 0
    end_epoch: int = 0
    phase_idx: int = 0

    # Optional override fields
    training: Optional[Dict[str, Any]] = None
    optimizer: Optional[Dict[str, Any]] = None
    scheduler: Optional[Dict[str, Any]] = None
    activation_fault_injection: Optional[Dict[str, Any]] = None
    weight_fault_injection: Optional[Dict[str, Any]] = None
    load_checkpoint: Optional[str] = None
    freeze_layers: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate phase configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate phase configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        # Validate name
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"Phase name must be a non-empty string, got: {self.name}")

        # Validate epochs
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError(
                f"Phase '{self.name}': epochs must be a positive integer, got: {self.epochs}"
            )

        # Validate checkpoint path if specified
        if self.load_checkpoint is not None:
            if not isinstance(self.load_checkpoint, str):
                raise ValueError(
                    f"Phase '{self.name}': load_checkpoint must be a string path"
                )

        # Validate optimizer override if specified
        if self.optimizer is not None:
            if not isinstance(self.optimizer, dict):
                raise ValueError(
                    f"Phase '{self.name}': optimizer must be a dict, got: {type(self.optimizer)}"
                )

        # Validate scheduler override if specified
        if self.scheduler is not None:
            if not isinstance(self.scheduler, dict):
                raise ValueError(
                    f"Phase '{self.name}': scheduler must be a dict, got: {type(self.scheduler)}"
                )

        # Validate fault injection overrides
        for fi_key in ["activation_fault_injection", "weight_fault_injection"]:
            fi_config = getattr(self, fi_key)
            if fi_config is not None:
                if not isinstance(fi_config, dict):
                    raise ValueError(f"Phase '{self.name}': {fi_key} must be a dict")

        # Validate freeze_layers if specified
        if self.freeze_layers is not None:
            if not isinstance(self.freeze_layers, list):
                raise ValueError(
                    f"Phase '{self.name}': freeze_layers must be a list of strings"
                )
            for layer_pattern in self.freeze_layers:
                if not isinstance(layer_pattern, str):
                    raise ValueError(
                        f"Phase '{self.name}': freeze_layers must contain strings, got: {layer_pattern}"
                    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert phase config to dictionary.

        Returns:
            Dictionary representation of phase config.
        """
        result: Dict[str, Any] = {
            "name": self.name,
            "epochs": self.epochs,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "phase_idx": self.phase_idx,
        }

        # Add optional fields if present
        if self.training is not None:
            result["training"] = self.training
        if self.optimizer is not None:
            result["optimizer"] = self.optimizer
        if self.scheduler is not None:
            result["scheduler"] = self.scheduler
        if self.activation_fault_injection is not None:
            result["activation_fault_injection"] = self.activation_fault_injection
        if self.weight_fault_injection is not None:
            result["weight_fault_injection"] = self.weight_fault_injection
        if self.load_checkpoint is not None:
            result["load_checkpoint"] = self.load_checkpoint
        if self.freeze_layers is not None:
            result["freeze_layers"] = self.freeze_layers
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PhaseConfig:
        """Create PhaseConfig from dictionary.

        Args:
            data: Dictionary with phase configuration.

        Returns:
            PhaseConfig instance.

        Example:
            ```python
            config = PhaseConfig.from_dict({
                "name": "fault_aware_finetuning",
                "epochs": 50,
                "optimizer": {"learning_rate": 0.01}
            })
            ```
        """
        return cls(
            name=data["name"],
            epochs=data["epochs"],
            start_epoch=data.get("start_epoch", 0),
            end_epoch=data.get("end_epoch", 0),
            phase_idx=data.get("phase_idx", 0),
            training=data.get("training"),
            optimizer=data.get("optimizer"),
            scheduler=data.get("scheduler"),
            activation_fault_injection=data.get("activation_fault_injection"),
            weight_fault_injection=data.get("weight_fault_injection"),
            load_checkpoint=data.get("load_checkpoint"),
            freeze_layers=data.get("freeze_layers"),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PhaseConfig(name='{self.name}', "
            f"epochs={self.epochs}, "
            f"range=[{self.start_epoch}, {self.end_epoch}), "
            f"idx={self.phase_idx})"
        )
