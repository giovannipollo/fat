"""Configuration validation for multi-phase training.

Provides comprehensive validation of YAML configs including schema validation,
semantic validation, and warning generation for potential issues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""

    pass


class ConfigValidator:
    """Validates multi-phase training configurations.

    Performs comprehensive validation of YAML configs including:
    - Schema validation (types, required fields)
    - Semantic validation (consistency, ranges)
    - Warning generation for potential issues
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize validator with config.

        Args:
            config: Configuration dictionary from YAML.
        """
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> None:
        """Validate entire configuration.

        Raises:
            ConfigValidationError: If validation fails.
        """
        # Validate base config sections
        self._validate_required_sections()

        # Validate phases if present
        if "phases" in self.config and self.config["phases"] is not None:
            self._validate_phases()

        # Report errors
        if self.errors:
            error_msg = "Configuration validation failed:\n"
            for i, error in enumerate(self.errors, 1):
                error_msg += f"  {i}. {error}\n"
            raise ConfigValidationError(error_msg)

        # Report warnings
        if self.warnings:
            print("\nConfiguration Warnings:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
            print()

    def _validate_required_sections(self) -> None:
        """Validate that required base config sections exist."""
        required = ["dataset", "model", "training", "optimizer"]

        for section in required:
            if section not in self.config:
                self.errors.append(f"Missing required section: '{section}'")

    def _validate_phases(self) -> None:
        """Validate multi-phase configuration."""
        phases = self.config["phases"]

        # Type check
        if not isinstance(phases, list):
            self.errors.append(f"'phases' must be a list, got {type(phases).__name__}")
            return

        # Empty check
        if len(phases) == 0:
            self.errors.append(
                "'phases' list is empty - remove 'phases' key or add at least one phase"
            )
            return

        # Validate each phase
        for idx, phase_dict in enumerate(phases):
            self._validate_phase(idx, phase_dict)

        # Cross-phase validation
        self._validate_phase_consistency(phases)

    def _validate_phase(self, idx: int, phase: Dict[str, Any]) -> None:
        """Validate a single phase configuration.

        Args:
            idx: Phase index (for error messages).
            phase: Phase configuration dict.
        """
        prefix = f"Phase {idx}"

        # Type check
        if not isinstance(phase, dict):
            self.errors.append(f"{prefix}: must be a dict, got {type(phase).__name__}")
            return

        # Required fields
        if "name" not in phase:
            self.errors.append(f"{prefix}: missing required field 'name'")
        elif not isinstance(phase["name"], str) or not phase["name"]:
            self.errors.append(f"{prefix}: 'name' must be a non-empty string")

        if "epochs" not in phase:
            self.errors.append(f"{prefix}: missing required field 'epochs'")
        elif not isinstance(phase["epochs"], int) or phase["epochs"] <= 0:
            self.errors.append(
                f"{prefix}: 'epochs' must be a positive integer, got {phase.get('epochs')}"
            )

        # Optional field validation
        if "training" in phase:
            self._validate_phase_training(prefix, phase["training"])

        if "optimizer" in phase:
            self._validate_phase_optimizer(prefix, phase["optimizer"])

        if "scheduler" in phase:
            self._validate_phase_scheduler(prefix, phase["scheduler"])

        if "activation_fault_injection" in phase:
            self._validate_phase_fault_injection(
                prefix, phase["activation_fault_injection"], "activation"
            )

        if "weight_fault_injection" in phase:
            self._validate_phase_fault_injection(
                prefix, phase["weight_fault_injection"], "weight"
            )

        if "load_checkpoint" in phase:
            self._validate_phase_checkpoint(prefix, phase["load_checkpoint"])

        if "freeze_layers" in phase:
            self._validate_phase_freeze_layers(prefix, phase["freeze_layers"])

    def _validate_phase_training(self, prefix: str, training: Any) -> None:
        """Validate phase training config.

        Args:
            prefix: Error message prefix.
            training: Training configuration dict.
        """
        if not isinstance(training, dict):
            self.errors.append(f"{prefix}: 'training' must be a dict")
            return

        # Warn if 'epochs' is specified (should be at phase level)
        if "epochs" in training:
            self.warnings.append(
                f"{prefix}: 'training.epochs' is ignored - specify 'epochs' at phase level"
            )

        # Validate batch_size if present
        if "batch_size" in training:
            bs = training["batch_size"]
            if not isinstance(bs, int) or bs <= 0:
                self.errors.append(
                    f"{prefix}: training.batch_size must be positive integer"
                )

    def _validate_phase_optimizer(self, prefix: str, optimizer: Any) -> None:
        """Validate phase optimizer config.

        Args:
            prefix: Error message prefix.
            optimizer: Optimizer configuration dict.
        """
        if not isinstance(optimizer, dict):
            self.errors.append(f"{prefix}: 'optimizer' must be a dict")
            return

        # Check for valid optimizer name if specified
        if "name" in optimizer:
            valid_names = ["adam", "sgd", "adamw", "rmsprop"]
            if optimizer["name"] not in valid_names:
                self.warnings.append(
                    f"{prefix}: optimizer.name '{optimizer['name']}' may not be supported. "
                    f"Valid names: {valid_names}"
                )

        # Check learning_rate is positive
        if "learning_rate" in optimizer:
            lr = optimizer["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.errors.append(
                    f"{prefix}: optimizer.learning_rate must be positive number"
                )

    def _validate_phase_scheduler(self, prefix: str, scheduler: Any) -> None:
        """Validate phase scheduler config.

        Args:
            prefix: Error message prefix.
            scheduler: Scheduler configuration dict.
        """
        if not isinstance(scheduler, dict):
            self.errors.append(f"{prefix}: 'scheduler' must be a dict")
            return

        # Check for valid scheduler name if specified
        if "name" in scheduler:
            valid_names = ["cosine", "step", "plateau", "exponential", "constant"]
            if scheduler["name"] not in valid_names:
                self.warnings.append(
                    f"{prefix}: scheduler.name '{scheduler['name']}' may not be supported. "
                    f"Valid names: {valid_names}"
                )

    def _validate_phase_fault_injection(
        self, prefix: str, fi_config: Any, target_type: str
    ) -> None:
        """Validate phase fault injection config.

        Args:
            prefix: Error message prefix.
            fi_config: Fault injection configuration dict.
            target_type: Either "activation" or "weight".
        """
        if not isinstance(fi_config, dict):
            self.errors.append(
                f"{prefix}: '{target_type}_fault_injection' must be a dict"
            )
            return

        # Check probability range
        if "probability" in fi_config:
            prob = fi_config["probability"]
            if not isinstance(prob, (int, float)) or prob < 0 or prob > 100:
                self.errors.append(
                    f"{prefix}: {target_type}_fault_injection.probability must be in range [0, 100]"
                )

        # Check injection_type
        if "injection_type" in fi_config:
            valid_types = ["random", "lsb_flip", "msb_flip", "full_flip"]
            if fi_config["injection_type"] not in valid_types:
                self.errors.append(
                    f"{prefix}: {target_type}_fault_injection.injection_type must be one of {valid_types}"
                )

        # Check apply_during
        if "apply_during" in fi_config:
            valid_values = ["train", "eval", "both"]
            if fi_config["apply_during"] not in valid_values:
                self.errors.append(
                    f"{prefix}: {target_type}_fault_injection.apply_during must be one of {valid_values}"
                )

    def _validate_phase_checkpoint(self, prefix: str, checkpoint: Any) -> None:
        """Validate phase checkpoint path.

        Args:
            prefix: Error message prefix.
            checkpoint: Checkpoint path string.
        """
        if not isinstance(checkpoint, str):
            self.errors.append(f"{prefix}: 'load_checkpoint' must be a string")
            return

        # Special values are always valid
        if checkpoint in ("best", "latest"):
            return

        # Check if path exists (warning only, might be created later)
        path = Path(checkpoint)
        if not path.exists():
            self.warnings.append(
                f"{prefix}: checkpoint '{checkpoint}' does not exist yet. "
                f"Ensure it will be created before this phase."
            )

    def _validate_phase_freeze_layers(self, prefix: str, freeze_layers: Any) -> None:
        """Validate freeze_layers configuration.

        Args:
            prefix: Error message prefix.
            freeze_layers: List of layer name patterns.
        """
        if not isinstance(freeze_layers, list):
            self.errors.append(f"{prefix}: 'freeze_layers' must be a list")
            return

        for layer in freeze_layers:
            if not isinstance(layer, str):
                self.errors.append(
                    f"{prefix}: freeze_layers must contain strings, got {type(layer).__name__}"
                )

    def _validate_phase_consistency(self, phases: List[Dict[str, Any]]) -> None:
        """Cross-phase validation.

        Args:
            phases: List of phase dicts.
        """
        # Check for duplicate phase names
        phase_names = [p.get("name", f"phase_{i}") for i, p in enumerate(phases)]
        seen: Set[str] = set()
        duplicates: Set[str] = set()

        for name in phase_names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)

        if duplicates:
            self.warnings.append(f"Duplicate phase names detected: {duplicates}")

        # Validate scheduler T_max matches phase epochs (for cosine scheduler)
        for idx, phase in enumerate(phases):
            if "scheduler" in phase and phase["scheduler"].get("name") == "cosine":
                t_max = phase["scheduler"].get("T_max")
                phase_epochs = phase.get("epochs", 0)

                if t_max is not None and t_max != phase_epochs:
                    self.warnings.append(
                        f"Phase {idx} ('{phase.get('name')}'): cosine scheduler T_max={t_max} "
                        f"doesn't match phase epochs={phase_epochs}. Consider T_max={phase_epochs}."
                    )


def validate_config(config: Dict[str, Any]) -> None:
    """Convenience function to validate config.

    Args:
        config: Configuration dictionary.

    Raises:
        ConfigValidationError: If validation fails.

    Example:
        ```python
        from utils.config_validator import validate_config, ConfigValidationError

        try:
            validate_config(config)
        except ConfigValidationError as e:
            print(f"Configuration error: {e}")
            sys.exit(1)
        ```
    """
    validator = ConfigValidator(config)
    validator.validate()
