"""Configuration validation for training.

Provides comprehensive validation of YAML configs including schema validation,
semantic validation, and warning generation for potential issues.
"""

from __future__ import annotations

from typing import Any, Dict, List


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""

    pass


class ConfigValidator:
    """Validates training configurations.

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
        required_global = ["dataset", "model"]

        for section in required_global:
            if section not in self.config:
                self.errors.append(f"Missing required section: '{section}'")

        if "phases" not in self.config:
            self.errors.append(
                "Missing required section: 'phases'. "
                "All configs must use the phases format."
            )
        else:
            self._validate_phases()
            self._warn_ignored_top_level_keys()

    def _validate_phases(self) -> None:
        """Validate the phases list and each phase's contents.

        All training fields are required in every phase. Phases do not
        inherit from each other or from top-level keys.
        """
        phases = self.config.get("phases", [])

        if not isinstance(phases, list):
            self.errors.append("'phases' must be a list")
            return

        if len(phases) == 0:
            self.errors.append("'phases' must contain at least one phase")
            return

        seen_names: set = set()
        for i, phase in enumerate(phases):
            prefix = f"phases[{i}]"

            name = phase.get("name")
            if not name:
                self.errors.append(f"{prefix}: 'name' is required")
            elif name in seen_names:
                self.errors.append(f"{prefix}: duplicate phase name '{name}'")
            else:
                seen_names.add(name)
                prefix = f"phase '{name}'"

            training = phase.get("training")
            if not training:
                self.errors.append(f"{prefix}: 'training' section is required")
            else:
                if "epochs" not in training:
                    self.errors.append(f"{prefix}: 'training.epochs' is required")
                elif not isinstance(training["epochs"], int) or training["epochs"] <= 0:
                    self.errors.append(
                        f"{prefix}: 'training.epochs' must be a positive integer"
                    )

                if "batch_size" not in training:
                    self.errors.append(f"{prefix}: 'training.batch_size' is required")
                elif not isinstance(training["batch_size"], int) or training["batch_size"] <= 0:
                    self.errors.append(
                        f"{prefix}: 'training.batch_size' must be a positive integer"
                    )

            loss = phase.get("loss")
            if not loss:
                self.errors.append(f"{prefix}: 'loss' section is required")
            elif "name" not in loss:
                self.errors.append(f"{prefix}: 'loss.name' is required")

            opt = phase.get("optimizer")
            if not opt:
                self.errors.append(f"{prefix}: 'optimizer' section is required")
            elif "name" not in opt:
                self.errors.append(f"{prefix}: 'optimizer.name' is required")
            elif "learning_rate" not in opt:
                self.errors.append(
                    f"{prefix}: 'optimizer.learning_rate' is required"
                )

            sched = phase.get("scheduler")
            if not sched:
                self.errors.append(f"{prefix}: 'scheduler' section is required")
            elif "name" not in sched:
                self.errors.append(f"{prefix}: 'scheduler.name' is required")

    def _warn_ignored_top_level_keys(self) -> None:
        """Warn if top-level training keys exist alongside phases."""
        ignored_keys = [
            "optimizer",
            "scheduler",
            "training",
            "loss",
            "activation_fault_injection",
            "weight_fault_injection",
        ]
        found = [k for k in ignored_keys if k in self.config]
        if found:
            for key in found:
                self.warnings.append(
                    f"Top-level '{key}' is ignored when using 'phases'. "
                    f"All training parameters must be defined inside each phase."
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
