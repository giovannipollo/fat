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
        self._validate_seed_section()

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

        if "epochs" not in self.config.get("training", {}):
            self.errors.append("Missing 'training.epochs'")

    def _validate_seed_section(self) -> None:
        """Validate optional seed section values."""
        seed_cfg = self.config.get("seed", {})
        if not seed_cfg.get("enabled", False):
            return

        train_seed = seed_cfg.get("value")
        val_seed = seed_cfg.get("val_seed")
        test_seed = seed_cfg.get("test_seed")

        for name, seed_value in (("val_seed", val_seed), ("test_seed", test_seed)):
            if seed_value is None:
                continue

            if not isinstance(seed_value, int) or seed_value < 0:
                self.errors.append(
                    f"seed.{name} must be a non-negative integer, got {seed_value!r}"
                )

            if seed_value == train_seed:
                self.warnings.append(
                    f"seed.{name} ({seed_value}) equals seed.value ({train_seed}); "
                    "consider using distinct seeds for training and evaluation"
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
