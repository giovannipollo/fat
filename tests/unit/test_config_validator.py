"""Unit tests for ConfigValidator."""

from __future__ import annotations

import pytest

from utils.config_validator import ConfigValidator, ConfigValidationError


class TestConfigValidator:
    """Test configuration validation."""

    def test_valid_config(self) -> None:
        """Test that valid config passes."""
        config = {
            "dataset": {"name": "cifar10"},
            "model": {"name": "quant_cnv"},
            "training": {"epochs": 100},
            "optimizer": {"name": "adam"},
        }

        validator = ConfigValidator(config)
        validator.validate()  # Should not raise

    def test_missing_required_section(self) -> None:
        """Test that missing required section raises error."""
        config = {
            "dataset": {"name": "cifar10"},
            # Missing model, training, optimizer
        }

        validator = ConfigValidator(config)

        with pytest.raises(ConfigValidationError, match="Missing required section"):
            validator.validate()

    def test_missing_epochs(self) -> None:
        """Test that missing training.epochs raises error."""
        config = {
            "dataset": {"name": "cifar10"},
            "model": {"name": "quant_cnv"},
            "training": {"batch_size": 256},  # Missing epochs
            "optimizer": {"name": "adam"},
        }

        validator = ConfigValidator(config)

        with pytest.raises(ConfigValidationError, match="Missing 'training.epochs'"):
            validator.validate()
