"""Unit tests for ConfigValidator."""

from __future__ import annotations

import pytest

from utils.config_validator import ConfigValidator, ConfigValidationError


class TestConfigValidator:
    """Test configuration validation."""

    def test_valid_single_phase_config(self) -> None:
        """Test that valid single-phase config passes."""
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

    def test_valid_multi_phase_config(self) -> None:
        """Test that valid multi-phase config passes."""
        config = {
            "dataset": {"name": "cifar10"},
            "model": {"name": "quant_cnv"},
            "training": {"batch_size": 256},
            "optimizer": {"name": "adam"},
            "phases": [
                {"name": "phase1", "epochs": 100},
                {"name": "phase2", "epochs": 50},
            ],
        }

        validator = ConfigValidator(config)
        validator.validate()  # Should not raise

    def test_phase_missing_name(self) -> None:
        """Test that phase without name raises error."""
        config = {
            "dataset": {"name": "cifar10"},
            "model": {"name": "quant_cnv"},
            "training": {"batch_size": 256},
            "optimizer": {"name": "adam"},
            "phases": [
                {"epochs": 100},  # Missing name
            ],
        }

        validator = ConfigValidator(config)

        with pytest.raises(ConfigValidationError, match="missing required field 'name'"):
            validator.validate()

    def test_invalid_fault_injection_probability(self) -> None:
        """Test that out-of-range probability raises error."""
        config = {
            "dataset": {"name": "cifar10"},
            "model": {"name": "quant_cnv"},
            "training": {"batch_size": 256},
            "optimizer": {"name": "adam"},
            "phases": [
                {
                    "name": "test",
                    "epochs": 100,
                    "activation_fault_injection": {
                        "probability": 150,  # Invalid: > 100
                    },
                },
            ],
        }

        validator = ConfigValidator(config)

        with pytest.raises(ConfigValidationError, match="probability must be in range"):
            validator.validate()

    def test_duplicate_phase_names_warning(self) -> None:
        """Test that duplicate phase names generate warning."""
        config = {
            "dataset": {"name": "cifar10"},
            "model": {"name": "quant_cnv"},
            "training": {"batch_size": 256},
            "optimizer": {"name": "adam"},
            "phases": [
                {"name": "duplicate", "epochs": 100},
                {"name": "duplicate", "epochs": 50},
            ],
        }

        validator = ConfigValidator(config)
        # Should pass but with warnings
        validator.validate()
        assert len(validator.warnings) > 0
        assert any("duplicate" in w.lower() for w in validator.warnings)
