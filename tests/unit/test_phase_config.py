"""Unit tests for PhaseConfig dataclass."""

from __future__ import annotations

import pytest

from utils.phase_config import PhaseConfig


class TestPhaseConfig:
    """Test PhaseConfig dataclass."""

    def test_valid_phase_config(self) -> None:
        """Test creating valid PhaseConfig."""
        phase = PhaseConfig(
            name="test_phase",
            epochs=100,
        )

        assert phase.name == "test_phase"
        assert phase.epochs == 100
        assert phase.start_epoch == 0
        assert phase.end_epoch == 0  # Not set until PhaseManager computes it

    def test_phase_config_with_overrides(self) -> None:
        """Test PhaseConfig with optional overrides."""
        phase = PhaseConfig(
            name="test",
            epochs=50,
            optimizer={"name": "adam", "learning_rate": 0.001},
            activation_fault_injection={"enabled": True, "probability": 5.0},
        )

        assert phase.optimizer is not None
        assert phase.optimizer["learning_rate"] == 0.001
        assert phase.activation_fault_injection["enabled"] is True

    def test_invalid_name(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            PhaseConfig(name="", epochs=100)

    def test_invalid_epochs_negative(self) -> None:
        """Test that negative epochs raises ValueError."""
        with pytest.raises(ValueError, match="epochs must be a positive integer"):
            PhaseConfig(name="test", epochs=-10)

    def test_invalid_epochs_zero(self) -> None:
        """Test that zero epochs raises ValueError."""
        with pytest.raises(ValueError, match="epochs must be a positive integer"):
            PhaseConfig(name="test", epochs=0)

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        phase = PhaseConfig(
            name="test",
            epochs=100,
            optimizer={"learning_rate": 0.01},
        )

        phase_dict = phase.to_dict()

        assert phase_dict["name"] == "test"
        assert phase_dict["epochs"] == 100
        assert "optimizer" in phase_dict

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "name": "test",
            "epochs": 100,
            "optimizer": {"learning_rate": 0.01},
        }

        phase = PhaseConfig.from_dict(data)

        assert phase.name == "test"
        assert phase.epochs == 100
        assert phase.optimizer is not None
        assert phase.optimizer["learning_rate"] == 0.01
