"""Unit tests for PhaseManager."""

from __future__ import annotations

import pytest

from utils.phase_manager import PhaseManager


class TestPhaseManager:
    """Test PhaseManager functionality."""

    def test_single_phase_mode(self) -> None:
        """Test that config without phases returns inactive PhaseManager."""
        config = {
            "training": {"epochs": 100},
            "optimizer": {"name": "adam"},
        }

        phase_mgr = PhaseManager(config)

        assert not phase_mgr.is_multi_phase()
        assert phase_mgr.get_num_phases() == 0
        assert phase_mgr.get_current_phase(50) is None

    def test_multi_phase_parsing(self) -> None:
        """Test parsing multi-phase config."""
        config = {
            "phases": [
                {"name": "phase1", "epochs": 100},
                {"name": "phase2", "epochs": 50},
            ]
        }

        phase_mgr = PhaseManager(config)

        assert phase_mgr.is_multi_phase()
        assert phase_mgr.get_num_phases() == 2
        assert phase_mgr.get_total_epochs() == 150

    def test_epoch_range_computation(self) -> None:
        """Test that epoch ranges are computed correctly."""
        config = {
            "phases": [
                {"name": "phase1", "epochs": 100},
                {"name": "phase2", "epochs": 50},
            ]
        }

        phase_mgr = PhaseManager(config)

        phase1 = phase_mgr.get_phase_by_index(0)
        assert phase1.start_epoch == 0
        assert phase1.end_epoch == 100

        phase2 = phase_mgr.get_phase_by_index(1)
        assert phase2.start_epoch == 100
        assert phase2.end_epoch == 150

    def test_get_phase_at_epoch(self) -> None:
        """Test phase detection by epoch number."""
        config = {
            "phases": [
                {"name": "phase1", "epochs": 100},
                {"name": "phase2", "epochs": 50},
            ]
        }

        phase_mgr = PhaseManager(config)

        assert phase_mgr.get_phase_at_epoch(0) == 0
        assert phase_mgr.get_phase_at_epoch(50) == 0
        assert phase_mgr.get_phase_at_epoch(99) == 0
        assert phase_mgr.get_phase_at_epoch(100) == 1
        assert phase_mgr.get_phase_at_epoch(149) == 1
        assert phase_mgr.get_phase_at_epoch(150) == -1  # Out of range

    def test_phase_transition_detection(self) -> None:
        """Test is_phase_transition()."""
        config = {
            "phases": [
                {"name": "phase1", "epochs": 100},
                {"name": "phase2", "epochs": 50},
            ]
        }

        phase_mgr = PhaseManager(config)

        assert not phase_mgr.is_phase_transition(0)  # Start of phase1
        assert not phase_mgr.is_phase_transition(50)
        assert phase_mgr.is_phase_transition(100)  # Start of phase2
        assert not phase_mgr.is_phase_transition(101)

    def test_config_merging(self) -> None:
        """Test config merging with phase overrides."""
        config = {
            "optimizer": {
                "name": "adam",
                "learning_rate": 0.1,
                "weight_decay": 0.0001,
            },
            "phases": [
                {
                    "name": "phase1",
                    "epochs": 100,
                },
                {
                    "name": "phase2",
                    "epochs": 50,
                    "optimizer": {
                        "learning_rate": 0.01,  # Override only LR
                    },
                },
            ],
        }

        phase_mgr = PhaseManager(config)

        # Phase 1 uses base config
        merged1 = phase_mgr.get_merged_config(0)
        assert merged1["optimizer"]["learning_rate"] == 0.1
        assert merged1["optimizer"]["weight_decay"] == 0.0001

        # Phase 2 overrides LR, keeps weight_decay
        merged2 = phase_mgr.get_merged_config(100)
        assert merged2["optimizer"]["learning_rate"] == 0.01
        assert merged2["optimizer"]["weight_decay"] == 0.0001

    def test_empty_phases_list(self) -> None:
        """Test that empty phases list raises error."""
        config = {"phases": []}

        with pytest.raises(ValueError, match="phases.*empty"):
            PhaseManager(config)

    def test_invalid_phases_type(self) -> None:
        """Test that non-list phases raises error."""
        config = {"phases": "not_a_list"}

        with pytest.raises(ValueError, match="phases.*must be a list"):
            PhaseManager(config)

    def test_get_phase_info(self) -> None:
        """Test get_phase_info() method."""
        config = {
            "phases": [
                {"name": "phase1", "epochs": 50},
                {"name": "phase2", "epochs": 50},
            ]
        }

        phase_mgr = PhaseManager(config)

        # Single phase config returns different info
        info_single = PhaseManager({"training": {"epochs": 100}}).get_phase_info(50)
        assert info_single["mode"] == "single_phase"

        # Multi phase config returns detailed info
        info_multi = phase_mgr.get_phase_info(25)
        assert info_multi["mode"] == "multi_phase"
        assert info_multi["phase_idx"] == 0
        assert info_multi["phase_name"] == "phase1"
        assert info_multi["phase_progress"] == "25/50"
        assert info_multi["total_phases"] == 2
        assert info_multi["total_epochs"] == 100
