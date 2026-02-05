"""Manager for multi-phase training configuration and execution.

Provides a PhaseManager class for parsing phase configurations from YAML,
computing epoch ranges, determining which phase is active at any given epoch,
and merging phase-specific configs with base config.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from .phase_config import PhaseConfig


class PhaseManager:
    """Manages multi-phase training configuration and phase transitions.

    Parses the 'phases' section from YAML config, validates phase parameters,
    computes epoch ranges, and provides merged configurations for each phase.

    Attributes:
        base_config: Full configuration dictionary (base + phases).
        phases: List of PhaseConfig instances parsed from config.
        total_epochs: Total number of epochs across all phases.

    Example:
        ```python
        config = load_config("config.yaml")
        phase_mgr = PhaseManager(config)

        for epoch in range(total_epochs):
            current_phase = phase_mgr.get_current_phase(epoch)
            merged_config = phase_mgr.get_merged_config(epoch)
            # ... train with merged_config
        ```
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize PhaseManager from config dictionary.

        Args:
            config: Full configuration dictionary (base + phases).
        """
        self.base_config: Dict[str, Any] = config
        self.phases: List[PhaseConfig] = []
        self.total_epochs: int = 0

        # Check if multi-phase training is enabled
        if "phases" in config and config["phases"] is not None:
            self._parse_phases(config["phases"])
            self._compute_epoch_ranges()
            self._validate_phases()
        else:
            # Single-phase training (legacy mode)
            # PhaseManager is inactive, Trainer uses base config directly
            pass

    def is_multi_phase(self) -> bool:
        """Check if multi-phase training is enabled.

        Returns:
            True if phases are defined, False otherwise.
        """
        return len(self.phases) > 0

    def get_total_epochs(self) -> int:
        """Get total number of epochs across all phases.

        Returns:
            Total epochs (sum of all phase epochs).
        """
        return self.total_epochs

    def get_num_phases(self) -> int:
        """Get number of phases.

        Returns:
            Number of phases.
        """
        return len(self.phases)

    def get_phase_at_epoch(self, epoch: int) -> int:
        """Determine which phase a given epoch belongs to.

        Args:
            epoch: Current epoch number (0-based).

        Returns:
            Phase index (0-based), or -1 if epoch out of range.
        """
        if not self.is_multi_phase():
            return 0  # Single phase

        for phase in self.phases:
            if phase.start_epoch <= epoch < phase.end_epoch:
                return phase.phase_idx

        return -1  # Epoch out of range

    def get_current_phase(self, epoch: int) -> Optional[PhaseConfig]:
        """Get the PhaseConfig for a given epoch.

        Args:
            epoch: Current epoch number (0-based).

        Returns:
            PhaseConfig instance, or None if not multi-phase or out of range.
        """
        if not self.is_multi_phase():
            return None

        phase_idx = self.get_phase_at_epoch(epoch)
        if phase_idx == -1:
            return None

        return self.phases[phase_idx]

    def get_phase_by_index(self, idx: int) -> Optional[PhaseConfig]:
        """Get phase by index.

        Args:
            idx: Phase index (0-based).

        Returns:
            PhaseConfig instance or None if index invalid.
        """
        if 0 <= idx < len(self.phases):
            return self.phases[idx]
        return None

    def get_phase_by_name(self, name: str) -> Optional[PhaseConfig]:
        """Get phase by name.

        Args:
            name: Phase name.

        Returns:
            PhaseConfig instance or None if not found.
        """
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None

    def is_phase_transition(self, epoch: int) -> bool:
        """Check if given epoch is the start of a new phase.

        Args:
            epoch: Current epoch number (0-based).

        Returns:
            True if this epoch starts a new phase.
        """
        if not self.is_multi_phase():
            return False

        for phase in self.phases:
            if epoch == phase.start_epoch and epoch > 0:
                return True

        return False

    def get_merged_config(self, epoch: int) -> Dict[str, Any]:
        """Get merged configuration for a given epoch.

        Merges base config with phase-specific overrides using deep merge.
        Phase config takes precedence for conflicting keys.

        Args:
            epoch: Current epoch number (0-based).

        Returns:
            Merged configuration dictionary.
        """
        current_phase = self.get_current_phase(epoch)

        if current_phase is None:
            # Single-phase training, return base config
            return copy.deepcopy(self.base_config)

        return self._merge_configs(self.base_config, current_phase)

    def get_phase_info(self, epoch: int) -> Dict[str, Any]:
        """Get human-readable phase information for logging.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary with phase info (name, idx, progress, etc.).
        """
        if not self.is_multi_phase():
            return {
                "mode": "single_phase",
                "total_epochs": self.base_config.get("training", {}).get("epochs", 0),
            }

        current_phase = self.get_current_phase(epoch)
        if current_phase is None:
            return {"mode": "unknown", "epoch": epoch}

        phase_progress = epoch - current_phase.start_epoch
        phase_total = current_phase.epochs

        return {
            "mode": "multi_phase",
            "phase_idx": current_phase.phase_idx,
            "phase_name": current_phase.name,
            "phase_progress": f"{phase_progress}/{phase_total}",
            "phase_start": current_phase.start_epoch,
            "phase_end": current_phase.end_epoch,
            "total_phases": len(self.phases),
            "total_epochs": self.total_epochs,
        }

    # Private methods

    def _parse_phases(self, phases_config: List[Dict[str, Any]]) -> None:
        """Parse phases from YAML config list.

        Args:
            phases_config: List of phase configuration dicts.

        Raises:
            ValueError: If phases_config is invalid.
        """
        if not isinstance(phases_config, list):
            raise ValueError(f"'phases' must be a list, got: {type(phases_config)}")

        if len(phases_config) == 0:
            raise ValueError(
                "'phases' list is empty - either remove 'phases' key or add at least one phase"
            )

        for idx, phase_dict in enumerate(phases_config):
            if not isinstance(phase_dict, dict):
                raise ValueError(f"Phase {idx} must be a dict, got: {type(phase_dict)}")

            # Create PhaseConfig (validation happens in __post_init__)
            phase = PhaseConfig.from_dict(phase_dict)
            phase.phase_idx = idx
            self.phases.append(phase)

    def _compute_epoch_ranges(self) -> None:
        """Compute start_epoch and end_epoch for each phase.

        Phases run sequentially: phase[i].end_epoch = phase[i+1].start_epoch.
        """
        current_epoch = 0

        for phase in self.phases:
            phase.start_epoch = current_epoch
            phase.end_epoch = current_epoch + phase.epochs
            current_epoch = phase.end_epoch

        self.total_epochs = current_epoch

    def _validate_phases(self) -> None:
        """Validate phase configurations for consistency.

        Raises:
            ValueError: If validation fails.
        """
        # Check for duplicate phase names (warning, not error)
        phase_names = [p.name for p in self.phases]
        if len(phase_names) != len(set(phase_names)):
            # Find duplicates
            seen = set()
            duplicates = set()
            for name in phase_names:
                if name in seen:
                    duplicates.add(name)
                seen.add(name)
            print(f"WARNING: Duplicate phase names detected: {duplicates}")

        # Validate checkpoint paths
        for phase in self.phases:
            if phase.load_checkpoint is not None:
                self._validate_checkpoint_path(phase)

        # Validate scheduler compatibility
        for phase in self.phases:
            if phase.scheduler is not None:
                self._validate_scheduler_compatibility(phase)

    def _validate_checkpoint_path(self, phase: PhaseConfig) -> None:
        """Validate checkpoint path for a phase.

        Args:
            phase: PhaseConfig to validate.

        Note:
            Special values "best" and "latest" are always valid.
            File paths are checked for existence (warning if not found).
        """
        ckpt_path = phase.load_checkpoint

        # Special values are always valid
        if ckpt_path in ("best", "latest"):
            return

        # Check if file exists (warn if not, don't error - might be created later)
        path = Path(ckpt_path)
        if not path.exists():
            print(
                f"WARNING: Phase '{phase.name}' specifies checkpoint '{ckpt_path}' "
                f"which does not exist yet. Ensure it will be created before this phase."
            )

    def _validate_scheduler_compatibility(self, phase: PhaseConfig) -> None:
        """Validate scheduler configuration for a phase.

        Args:
            phase: PhaseConfig to validate.
        """
        scheduler_config = phase.scheduler
        scheduler_name = scheduler_config.get("name", "")

        # For cosine scheduler, warn if T_max doesn't match phase duration
        if scheduler_name == "cosine":
            t_max = scheduler_config.get("T_max")
            if t_max is not None and t_max != phase.epochs:
                print(
                    f"WARNING: Phase '{phase.name}' has cosine scheduler with "
                    f"T_max={t_max} but phase runs for {phase.epochs} epochs. "
                    f"Consider setting T_max={phase.epochs}."
                )

    def _merge_configs(
        self, base: Dict[str, Any], phase: PhaseConfig
    ) -> Dict[str, Any]:
        """Deep merge base config with phase-specific overrides.

        Args:
            base: Base configuration dict.
            phase: PhaseConfig with override fields.

        Returns:
            Merged configuration dict.

        Note:
            Merge strategy:
            1. Start with deep copy of base config
            2. For each override field in phase:
               - If dict: recursively merge
               - Otherwise: replace entirely
            3. Special handling for 'training.epochs' - use phase.end_epoch
        """
        merged = copy.deepcopy(base)

        # Merge each override field
        if phase.training is not None:
            merged["training"] = self._deep_merge_dicts(
                merged.get("training", {}), phase.training
            )

        if phase.optimizer is not None:
            merged["optimizer"] = self._deep_merge_dicts(
                merged.get("optimizer", {}), phase.optimizer
            )

        if phase.scheduler is not None:
            merged["scheduler"] = self._deep_merge_dicts(
                merged.get("scheduler", {}), phase.scheduler
            )

        if phase.activation_fault_injection is not None:
            merged["activation_fault_injection"] = self._deep_merge_dicts(
                merged.get("activation_fault_injection", {}),
                phase.activation_fault_injection,
            )

        if phase.weight_fault_injection is not None:
            merged["weight_fault_injection"] = self._deep_merge_dicts(
                merged.get("weight_fault_injection", {}), phase.weight_fault_injection
            )

        # Override total epochs to reflect phase duration
        if "training" not in merged:
            merged["training"] = {}
        merged["training"]["epochs"] = phase.end_epoch

        return merged

    def _deep_merge_dicts(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge two dictionaries.

        Args:
            base: Base dictionary.
            override: Override dictionary (takes precedence).

        Returns:
            Merged dictionary.

        Note:
            - For dict values: recurse
            - For non-dict values: override replaces base
            - Keys only in base or override are kept
        """
        result = copy.deepcopy(base)

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Both are dicts - recurse
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                # Override takes precedence
                result[key] = copy.deepcopy(value)

        return result


def create_phase_manager(config: Dict[str, Any]) -> Optional[PhaseManager]:
    """Factory function to create PhaseManager from config.

    Args:
        config: Configuration dictionary.

    Returns:
        PhaseManager instance if multi-phase, None otherwise.
    """
    phase_mgr = PhaseManager(config)
    return phase_mgr if phase_mgr.is_multi_phase() else None
