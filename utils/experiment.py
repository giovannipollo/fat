"""Experiment manager for checkpoints and experiment organization.

Handles experiment directory structure, checkpoint saving/loading,
and configuration persistence for reproducibility.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from .fault_injection import ActivationFaultInjector, WeightFaultInjector

import torch

try:
    import yaml

    YAML_AVAILABLE: bool = True
except ImportError:
    YAML_AVAILABLE = False


class ExperimentManager:
    """Manages experiment directories, checkpoints, and configuration saving.

    Provides organized experiment tracking with:

    - Hierarchical directory organization (dataset/model_timestamp)
    - Directory structure: experiments/<dataset>/<model_timestamp>/checkpoints/
    - Config saving for reproducibility
    - Checkpoint saving/loading with best model tracking

    Directory Structure::

        experiments/
          cifar10/
            resnet18_20240101_120000/
              config.yaml
              training_log.txt
              checkpoints/
                best.pt
                latest.pt
                epoch_0010.pt
            resnet20_20240101_130000/
              ...
          cifar100/
            resnet50_20240101_140000/
              ...
    """

    def __init__(
        self,
        config: Dict[str, Any],
        enabled: bool = True,
        base_dir: str = "./experiments",
        experiment_name: Optional[str] = None,
        save_frequency: int = 10,
        save_best: bool = True,
    ) -> None:
        """Initialize the experiment manager.

        Args:
            config: Full configuration dictionary.
            enabled: Whether checkpointing is enabled.
            base_dir: Base directory for experiments.
            experiment_name: Optional custom experiment name prefix.
            save_frequency: How often to save periodic checkpoints (epochs).
            save_best: Whether to save the best model.
        """
        self.config: Dict[str, Any] = config
        self.enabled: bool = enabled
        self.save_frequency: int = save_frequency
        self.save_best: bool = save_best

        # Directories
        self.experiment_dir: Optional[Path] = None
        self.checkpoint_dir: Optional[Path] = None

        self._resumed_phase_index: int = 0
        self._resumed_phase_name: str = ""

        if enabled:
            self._setup_experiment_dir(base_dir, experiment_name)

    def _generate_experiment_name(self, custom_name: Optional[str] = None) -> str:
        """Generate a meaningful experiment name from config.

        Args:
            custom_name: Optional custom prefix.

        Returns:
            Experiment name string (format: [prefix_]timestamp_model_precision_dataset_sat/fat).
            Phase information is NOT included - it lives in subdirectories.
        """
        model_name: str = self.config["model"]["name"].lower()
        sat_or_fat: str = self._get_sat_or_fat()
        dataset_name: str = self._get_dataset_name()
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        precision: str = self._get_precision()

        base = f"{timestamp}_{model_name}_{precision}_{dataset_name}_{sat_or_fat}"

        if custom_name:
            base = f"{custom_name}_{base}"

        return base

    def _get_dataset_name(self) -> str:
        """Get the dataset name from config.

        Returns:
            Dataset name string in lowercase.
        """
        return self.config["dataset"]["name"].lower()

    def _get_sat_or_fat(self) -> str:
        """Determine if the experiment is SAT or FAT based on config.

        Checks all phases - FAT if any phase has fault injection enabled.

        Returns:
            "sat" if no fault injection, "fat" if any fault injection enabled.
        """
        phases = self.config.get("phases", [])
        for phase in phases:
            act = phase.get("activation_fault_injection", {})
            wgt = phase.get("weight_fault_injection", {})
            if act.get("enabled", False) or wgt.get("enabled", False):
                return "fat"

        activation_fault_injection: Dict[str, Any] = self.config.get(
            "activation_fault_injection", {}
        )
        weight_fault_injection: Dict[str, Any] = self.config.get(
            "weight_fault_injection", {}
        )
        if activation_fault_injection.get(
            "enabled", False
        ) or weight_fault_injection.get("enabled", False):
            return "fat"
        else:
            return "sat"

    def _get_activation_or_weight_fault_injection(self) -> str:
        """Determine if activation or weight fault injection is used.

        Returns:
            "activation" if activation fault injection is enabled,
            "weight" if weight fault injection is enabled,
            "none" if neither is enabled.
        """
        activation_fault_injection: Dict[str, Any] = self.config.get(
            "activation_fault_injection", {}
        )
        activation_fault_injection_probability = activation_fault_injection.get(
            "probability", 0
        )
        weight_fault_injection: Dict[str, Any] = self.config.get(
            "weight_fault_injection", {}
        )
        weight_fault_injection_probability = weight_fault_injection.get(
            "probability", 0
        )
        if activation_fault_injection.get(
            "enabled", False
        ) and weight_fault_injection.get("enabled", False):
            return f"weight_{weight_fault_injection_probability}_activation_{activation_fault_injection_probability}"
        elif weight_fault_injection.get("enabled", False):
            return f"weight_{weight_fault_injection_probability}"
        elif activation_fault_injection.get("enabled", False):
            return f"activation_{activation_fault_injection_probability}"
        else:
            return ""

    def _get_precision(self) -> str:
        """Get the precision string based on quantization config.

        Returns:
            Precision string (e.g., 'inw8_w4_a4' or 'fp32').
        """
        quant_config: Optional[Dict[str, Any]] = self.config.get("quantization")
        if quant_config is not None:
            weight_bit_width: int = quant_config.get("weight_bit_width", 8)
            in_weight_bit_width: int = quant_config.get("in_weight_bit_width", 8)
            act_bit_width: int = quant_config.get("act_bit_width", 8)
            return f"inw{in_weight_bit_width}_w{weight_bit_width}_a{act_bit_width}"
        else:
            return "fp32"

    def _setup_experiment_dir(
        self,
        base_dir: str,
        custom_name: Optional[str] = None,
    ):
        """Setup experiment directory structure with per-phase subdirectories.

        Creates a hierarchical structure: base_dir/sat_or_fat/dataset/model/precision/experiment_name/
        with per-phase subdirectories inside.

        Args:
            base_dir: Base directory for experiments.
            custom_name: Optional custom experiment name prefix.
        """
        base_path: Path = Path(base_dir)
        dataset_name: str = self._get_dataset_name()
        sat_or_fat: str = self._get_sat_or_fat()
        precision_folder: str = self._get_precision()
        model_name: str = self.config["model"]["name"].lower()
        experiment_name: str = self._generate_experiment_name(custom_name)

        self.experiment_dir = (
            base_path
            / sat_or_fat
            / dataset_name
            / model_name
            / precision_folder
            / experiment_name
        )

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.phase_dirs: Dict[int, Path] = {}
        phases = self.config.get("phases", [])
        for i, phase in enumerate(phases):
            phase_name = phase.get("name", f"phase_{i}")
            phase_dir = self.experiment_dir / f"{i}_{phase_name}"
            phase_dir.mkdir(parents=True, exist_ok=True)
            (phase_dir / "checkpoints").mkdir(exist_ok=True)
            self.phase_dirs[i] = phase_dir

        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._save_config()

    def _save_config(self) -> None:
        """Save configuration to the experiment directory.

        Saves as YAML if available, otherwise as text representation.
        """
        if self.experiment_dir is None:
            return

        if YAML_AVAILABLE:
            import yaml as yaml_module

            config_path: Path = self.experiment_dir / "config.yaml"
            with open(config_path, "w") as f:
                yaml_module.dump(
                    self.config, f, default_flow_style=False, sort_keys=False
                )
        else:
            # Fallback: save as simple text representation
            config_path = self.experiment_dir / "config.txt"
            with open(config_path, "w") as f:
                f.write(str(self.config))

        print(f"Config saved to: {config_path}")

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        best_acc: float,
        current_acc: float,
        scaler: Optional[Any] = None,
        is_best: bool = False,
        is_phase_best: bool = False,
        test_acc: Optional[float] = None,
        act_fault_injector: Optional["ActivationFaultInjector"] = None,
        weight_fault_injector: Optional["WeightFaultInjector"] = None,
        act_fault_config: Optional[Any] = None,
        weight_fault_config: Optional[Any] = None,
        phase_index: int = 0,
        phase_name: str = "default",
        phase_local_epoch: int = 0,
        total_phases: int = 1,
    ):
        """Save a model checkpoint.

        Args:
            epoch: Current epoch number (0-indexed, global across all phases).
            model: The model to save.
            optimizer: The optimizer state.
            scheduler: The scheduler state (can be None).
            best_acc: Best accuracy achieved so far (global).
            current_acc: Current epoch's accuracy.
            scaler: GradScaler for AMP (optional).
            is_best: Whether this is the best model so far (global).
            is_phase_best: Whether this is the best model within the current phase.
            test_acc: Test accuracy (for best model with validation).
            act_fault_injector: Optional activation fault injector to remove before saving.
            weight_fault_injector: Optional weight fault injector to remove before saving.
            act_fault_config: Optional activation fault injection config for re-injection.
            weight_fault_config: Optional weight fault injection config for re-injection.
            phase_index: Index of the current phase (0-based).
            phase_name: Name of the current phase.
            phase_local_epoch: Epoch number within the current phase (0-based).
            total_phases: Total number of phases in the training run.
        """
        if not self.enabled or self.checkpoint_dir is None:
            return

        needs_act_reinject = act_fault_injector is not None
        needs_weight_reinject = weight_fault_injector is not None

        if needs_act_reinject:
            model = act_fault_injector.remove(model)

        if needs_weight_reinject:
            model = weight_fault_injector.remove(model)

        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_acc": best_acc,
            "current_acc": current_acc,
            "config": self.config,
            "phase_index": phase_index,
            "phase_name": phase_name,
            "phase_local_epoch": phase_local_epoch,
            "total_phases": total_phases,
        }

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        phase_ckpt_dir = self.get_phase_checkpoint_dir(phase_index, phase_name)
        if phase_ckpt_dir is not None:
            checkpoint_path = phase_ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save(checkpoint, checkpoint_path)

            torch.save(checkpoint, phase_ckpt_dir / "latest.pt")

            if is_phase_best and self.save_best:
                phase_best_ckpt = dict(checkpoint)
                if test_acc is not None:
                    phase_best_ckpt["test_acc"] = test_acc
                torch.save(phase_best_ckpt, phase_ckpt_dir / "best.pt")

        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        if is_best and self.save_best:
            global_best_ckpt = dict(checkpoint)
            if test_acc is not None:
                global_best_ckpt["test_acc"] = test_acc

            best_path: Path = self.checkpoint_dir / "best.pt"
            torch.save(global_best_ckpt, best_path)

            if test_acc is not None:
                print(
                    f"  -> New best model saved! (val: {current_acc:.2f}%, test: {test_acc:.2f}%)"
                )
            else:
                print(f"  -> New best model saved! (acc: {current_acc:.2f}%)")

        if needs_act_reinject and act_fault_config is not None:
            model = act_fault_injector.inject(model, act_fault_config)

        if needs_weight_reinject and weight_fault_config is not None:
            model = weight_fault_injector.inject(model, weight_fault_config)

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[Any] = None,
        device: torch.device = torch.device("cpu"),
        strict: bool = True,
    ) -> Tuple[int, float]:
        """Load a model checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            model: Model to load state into.
            optimizer: Optimizer to load state into.
            scheduler: Scheduler to load state into (can be None).
            scaler: GradScaler for AMP (optional).
            device: Device to load the checkpoint to.
            strict: If False, allows loading state dict with missing/unexpected keys.
                    Useful for loading float checkpoints into quantized models.

        Returns:
            Tuple of (start_epoch, best_acc).
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return 0, 0.0

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)

        # Load model state dict with strict parameter
        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint["model_state_dict"], strict=strict
            )

            # Log results if non-strict loading
            if not strict:
                if missing_keys:
                    print(
                        f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}..."
                    )
                if unexpected_keys:
                    print(
                        f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}..."
                    )

        except RuntimeError as e:
            if strict:
                raise
            print(f"Warning: Non-strict loading encountered error: {e}")

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state if using AMP
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch: int = checkpoint["epoch"] + 1
        best_acc: float = checkpoint.get("best_acc", 0.0)

        self._resumed_phase_index = checkpoint.get("phase_index", 0)
        self._resumed_phase_name = checkpoint.get("phase_name", "")

        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")
        return start_epoch, best_acc

    def get_resumed_phase_index(self) -> int:
        """Get the phase index from the last loaded checkpoint."""
        return self._resumed_phase_index

    def get_resumed_phase_name(self) -> str:
        """Get the phase name from the last loaded checkpoint."""
        return self._resumed_phase_name

    def get_phase_dir(self, phase_index: int) -> Optional[Path]:
        """Get the directory for a specific phase.

        Args:
            phase_index: Index of the phase (0-based).

        Returns:
            Path to phase directory, or None if disabled.
        """
        if not self.enabled:
            return None
        if hasattr(self, "phase_dirs") and phase_index in self.phase_dirs:
            return self.phase_dirs[phase_index]
        if self.experiment_dir is None:
            return None
        return self.experiment_dir / f"{phase_index}_phase_{phase_index}"

    def get_phase_checkpoint_dir(
        self, phase_index: int, phase_name: str = ""
    ) -> Optional[Path]:
        """Get the checkpoint directory for a specific phase.

        Args:
            phase_index: Index of the phase (0-based).
            phase_name: Name of the phase (optional, used for fallback).

        Returns:
            Path to phase checkpoint directory, or None if disabled.
        """
        if not self.enabled:
            return None
        if hasattr(self, "phase_dirs") and phase_index in self.phase_dirs:
            return self.phase_dirs[phase_index] / "checkpoints"
        if self.experiment_dir is None:
            return None
        phase_dir = self.experiment_dir / f"{phase_index}_{phase_name}" / "checkpoints"
        phase_dir.mkdir(parents=True, exist_ok=True)
        return phase_dir

    def should_save(self, epoch: int, is_best: bool = False) -> bool:
        """Check if a checkpoint should be saved at this epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            is_best: Whether this is the best model.

        Returns:
            True if checkpoint should be saved.
        """
        if not self.enabled:
            return False
        return (epoch + 1) % self.save_frequency == 0 or is_best

    def get_experiment_dir(self) -> Optional[Path]:
        """Get the experiment directory.

        Returns:
            Path to experiment directory, or None if disabled.
        """
        return self.experiment_dir

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Remove old periodic checkpoints, keeping only the last N.

        Args:
            keep_last_n: Number of recent checkpoints to keep.

        Note:
            Does not remove best.pt or latest.pt.
        """
        if self.checkpoint_dir is None:
            return

        # Find all epoch checkpoints
        checkpoints: list[Path] = sorted(self.checkpoint_dir.glob("epoch_*.pt"))

        # Keep best.pt and latest.pt, remove old epoch checkpoints
        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt.name}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ExperimentManager:
        """Create an ExperimentManager from configuration.

        Args:
            config: Full configuration dictionary.

        Returns:
            Configured ExperimentManager instance.
        """
        checkpoint_config: Dict[str, Any] = config.get("checkpoint", {})
        return cls(
            config=config,
            enabled=checkpoint_config.get("enabled", False),
            base_dir=checkpoint_config.get("dir", "./experiments"),
            experiment_name=checkpoint_config.get("experiment_name"),
            save_frequency=checkpoint_config.get("save_frequency", 10),
            save_best=checkpoint_config.get("save_best", True),
        )
