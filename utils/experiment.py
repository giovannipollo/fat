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
    - Directory structure: experiments/<dataset>/<model_timestamp>/checkpoints/, tensorboard/
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
              tensorboard/
                events.out.tfevents.*
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
        self.tensorboard_dir: Optional[Path] = None

        if enabled:
            self._setup_experiment_dir(base_dir, experiment_name)

    def _generate_experiment_name(self, custom_name: Optional[str] = None) -> str:
        """Generate a meaningful experiment name from config.

        Args:
            custom_name: Optional custom prefix.

        Returns:
            Experiment name string (format: [prefix_]model_timestamp).
        """
        model_name: str = self.config["model"]["name"].lower()
        sat_or_fat: str = self._get_sat_or_fat()
        activation_or_weight: str = self._get_activation_or_weight_fault_injection()
        dataset_name: str = self._get_dataset_name()
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        precision: str = self._get_precision()

        if custom_name:
            if activation_or_weight == "":
                return f"{custom_name}_{timestamp}_{model_name}_{precision}_{dataset_name}_{sat_or_fat}"
            else:
                return f"{custom_name}_{timestamp}_{model_name}_{precision}_{dataset_name}_{sat_or_fat}_{activation_or_weight}"
        else:
            if activation_or_weight == "":
                return (
                    f"{timestamp}_{model_name}_{precision}_{dataset_name}_{sat_or_fat}"
                )
            else:
                return f"{timestamp}_{model_name}_{precision}_{dataset_name}_{sat_or_fat}_{activation_or_weight}"

    def _get_dataset_name(self) -> str:
        """Get the dataset name from config.

        Returns:
            Dataset name string in lowercase.
        """
        return self.config["dataset"]["name"].lower()

    def _get_sat_or_fat(self) -> str:
        """Determine if the experiment is SAT or FAT based on config.

        Returns:
            "sat" if no fault injection, "fat" if any fault injection enabled.
        """
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
        """Setup experiment directory structure.

        Creates a hierarchical structure: base_dir/dataset/model_timestamp/

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
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.tensorboard_dir = self.experiment_dir / "tensorboard"

        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # Save config
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
        test_acc: Optional[float] = None,
        phase_info: Optional[Dict[str, Any]] = None,
        act_fault_injector: Optional["ActivationFaultInjector"] = None,
        weight_fault_injector: Optional["WeightFaultInjector"] = None,
        act_fault_config: Optional[Any] = None,
        weight_fault_config: Optional[Any] = None,
    ):
        """Save a model checkpoint.

        Args:
            epoch: Current epoch number (0-indexed).
            model: The model to save.
            optimizer: The optimizer state.
            scheduler: The scheduler state (can be None).
            best_acc: Best accuracy achieved so far.
            current_acc: Current epoch's accuracy.
            scaler: GradScaler for AMP (optional).
            is_best: Whether this is the best model so far.
            test_acc: Test accuracy (for best model with validation).
            phase_info: Optional phase information for multi-phase training.
            act_fault_injector: Optional activation fault injector to remove before saving.
            weight_fault_injector: Optional weight fault injector to remove before saving.
            act_fault_config: Optional activation fault injection config for re-injection.
            weight_fault_config: Optional weight fault injection config for re-injection.
        """
        if not self.enabled or self.checkpoint_dir is None:
            return

        # Remove fault injection wrappers/hooks before saving
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
        }

        # Save phase info if provided
        if phase_info is not None:
            checkpoint["phase_info"] = phase_info

        # Save scaler state if using AMP
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        # Save periodic checkpoint
        # Include phase name in filename for multi-phase training
        if phase_info is not None and phase_info.get("mode") == "multi_phase":
            phase_name = phase_info.get("phase_name", "unknown")
            checkpoint_path: Path = (
                self.checkpoint_dir / f"epoch_{epoch:04d}_{phase_name}.pt"
            )
        else:
            checkpoint_path: Path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save latest checkpoint (always overwritten)
        latest_path: Path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best and self.save_best:
            # Include test accuracy in best checkpoint
            if test_acc is not None:
                checkpoint["test_acc"] = test_acc
            best_path: Path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            if test_acc is not None:
                print(
                    f"  -> New best model saved! (val: {current_acc:.2f}%, test: {test_acc:.2f}%)"
                )
            else:
                print(f"  -> New best model saved! (acc: {current_acc:.2f}%)")

        # Re-inject fault injection wrappers/hooks after saving
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
    ) -> Tuple[int, float, Optional[Dict[str, Any]]]:
        """Load a model checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            model: Model to load state into.
            optimizer: Optimizer to load state into.
            scheduler: Scheduler to load state into (can be None).
            scaler: GradScaler for AMP (optional).
            device: Device to load the checkpoint to.

        Returns:
            Tuple of (start_epoch, best_acc, phase_info).
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return 0, 0.0, None

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state if using AMP
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch: int = checkpoint["epoch"] + 1
        best_acc: float = checkpoint.get("best_acc", 0.0)
        phase_info: Optional[Dict[str, Any]] = checkpoint.get("phase_info", None)

        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")
        return start_epoch, best_acc, phase_info

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

    def get_tensorboard_dir(self) -> Optional[Path]:
        """Get the TensorBoard log directory.

        Returns:
            Path to TensorBoard directory, or None if disabled.
        """
        return self.tensorboard_dir

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
