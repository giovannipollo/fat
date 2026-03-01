"""Experiment manager for checkpoints and experiment organization.

Handles experiment directory structure, checkpoint saving,
and configuration persistence for reproducibility.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import yaml as yaml_module
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
    - Config saving for reproducibility
    - Checkpoint saving with best model tracking

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

        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self._save_config()

    def _save_config(self) -> None:
        """Save configuration to the experiment directory.

        Saves as YAML if available, otherwise as text representation.
        """
        if self.experiment_dir is None:
            return

        config_path: Path = self.experiment_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml_module.dump(
                self.config, f, default_flow_style=False, sort_keys=False
            )

        print(f"Config saved to: {config_path}")

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        val_acc: Optional[float],
        best_val_acc: Optional[float],
        best_test_acc: float,
        is_best: bool = False,
        test_acc: Optional[float] = None,
        act_fault_injector: Optional["ActivationFaultInjector"] = None,
        weight_fault_injector: Optional["WeightFaultInjector"] = None,
        act_fault_config: Optional[Any] = None,
        weight_fault_config: Optional[Any] = None,
    ):
        """Save a model checkpoint.

        Args:
            epoch: Current epoch number (0-indexed).
            model: The model to save.
            val_acc: Current epoch's validation accuracy (None if no validation).
            best_val_acc: Best validation accuracy achieved so far (None if no validation).
            best_test_acc: Best test accuracy achieved so far.
            is_best: Whether this is the best model so far.
            test_acc: Current epoch's test accuracy (None if not evaluated this epoch).
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
            "best_test_acc": best_test_acc,
            "config": self.config,
        }
        
        # Add validation metrics if available
        if val_acc is not None:
            checkpoint["val_acc"] = val_acc
        
        if best_val_acc is not None:
            checkpoint["best_val_acc"] = best_val_acc
        
        # Add test accuracy if evaluated this epoch
        if test_acc is not None:
            checkpoint["test_acc"] = test_acc

        # Save periodic checkpoint
        checkpoint_path: Path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save latest checkpoint (always overwritten)
        latest_path: Path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best and self.save_best:
            best_path: Path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            
            if val_acc is not None and test_acc is not None:
                print(
                    f"  -> New best model saved! (val: {val_acc:.2f}%, test: {test_acc:.2f}%)"
                )
            elif test_acc is not None:
                print(f"  -> New best model saved! (test: {test_acc:.2f}%)")
            else:
                print(f"  -> New best model saved!")

        # Re-inject fault injection wrappers/hooks after saving
        if needs_act_reinject and act_fault_config is not None:
            model = act_fault_injector.inject(model, act_fault_config)

        if needs_weight_reinject and weight_fault_config is not None:
            model = weight_fault_injector.inject(model, weight_fault_config)

    def load_weights(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        strict: bool = False,
    ) -> None:
        """Load only model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (absolute or relative to experiment_dir).
            model: Model to load weights into.
            device: Device to load the checkpoint to.
            strict: If False, allows loading state dict with missing/unexpected keys.
        """
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.is_absolute():
            if self.experiment_dir is not None:
                ckpt_file = self.experiment_dir / checkpoint_path
            else:
                raise FileNotFoundError(
                    f"Cannot resolve relative path: {checkpoint_path}"
                )

        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        print(f"Loading weights from: {ckpt_file}")

        ckpt = torch.load(str(ckpt_file), map_location=device)

        missing_keys, unexpected_keys = model.load_state_dict(
            ckpt["model_state_dict"], strict=strict
        )

        if missing_keys:
            print(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(
                f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}..."
            )

        print("Loaded model weights")

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
