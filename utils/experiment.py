"""Experiment manager for handling checkpoints, config saving, and experiment organization."""

import shutil
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ExperimentManager:
    """
    Manages experiment directories, checkpoints, and configuration saving.
    
    Features:
    - Auto-generated experiment names based on model/dataset/timestamp
    - Organized directory structure: experiments/<name>/checkpoints/, tensorboard/
    - Config saving for reproducibility
    - Checkpoint saving/loading with best model tracking
    """

    def __init__(
        self,
        config: dict,
        enabled: bool = True,
        base_dir: str = "./experiments",
        experiment_name: Optional[str] = None,
        save_frequency: int = 10,
        save_best: bool = True,
    ):
        """
        Initialize the experiment manager.
        
        Args:
            config: Full configuration dictionary
            enabled: Whether checkpointing is enabled
            base_dir: Base directory for experiments
            experiment_name: Optional custom experiment name
            save_frequency: How often to save periodic checkpoints (in epochs)
            save_best: Whether to save the best model
        """
        self.config = config
        self.enabled = enabled
        self.save_frequency = save_frequency
        self.save_best = save_best
        
        # Directories
        self.experiment_dir: Optional[Path] = None
        self.checkpoint_dir: Optional[Path] = None
        self.tensorboard_dir: Optional[Path] = None
        
        if enabled:
            self._setup_experiment_dir(base_dir, experiment_name)

    def _generate_experiment_name(self, custom_name: Optional[str] = None) -> str:
        """Generate a meaningful experiment name based on config."""
        model_name = self.config["model"]["name"].lower()
        dataset_name = self.config["dataset"]["name"].lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if custom_name:
            return f"{custom_name}_{model_name}_{dataset_name}_{timestamp}"
        else:
            return f"{model_name}_{dataset_name}_{timestamp}"

    def _setup_experiment_dir(
        self,
        base_dir: str,
        custom_name: Optional[str] = None,
    ):
        """Setup experiment directory structure."""
        base_path = Path(base_dir)
        experiment_name = self._generate_experiment_name(custom_name)
        
        self.experiment_dir = base_path / experiment_name
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.tensorboard_dir = self.experiment_dir / "tensorboard"
        
        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self._save_config()

    def _save_config(self):
        """Save the configuration to the experiment directory."""
        if self.experiment_dir is None:
            return
        
        if YAML_AVAILABLE:
            import yaml as yaml_module
            config_path = self.experiment_dir / "config.yaml"
            with open(config_path, "w") as f:
                yaml_module.dump(self.config, f, default_flow_style=False, sort_keys=False)
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
    ):
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch number
            model: The model to save
            optimizer: The optimizer state
            scheduler: The scheduler state (can be None)
            best_acc: Best accuracy achieved so far (val or test depending on config)
            current_acc: Current epoch's accuracy
            scaler: GradScaler for AMP (optional)
            is_best: Whether this is the best model so far
            test_acc: Test set accuracy (only for best model when using validation)
        """
        if not self.enabled or self.checkpoint_dir is None:
            return
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_acc": best_acc,
            "current_acc": current_acc,
            "config": self.config,
        }
        
        # Save scaler state if using AMP
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        
        # Save periodic checkpoint
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint (always overwritten)
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best and self.save_best:
            # Include test accuracy in best checkpoint
            if test_acc is not None:
                checkpoint["test_acc"] = test_acc
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            if test_acc is not None:
                print(f"  -> New best model saved! (val: {current_acc:.2f}%, test: {test_acc:.2f}%)")
            else:
                print(f"  -> New best model saved! (acc: {current_acc:.2f}%)")

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[Any] = None,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[int, float]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into (can be None)
            scaler: GradScaler for AMP (optional)
            device: Device to load the checkpoint to
            
        Returns:
            Tuple of (start_epoch, best_acc)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return 0, 0.0
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load scaler state if using AMP
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")
        return start_epoch, best_acc

    def should_save(self, epoch: int, is_best: bool = False) -> bool:
        """Check if a checkpoint should be saved at this epoch."""
        if not self.enabled:
            return False
        return (epoch + 1) % self.save_frequency == 0 or is_best

    def get_tensorboard_dir(self) -> Optional[Path]:
        """Get the TensorBoard log directory."""
        return self.tensorboard_dir

    def get_experiment_dir(self) -> Optional[Path]:
        """Get the experiment directory."""
        return self.experiment_dir

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Remove old periodic checkpoints, keeping only the last N.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        if self.checkpoint_dir is None:
            return
        
        # Find all epoch checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        
        # Keep best.pt and latest.pt, remove old epoch checkpoints
        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt.name}")

    @classmethod
    def from_config(cls, config: dict) -> "ExperimentManager":
        """
        Create an ExperimentManager from a configuration dictionary.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Configured ExperimentManager instance
        """
        checkpoint_config = config.get("checkpoint", {})
        return cls(
            config=config,
            enabled=checkpoint_config.get("enabled", False),
            base_dir=checkpoint_config.get("dir", "./experiments"),
            experiment_name=checkpoint_config.get("experiment_name"),
            save_frequency=checkpoint_config.get("save_frequency", 10),
            save_best=checkpoint_config.get("save_best", True),
        )
