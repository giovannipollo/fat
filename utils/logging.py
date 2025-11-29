"""Metrics logging utilities for TensorBoard, console, and file output."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, TextIO

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore
    TENSORBOARD_AVAILABLE = False


class MetricsLogger:
    """
    Handles logging of training metrics to TensorBoard, console, and file.
    
    Features:
    - TensorBoard scalar logging
    - Console output formatting
    - Text file logging (training_log.txt)
    - Epoch summary logging
    """

    def __init__(
        self,
        tensorboard_enabled: bool = False,
        log_dir: Optional[Path] = None,
        console_enabled: bool = True,
        file_logging_enabled: bool = False,
        experiment_dir: Optional[Path] = None,
    ):
        """
        Initialize the metrics logger.
        
        Args:
            tensorboard_enabled: Whether to enable TensorBoard logging
            log_dir: Directory for TensorBoard logs
            console_enabled: Whether to print metrics to console
            file_logging_enabled: Whether to log to a text file
            experiment_dir: Directory for the log file
        """
        self.console_enabled = console_enabled
        self.tensorboard_enabled = tensorboard_enabled
        self.file_logging_enabled = file_logging_enabled
        self.writer: Optional[Any] = None
        self.log_file: Optional[TextIO] = None
        self.log_file_path: Optional[Path] = None
        
        # Setup TensorBoard
        if tensorboard_enabled:
            if not TENSORBOARD_AVAILABLE or SummaryWriter is None:
                print(
                    "Warning: TensorBoard not available. "
                    "Install with: pip install tensorboard"
                )
                self.tensorboard_enabled = False
            elif log_dir is not None:
                log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(log_dir))
                print(f"TensorBoard logging enabled: {log_dir}")
        
        # Setup file logging
        if file_logging_enabled and experiment_dir is not None:
            self.log_file_path = experiment_dir / "training_log.txt"
            self.log_file = open(self.log_file_path, "w")
            self._write_file_header()

    def _write_file_header(self):
        """Write header information to the log file."""
        if self.log_file is None:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"Training Log - Started at {timestamp}\n")
        self.log_file.write("=" * 80 + "\n\n")
        self.log_file.flush()

    def _write_to_file(self, message: str):
        """Write a message to the log file."""
        if self.log_file is not None:
            self.log_file.write(message + "\n")
            self.log_file.flush()

    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value.
        
        Args:
            tag: Name of the metric
            value: Value to log
            step: Current step/epoch
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: int):
        """
        Log multiple scalar values.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Current step/epoch
        """
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        lr: float,
        train_loss: float,
        train_acc: float,
        eval_loss: float,
        eval_acc: float,
        eval_name: str = "Val",
        test_loss: Optional[float] = None,
        test_acc: Optional[float] = None,
    ):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            lr: Current learning rate
            train_loss: Training loss
            train_acc: Training accuracy
            eval_loss: Evaluation loss (validation or test)
            eval_acc: Evaluation accuracy (validation or test)
            eval_name: Name of the evaluation set ("Val" or "Test")
            test_loss: Optional test loss (when using validation set)
            test_acc: Optional test accuracy (when using validation set)
        """
        # Build log message
        log_msg = (
            f"Epoch [{epoch + 1}/{total_epochs}] "
            f"LR: {lr:.5f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"{eval_name} Loss: {eval_loss:.4f} | "
            f"{eval_name} Acc: {eval_acc:.2f}%"
        )
        
        # Add test metrics if available
        if test_loss is not None and test_acc is not None:
            log_msg += f" | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        
        # Console logging
        if self.console_enabled:
            print(log_msg)
        
        # File logging
        self._write_to_file(log_msg)
        
        # TensorBoard logging
        if self.writer is not None:
            self.writer.add_scalar("Learning Rate", lr, epoch)
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar(f"Loss/{eval_name.lower()}", eval_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar(f"Accuracy/{eval_name.lower()}", eval_acc, epoch)
            
            # Log test metrics if available
            if test_loss is not None and test_acc is not None:
                self.writer.add_scalar("Loss/test", test_loss, epoch)
                self.writer.add_scalar("Accuracy/test", test_acc, epoch)

    def log_final_test(self, loss: float, accuracy: float, epoch: int):
        """
        Log final test results.
        
        Args:
            loss: Final test loss
            accuracy: Final test accuracy
            epoch: Final epoch number
        """
        msg = f"Final Test Loss: {loss:.4f} | Final Test Accuracy: {accuracy:.2f}%"
        
        if self.console_enabled:
            print(msg)
        
        self._write_to_file(msg)
        
        if self.writer is not None:
            self.writer.add_scalar("Loss/test_final", loss, epoch)
            self.writer.add_scalar("Accuracy/test_final", accuracy, epoch)

    def log_best_model(
        self,
        epoch: int,
        val_acc: float,
        test_acc: Optional[float] = None,
    ):
        """
        Log when a new best model is saved.
        
        Args:
            epoch: Epoch number (0-indexed)
            val_acc: Validation accuracy (or test if no validation set)
            test_acc: Optional test accuracy (when using validation set)
        """
        if test_acc is not None:
            msg = f"  -> New best model saved at epoch {epoch + 1}! (val: {val_acc:.2f}%, test: {test_acc:.2f}%)"
        else:
            msg = f"  -> New best model saved at epoch {epoch + 1}! (acc: {val_acc:.2f}%)"
        
        # Only write to file, console output is handled by ExperimentManager
        self._write_to_file(msg)

    def log_training_start(
        self,
        model_name: str,
        epochs: int,
        device: str,
        has_validation: bool = False,
        use_amp: bool = False,
        experiment_dir: Optional[Path] = None,
    ):
        """
        Log training start information.
        
        Args:
            model_name: Name of the model
            epochs: Number of epochs
            device: Device being used
            has_validation: Whether validation set is available
            use_amp: Whether AMP is enabled
            experiment_dir: Experiment directory path
        """
        # Build messages
        messages = [
            f"\nStarting training {model_name} for {epochs} epochs...",
            f"Using device: {device}",
        ]
        if has_validation:
            messages.append("Validation set: enabled")
        if use_amp:
            messages.append("Mixed precision training (AMP): enabled")
        if self.writer is not None:
            messages.append(f"TensorBoard logging: enabled (log_dir: {self.writer.log_dir})")
        if experiment_dir is not None:
            messages.append(f"Experiment directory: {experiment_dir}")
        messages.append("")  # Empty line
        
        # Console logging
        if self.console_enabled:
            for msg in messages:
                print(msg)
        
        # File logging
        for msg in messages:
            self._write_to_file(msg)

    def log_training_complete(self, best_acc: float, eval_name: str = "Val"):
        """
        Log training completion message.
        
        Args:
            best_acc: Best accuracy achieved
            eval_name: Name of the evaluation set
        """
        msg = f"\nTraining complete! Best {eval_name} accuracy: {best_acc:.2f}%"
        
        if self.console_enabled:
            print(msg)
        
        self._write_to_file(msg)

    def close(self):
        """Close the TensorBoard writer and log file."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the writer."""
        self.close()
        return False

    @classmethod
    def from_config(
        cls,
        config: dict,
        tensorboard_dir: Optional[Path] = None,
        experiment_dir: Optional[Path] = None,
    ) -> "MetricsLogger":
        """
        Create a MetricsLogger from configuration.
        
        Args:
            config: Full configuration dictionary
            tensorboard_dir: Override directory for TensorBoard logs
            experiment_dir: Directory for file logging
            
        Returns:
            Configured MetricsLogger instance
        """
        tensorboard_config = config.get("tensorboard", {})
        progress_config = config.get("progress", {})
        logging_config = config.get("logging", {})
        
        # Determine log directory
        log_dir = tensorboard_dir
        if log_dir is None and tensorboard_config.get("enabled", False):
            log_dir = Path(tensorboard_config.get("log_dir", "./runs"))
        
        # File logging is enabled by default if experiment_dir is provided
        file_logging_enabled = logging_config.get("file_enabled", experiment_dir is not None)
        
        return cls(
            tensorboard_enabled=tensorboard_config.get("enabled", False),
            log_dir=log_dir,
            console_enabled=progress_config.get("enabled", True),
            file_logging_enabled=file_logging_enabled,
            experiment_dir=experiment_dir,
        )
