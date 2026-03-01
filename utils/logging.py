"""Metrics logging utilities for training.

Provides unified logging to console and text files
for tracking training progress and metrics.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Type


class MetricsLogger:
    """Handles logging of training metrics to multiple outputs.

    Provides unified interface for logging to:

    - Console (formatted epoch summaries)
    - Text file (training_log.txt)

    Example:
        ```python
        logger = MetricsLogger.from_config(config, experiment_dir)
        logger.log_epoch(epoch, total_epochs, lr, train_loss, train_acc, val_loss, val_acc)
        logger.close()
        ```
    """

    def __init__(
        self,
        console_enabled: bool = True,
        file_logging_enabled: bool = False,
        experiment_dir: Optional[Path] = None,
    ):
        """Initialize the metrics logger.

        Args:
            console_enabled: Whether to print metrics to console.
            file_logging_enabled: Whether to log to a text file.
            experiment_dir: Directory for the log file.
        """
        self.console_enabled = console_enabled
        self.file_logging_enabled = file_logging_enabled
        self.log_file: Optional[TextIO] = None
        self.log_file_path: Optional[Path] = None

        # Setup file logging
        if file_logging_enabled and experiment_dir is not None:
            self.log_file_path = experiment_dir / "training_log.txt"
            self.log_file = open(self.log_file_path, "w")
            self._write_file_header()

    def _write_file_header(self) -> None:
        """Write header information to the log file."""
        if self.log_file is None:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"Training Log - Started at {timestamp}\n")
        self.log_file.write("=" * 80 + "\n\n")
        self.log_file.flush()

    def _write_to_file(self, message: str) -> None:
        """Write a message to the log file.

        Args:
            message: Message string to write.
        """
        if self.log_file is not None:
            self.log_file.write(message + "\n")
            self.log_file.flush()

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        lr: float,
        train_loss: float,
        train_acc: float,
        val_loss: Optional[float],
        val_acc: Optional[float],
        test_loss: Optional[float] = None,
        test_acc: Optional[float] = None,
        is_best: bool = False,
    ) -> None:
        """Log metrics for an epoch.

        Args:
            epoch: Current epoch (0-indexed).
            total_epochs: Total number of epochs.
            lr: Current learning rate.
            train_loss: Training loss.
            train_acc: Training accuracy (%).
            val_loss: Validation loss (None if no validation set).
            val_acc: Validation accuracy (%) (None if no validation set).
            test_loss: Test loss (optional, evaluated periodically with validation).
            test_acc: Test accuracy (%) (optional, evaluated periodically with validation).
            is_best: Whether this epoch achieved the best accuracy.
        """
        # Build log message
        log_msg = (
            f"Epoch [{epoch + 1}/{total_epochs}] "
            f"LR: {lr:.5f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )

        # Add validation metrics if available
        if val_loss is not None and val_acc is not None:
            log_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"

        # Add test metrics if available
        if test_loss is not None and test_acc is not None:
            log_msg += f" | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"

        # Add best model indicator
        if is_best:
            log_msg += " [BEST]"

        # Console logging
        if self.console_enabled:
            print(log_msg)

        # File logging
        self._write_to_file(log_msg)

    def log_final_test(self, loss: float, accuracy: float, epoch: int) -> None:
        """Log final test results.

        Args:
            loss: Final test loss.
            accuracy: Final test accuracy (%).
            epoch: Final epoch number.
        """
        msg = f"Final Test Loss: {loss:.4f} | Final Test Accuracy: {accuracy:.2f}%"

        if self.console_enabled:
            print(msg)

        self._write_to_file(msg)

    def log_best_model(
        self,
        epoch: int,
        val_acc: Optional[float],
        test_acc: Optional[float] = None,
    ) -> None:
        """Log when a new best model is saved.

        Args:
            epoch: Epoch number (0-indexed).
            val_acc: Validation accuracy (None if no validation set).
            test_acc: Test accuracy (always available since we always evaluate test).
        """
        if val_acc is not None and test_acc is not None:
            msg = f"  -> New best model saved at epoch {epoch + 1}! (val: {val_acc:.2f}%, test: {test_acc:.2f}%)"
        elif test_acc is not None:
            msg = f"  -> New best model saved at epoch {epoch + 1}! (test: {test_acc:.2f}%)"
        else:
            msg = f"  -> New best model saved at epoch {epoch + 1}!"

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
    ) -> None:
        """Log training start information.

        Args:
            model_name: Name of the model.
            epochs: Number of epochs.
            device: Device being used.
            has_validation: Whether validation set is available.
            use_amp: Whether AMP is enabled.
            experiment_dir: Experiment directory path.
        """
        # Build messages
        messages: List[str] = [
            f"\nStarting training {model_name} for {epochs} epochs...",
            f"Using device: {device}",
        ]
        if has_validation:
            messages.append("Validation set: enabled")
        if use_amp:
            messages.append("Mixed precision training (AMP): enabled")
        if experiment_dir is not None:
            messages.append(f"Experiment directory: {experiment_dir}")
        messages.append("")

        # Console logging
        if self.console_enabled:
            for msg in messages:
                print(msg)

        # File logging
        for msg in messages:
            self._write_to_file(msg)

    def log_training_complete(self, best_acc: float, monitor_name: str = "validation") -> None:
        """Log training completion message.

        Args:
            best_acc: Best accuracy achieved (%).
            monitor_name: Name of the metric being monitored ("validation" or "test").
        """
        msg = f"\nTraining complete! Best {monitor_name} accuracy: {best_acc:.2f}%"

        if self.console_enabled:
            print(msg)

        self._write_to_file(msg)

    def close(self) -> None:
        """Close the log file."""
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def __enter__(self) -> MetricsLogger:
        """Context manager entry.

        Returns:
            Self.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Context manager exit - closes the writer.

        Returns:
            False (don't suppress exceptions).
        """
        self.close()
        return False

    @classmethod
    def from_config(
        cls: Type[MetricsLogger],
        config: Dict[str, Any],
        experiment_dir: Optional[Path] = None,
    ) -> MetricsLogger:
        """Create a MetricsLogger from configuration.

        Args:
            config: Full configuration dictionary.
            experiment_dir: Directory for file logging.

        Returns:
            Configured MetricsLogger instance.
        """
        progress_config: Dict[str, Any] = config.get("progress", {})
        logging_config: Dict[str, Any] = config.get("logging", {})

        file_logging_enabled: bool = logging_config.get(
            "file_enabled", experiment_dir is not None
        )

        return cls(
            console_enabled=progress_config.get("enabled", True),
            file_logging_enabled=file_logging_enabled,
            experiment_dir=experiment_dir,
        )
