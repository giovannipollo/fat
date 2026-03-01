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
        eval_loss: float,
        eval_acc: float,
        eval_name: str = "Val",
        test_loss: Optional[float] = None,
        test_acc: Optional[float] = None,
        is_best: bool = False,
        phase_name: str = "",
    ) -> None:
        """Log metrics for an epoch.

        Args:
            epoch: Current epoch (0-indexed).
            total_epochs: Total number of epochs.
            lr: Current learning rate.
            train_loss: Training loss.
            train_acc: Training accuracy (%).
            eval_loss: Evaluation loss (validation or test).
            eval_acc: Evaluation accuracy (%).
            eval_name: Name of evaluation set ("Val" or "Test").
            test_loss: Optional test loss (when using validation).
            test_acc: Optional test accuracy (when using validation).
            is_best: Whether this epoch achieved the best accuracy.
            phase_name: Name of the current phase (for labeling).
        """
        phase_prefix = f"[{phase_name}] " if phase_name else ""
        log_msg = (
            f"{phase_prefix}Epoch [{epoch + 1}/{total_epochs}] "
            f"LR: {lr:.5f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"{eval_name} Loss: {eval_loss:.4f} | "
            f"{eval_name} Acc: {eval_acc:.2f}%"
        )

        if test_loss is not None and test_acc is not None:
            log_msg += f" | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"

        if is_best:
            log_msg += " [BEST]"

        if self.console_enabled:
            print(log_msg)

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
        val_acc: float,
        test_acc: Optional[float] = None,
    ) -> None:
        """Log when a new best model is saved.

        Args:
            epoch: Epoch number (0-indexed).
            val_acc: Validation accuracy (or test if no validation).
            test_acc: Optional test accuracy (when using validation).
        """
        if test_acc is not None:
            msg = f"  -> New best model saved at epoch {epoch + 1}! (val: {val_acc:.2f}%, test: {test_acc:.2f}%)"
        else:
            msg = (
                f"  -> New best model saved at epoch {epoch + 1}! (acc: {val_acc:.2f}%)"
            )

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
        num_phases: int = 1,
    ) -> None:
        """Log training start information.

        Args:
            model_name: Name of the model.
            epochs: Number of epochs.
            device: Device being used.
            has_validation: Whether validation set is available.
            use_amp: Whether AMP is enabled.
            experiment_dir: Experiment directory path.
            num_phases: Number of training phases.
        """
        phase_str = (
            f" ({num_phases} phase{'s' if num_phases > 1 else ''})"
            if num_phases > 1
            else ""
        )
        messages: List[str] = [
            f"\nStarting training {model_name} for {epochs} epochs{phase_str}...",
            f"Using device: {device}",
        ]
        if has_validation:
            messages.append("Validation set: enabled")
        if use_amp:
            messages.append("Mixed precision training (AMP): enabled")
        if experiment_dir is not None:
            messages.append(f"Experiment directory: {experiment_dir}")
        messages.append("")

        if self.console_enabled:
            for msg in messages:
                print(msg)

        for msg in messages:
            self._write_to_file(msg)

    def log_phase_start(
        self,
        phase_name: str,
        phase_index: int,
        total_phases: int,
        epochs: int,
        global_epoch_offset: int,
        optimizer_name: str,
        learning_rate: float,
        has_fault_injection: bool = False,
    ) -> None:
        """Log the start of a new training phase.

        Args:
            phase_name: Name of the phase.
            phase_index: Index of the phase (0-based).
            total_phases: Total number of phases.
            epochs: Number of epochs in this phase.
            global_epoch_offset: Starting global epoch for this phase.
            optimizer_name: Name of the optimizer.
            learning_rate: Learning rate for this phase.
            has_fault_injection: Whether fault injection is enabled.
        """
        separator = "=" * 60
        messages = [
            "",
            separator,
            f"Phase {phase_index + 1}/{total_phases}: '{phase_name}'",
            f"  Epochs: {epochs} (global {global_epoch_offset + 1} to {global_epoch_offset + epochs})",
            f"  Optimizer: {optimizer_name}, LR: {learning_rate}",
        ]
        if has_fault_injection:
            messages.append("  Fault injection: enabled")
        messages.append(separator)

        if self.console_enabled:
            for msg in messages:
                print(msg)

        for msg in messages:
            self._write_to_file(msg)

    def log_phase_complete(
        self,
        phase_name: str,
        phase_index: int,
        best_acc: float,
        eval_name: str = "Val",
    ) -> None:
        """Log completion of a training phase.

        Args:
            phase_name: Name of the phase.
            phase_index: Index of the phase (0-based).
            best_acc: Best accuracy achieved in this phase.
            eval_name: Name of the evaluation set.
        """
        msg = f"Phase '{phase_name}' complete. Best {eval_name} acc: {best_acc:.2f}%"

        if self.console_enabled:
            print(msg)

        self._write_to_file(msg)

    def log_training_complete(self, best_acc: float, eval_name: str = "Val") -> None:
        """Log training completion message.

        Args:
            best_acc: Best accuracy achieved (%).
            eval_name: Name of the evaluation set.
        """
        msg = f"\nTraining complete! Best {eval_name} accuracy: {best_acc:.2f}%"

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
        """Context manager exit - closes the log file.

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
