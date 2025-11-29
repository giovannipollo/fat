"""!
@file utils/trainer.py
@brief Trainer class for training and evaluating models.

@details Provides a complete training loop with support for:
- Progress bars with tqdm
- Checkpoint saving/loading
- Learning rate warmup
- Mixed precision training (AMP)
- TensorBoard logging
- Validation and test evaluation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Dict, Iterator, Optional, Tuple, Union

from .optimizer import OptimizerFactory
from .scheduler import SchedulerFactory, SchedulerType
from .experiment import ExperimentManager
from .logging import MetricsLogger


class Trainer:
    """!
    @brief Trainer class for training and evaluating models.
    
    @details Orchestrates the training process using modular components:
    - OptimizerFactory: Creates optimizers from config
    - SchedulerFactory: Creates schedulers with warmup support
    - ExperimentManager: Handles checkpoints and experiment organization
    - MetricsLogger: Handles TensorBoard and console logging
    
    @par Features
    - Progress bars with tqdm
    - Checkpoint saving/loading with meaningful experiment names
    - Learning rate warmup
    - Mixed precision training (AMP)
    - TensorBoard logging
    - Config saving for experiment reproducibility
    
    @par Usage
    @code{.py}
    trainer = Trainer(model, train_loader, test_loader, config, device)
    trainer.train()
    @endcode
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        test_loader: DataLoader[Any],
        config: Dict[str, Any],
        device: torch.device,
        val_loader: Optional[DataLoader[Any]] = None,
    ) -> None:
        """!
        @brief Initialize the trainer.
        
        @param model The model to train (will be moved to device)
        @param train_loader DataLoader for training data
        @param test_loader DataLoader for test data
        @param config Full configuration dictionary
        @param device Compute device (cuda/mps/cpu)
        @param val_loader Optional DataLoader for validation data
        """
        self.model: nn.Module = model.to(device)
        self.train_loader: DataLoader[Any] = train_loader
        self.val_loader: Optional[DataLoader[Any]] = val_loader
        self.test_loader: DataLoader[Any] = test_loader
        self.config: Dict[str, Any] = config
        self.device: torch.device = device

        # Training state
        self.start_epoch: int = 0
        self.best_acc: float = 0.0

        # Setup loss function
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        
        # Setup optimizer and scheduler using factories
        self.optimizer: torch.optim.Optimizer = OptimizerFactory.create(self.model, config)
        self.scheduler: SchedulerType = SchedulerFactory.create(self.optimizer, config)

        # Mixed precision training (AMP)
        amp_config: Dict[str, Any] = config.get("amp", {})
        self.use_amp: bool = amp_config.get("enabled", False) and device.type == "cuda"
        self.scaler: Optional[GradScaler] = GradScaler() if self.use_amp else None

        # Setup experiment manager for checkpoints
        self.experiment: ExperimentManager = ExperimentManager.from_config(config)

        # Setup metrics logger
        self.logger: MetricsLogger = MetricsLogger.from_config(
            config,
            tensorboard_dir=self.experiment.get_tensorboard_dir(),
            experiment_dir=self.experiment.get_experiment_dir(),
        )

        # Progress bar settings
        progress_config: Dict[str, Any] = config.get("progress", {})
        self.show_progress: bool = progress_config.get("enabled", True)

        # Resume from checkpoint if specified
        checkpoint_config: Dict[str, Any] = config.get("checkpoint", {})
        resume_path: Optional[str] = checkpoint_config.get("resume")
        if resume_path:
            self.start_epoch, self.best_acc = self.experiment.load_checkpoint(
                resume_path,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.device,
            )

    @property
    def has_validation(self) -> bool:
        """!
        @brief Check if validation loader is available.
        @return True if validation data is available
        """
        return self.val_loader is not None

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """!
        @brief Train for one epoch.
        
        @param epoch Current epoch number (for progress bar display)
        @return Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Create progress bar
        if self.show_progress:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            pbar = self.train_loader

        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.use_amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            if self.show_progress:
                current_acc = 100.0 * correct / total
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.2f}%"}
                )

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, loader: DataLoader[Any], desc: str = "Evaluating") -> Tuple[float, float]:
        """!
        @brief Evaluate model on a given dataloader.
        
        @param loader DataLoader to evaluate on
        @param desc Description for progress bar
        @return Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # Create progress bar for evaluation
        if self.show_progress:
            pbar = tqdm(
                loader,
                desc=desc,
                leave=False,
                dynamic_ncols=True,
            )
        else:
            pbar = loader

        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Use AMP for inference too (optional, for consistency)
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """!
        @brief Evaluate model on validation set.
        
        @return Tuple of (average_loss, accuracy)
        @throws ValueError If no validation set is available
        """
        if self.val_loader is None:
            raise ValueError("No validation set available")
        return self.evaluate(self.val_loader, desc="Validating")

    def test(self) -> Tuple[float, float]:
        """!
        @brief Evaluate model on test set.
        
        @return Tuple of (average_loss, accuracy)
        """
        return self.evaluate(self.test_loader, desc="Testing")

    def train(self) -> None:
        """!
        @brief Run the full training loop.
        
        @details Executes training for the configured number of epochs,
        handling validation/test evaluation, checkpointing, and logging.
        """
        epochs: int = self.config["training"]["epochs"]
        model_name: str = self.config["model"]["name"]
        eval_name: str = "Val" if self.has_validation else "Test"
        
        # Test evaluation frequency (only when using validation set)
        training_config: Dict[str, Any] = self.config.get("training", {})
        test_frequency: int = training_config.get("test_frequency", 10)

        # Log training start
        self.logger.log_training_start(
            model_name=model_name,
            epochs=epochs,
            device=str(self.device),
            has_validation=self.has_validation,
            use_amp=self.use_amp,
            experiment_dir=self.experiment.get_experiment_dir(),
        )

        for epoch in range(self.start_epoch, epochs):
            train_loss, train_acc = self.train_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = self.config["optimizer"]["learning_rate"]

            # Evaluate on validation or test set
            if self.has_validation:
                eval_loss, eval_acc = self.validate()
            else:
                eval_loss, eval_acc = self.test()

            # Check if this is the best model
            is_best = eval_acc > self.best_acc
            if is_best:
                self.best_acc = eval_acc

            # Evaluate on test set when:
            # 1. We find a new best model (when using validation), OR
            # 2. It's a periodic test evaluation epoch (when using validation)
            test_loss, test_acc = None, None
            if self.has_validation:
                is_periodic_test = test_frequency > 0 and (
                    (epoch + 1) % test_frequency == 0 or epoch == epochs - 1
                )
                if is_best or is_periodic_test:
                    test_loss, test_acc = self.test()

            # Log epoch metrics
            self.logger.log_epoch(
                epoch=epoch,
                total_epochs=epochs,
                lr=lr,
                train_loss=train_loss,
                train_acc=train_acc,
                eval_loss=eval_loss,
                eval_acc=eval_acc,
                eval_name=eval_name,
                test_loss=test_loss,
                test_acc=test_acc,
            )

            # Save checkpoint
            if self.experiment.should_save(epoch, is_best):
                self.experiment.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    best_acc=self.best_acc,
                    current_acc=eval_acc,
                    scaler=self.scaler,
                    is_best=is_best,
                    test_acc=test_acc if is_best and self.has_validation else None,
                )
                
                # Log best model to file
                if is_best:
                    self.logger.log_best_model(
                        epoch=epoch,
                        val_acc=eval_acc,
                        test_acc=test_acc if self.has_validation else None,
                    )

        # Final evaluation on test set (if we used validation during training)
        if self.has_validation:
            print("\nRunning final evaluation on test set...")
            final_test_loss, final_test_acc = self.test()
            self.logger.log_final_test(final_test_loss, final_test_acc, epochs)

        # Log completion
        self.logger.log_training_complete(self.best_acc, eval_name)
        
        experiment_dir = self.experiment.get_experiment_dir()
        if experiment_dir is not None:
            print(f"Results saved to: {experiment_dir}")

        # Close logger
        self.logger.close()
