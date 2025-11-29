"""Trainer class for training and evaluating models."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional

from .optimizer import OptimizerFactory
from .scheduler import SchedulerFactory
from .experiment import ExperimentManager
from .logging import MetricsLogger


class Trainer:
    """
    Trainer class for training and evaluating models.

    Features:
    - Progress bars with tqdm
    - Checkpoint saving/loading with meaningful experiment names
    - Learning rate warmup
    - Mixed precision training (AMP)
    - TensorBoard logging
    - Config saving for experiment reproducibility
    
    Uses modular components:
    - OptimizerFactory: Creates optimizers from config
    - SchedulerFactory: Creates schedulers with warmup support
    - ExperimentManager: Handles checkpoints and experiment organization
    - MetricsLogger: Handles TensorBoard and console logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        test_loader,
        config: dict,
        device: torch.device,
        val_loader=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # Training state
        self.start_epoch = 0
        self.best_acc = 0.0

        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer and scheduler using factories
        self.optimizer = OptimizerFactory.create(self.model, config)
        self.scheduler = SchedulerFactory.create(self.optimizer, config)

        # Mixed precision training (AMP)
        amp_config = config.get("amp", {})
        self.use_amp = amp_config.get("enabled", False) and device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Setup experiment manager for checkpoints
        self.experiment = ExperimentManager.from_config(config)

        # Setup metrics logger
        self.logger = MetricsLogger.from_config(
            config,
            tensorboard_dir=self.experiment.get_tensorboard_dir(),
            experiment_dir=self.experiment.get_experiment_dir(),
        )

        # Progress bar settings
        progress_config = config.get("progress", {})
        self.show_progress = progress_config.get("enabled", True)

        # Resume from checkpoint if specified
        checkpoint_config = config.get("checkpoint", {})
        resume_path = checkpoint_config.get("resume")
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
        """Check if validation loader is available."""
        return self.val_loader is not None

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number (for progress bar display)

        Returns:
            tuple: (average_loss, accuracy)
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
    def evaluate(self, loader, desc: str = "Evaluating") -> tuple[float, float]:
        """
        Evaluate model on a given dataloader.

        Args:
            loader: DataLoader to evaluate on
            desc: Description for progress bar

        Returns:
            tuple: (average_loss, accuracy)
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

    def validate(self) -> tuple[float, float]:
        """
        Evaluate model on validation set.

        Returns:
            tuple: (average_loss, accuracy)

        Raises:
            ValueError: If no validation set is available
        """
        if not self.has_validation:
            raise ValueError("No validation set available")
        return self.evaluate(self.val_loader, desc="Validating")

    def test(self) -> tuple[float, float]:
        """
        Evaluate model on test set.

        Returns:
            tuple: (average_loss, accuracy)
        """
        return self.evaluate(self.test_loader, desc="Testing")

    def train(self):
        """Run the full training loop."""
        epochs = self.config["training"]["epochs"]
        model_name = self.config["model"]["name"]
        eval_name = "Val" if self.has_validation else "Test"
        
        # Test evaluation frequency (only when using validation set)
        training_config = self.config.get("training", {})
        test_frequency = training_config.get("test_frequency", 10)

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
