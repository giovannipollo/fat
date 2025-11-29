# Training Framework

A modular PyTorch training framework for image classification, with support for CIFAR-10/100, MNIST, and FashionMNIST datasets.

> [!WARNING]  
> Tested with python 3.12.3

## Features

- **Modular architecture** with pluggable components (optimizer, scheduler, logging, checkpoints)
- Multiple model architectures (ResNet, VGG, MobileNet)
- Multiple datasets with automatic configuration
- Reproducible training with seed control
- Learning rate warmup with multiple scheduler options
- Mixed precision training (AMP) for faster training on CUDA
- TensorBoard logging
- Organized experiment directories with auto-generated names
- Checkpoint saving and resuming
- Validation split support
- Progress bars with tqdm

## Project Structure

```
training-framework/
├── configs/
│   ├── default.yaml          # Default training configuration
│   ├── quick_test.yaml       # Fast iteration testing
│   ├── debug.yaml            # Debugging configuration
│   ├── fast_amp_training.yaml # Mixed precision training
│   ├── resnet18_cifar10.yaml # ResNet-18 on CIFAR-10
│   ├── resnet50_cifar100.yaml # ResNet-50 on CIFAR-100
│   ├── resnet152_cifar100.yaml # ResNet-152 on CIFAR-100
│   ├── vgg16_cifar10.yaml    # VGG-16 on CIFAR-10
│   ├── mobilenetv1_cifar10.yaml # MobileNetV1 on CIFAR-10
│   ├── resnet18_mnist.yaml   # ResNet-18 on MNIST
│   └── resnet18_fashion_mnist.yaml # ResNet-18 on FashionMNIST
├── datasets/
│   ├── __init__.py           # Dataset registry and utilities
│   ├── base.py               # BaseDataset abstract class
│   ├── cifar10.py            # CIFAR-10 dataset
│   ├── cifar100.py           # CIFAR-100 dataset
│   ├── mnist.py              # MNIST dataset
│   └── fashion_mnist.py      # FashionMNIST dataset
├── models/
│   ├── __init__.py           # Model registry
│   ├── mobilenet.py          # MobileNetV1 architecture
│   ├── resnet_base.py        # ResNet base class and blocks
│   ├── resnet18.py           # ResNet-18
│   ├── resnet34.py           # ResNet-34
│   ├── resnet50.py           # ResNet-50
│   ├── resnet101.py          # ResNet-101
│   ├── resnet152.py          # ResNet-152
│   └── vgg.py                # VGG architectures (11, 13, 16, 19)
├── utils/
│   ├── __init__.py           # Exports all utilities
│   ├── config.py             # YAML config loader
│   ├── device.py             # Device detection (CUDA/MPS/CPU)
│   ├── experiment.py         # Experiment management & checkpoints
│   ├── logging.py            # TensorBoard & console logging
│   ├── optimizer.py          # Optimizer factory
│   ├── scheduler.py          # Scheduler factory with warmup
│   ├── seed.py               # Reproducibility utilities
│   └── trainer.py            # Training loop orchestration
├── experiments/              # Auto-generated experiment directories
├── data/                     # Downloaded datasets
├── runs/                     # TensorBoard logs (when not using experiments)
└── train.py                  # Main entry point
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd training-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision pyyaml tqdm

# Optional: TensorBoard for logging
pip install tensorboard
```

## Quick Start

Run training with default configuration:

```bash
python train.py
```

Use a custom configuration file:

```bash
python train.py --config configs/my_experiment.yaml
```

## Supported Models

| Model | Name | Parameters | Description |
|-------|------|------------|-------------|
| MobileNetV1 | `mobilenetv1` | ~3.2M | Lightweight, efficient |
| ResNet-18 | `resnet18` | ~11.2M | Good balance of speed/accuracy |
| ResNet-34 | `resnet34` | ~21.3M | Deeper ResNet |
| ResNet-50 | `resnet50` | ~23.5M | Bottleneck architecture |
| ResNet-101 | `resnet101` | ~42.5M | Very deep ResNet |
| ResNet-152 | `resnet152` | ~58.1M | Deepest ResNet |
| VGG-11 | `vgg11` | ~9.2M | Classic VGG |
| VGG-13 | `vgg13` | ~9.4M | Deeper VGG |
| VGG-16 | `vgg16` | ~14.7M | Popular VGG variant |
| VGG-19 | `vgg19` | ~20.0M | Deepest VGG |

All models are adapted for CIFAR-sized images (32x32) and support both RGB and grayscale inputs.

## Supported Datasets

| Dataset | Name | Classes | Image Size | Channels |
|---------|------|---------|------------|----------|
| CIFAR-10 | `cifar10` | 10 | 32x32 | 3 (RGB) |
| CIFAR-100 | `cifar100` | 100 | 32x32 | 3 (RGB) |
| MNIST | `mnist` | 10 | 28x28 | 1 (Grayscale) |
| FashionMNIST | `fashion_mnist` | 10 | 28x28 | 1 (Grayscale) |

## Configuration

All training parameters are configured via YAML files. Here's the full configuration reference:

```yaml
# Reproducibility
seed:
  enabled: true         # Enable for reproducible results
  value: 42             # Random seed value
  deterministic: false  # Use deterministic algorithms (slower but fully reproducible)

# Dataset Configuration
dataset:
  name: "cifar10"       # Options: cifar10, cifar100, mnist, fashion_mnist
  root: "./data"        # Dataset download directory
  download: true        # Download if not present
  num_workers: 16       # DataLoader workers
  val_split: 0.1        # Optional: fraction of training data for validation (0.0-1.0)
  seed: 42              # Optional: random seed for reproducible validation split

# Model Configuration
model:
  name: "resnet18"      # See "Supported Models" table above
  # num_classes: 10     # Optional: auto-detected from dataset
  # in_channels: 3      # Optional: auto-detected from dataset

# Training Hyperparameters
training:
  batch_size: 128
  epochs: 200

# Optimizer Configuration
optimizer:
  name: "sgd"           # Options: sgd, adam, adamw
  learning_rate: 0.1
  momentum: 0.9         # For SGD
  weight_decay: 5e-4

# Learning Rate Scheduler
scheduler:
  name: "cosine"        # Options: cosine, step, multistep, exponential, plateau, none
  T_max: 200            # For cosine (typically same as epochs)
  warmup_epochs: 5      # Linear warmup epochs (0 to disable)
  # step_size: 30       # For step scheduler
  # milestones: [30, 60, 90]  # For multistep scheduler
  # gamma: 0.1          # For step/multistep/exponential scheduler
  # factor: 0.1         # For plateau scheduler
  # patience: 10        # For plateau scheduler

# Mixed Precision Training (AMP)
amp:
  enabled: false        # Enable for faster training on CUDA GPUs

# TensorBoard Logging
tensorboard:
  enabled: false
  log_dir: "./runs"

# Progress Bar
progress:
  enabled: true

# Checkpoints
checkpoint:
  enabled: true
  dir: "./experiments"    # Base directory for experiments
  experiment_name: ""     # Optional custom experiment name
  save_frequency: 10      # Save every N epochs
  save_best: true         # Save best model by validation/test accuracy
  # resume: "./experiments/resnet18_cifar10_20240101_120000/checkpoints/latest.pt"
```

### Experiment Directory Structure

When checkpoints are enabled, the framework automatically creates organized experiment directories:

```
experiments/
└── resnet18_cifar10_20240115_143022/    # Auto-generated name
    ├── config.yaml                       # Saved configuration
    ├── checkpoints/
    │   ├── epoch_0010.pt                 # Periodic checkpoints
    │   ├── epoch_0020.pt
    │   ├── latest.pt                     # Most recent checkpoint
    │   └── best.pt                       # Best model checkpoint
    └── tensorboard/                      # TensorBoard logs
        └── events.out.tfevents.*
```

The experiment name is auto-generated as `{model}_{dataset}_{timestamp}` or you can provide a custom name via `experiment_name` in the config.

## Reproducibility

For reproducible experiments, enable the seed configuration:

```yaml
seed:
  enabled: true
  value: 42
  deterministic: false
```

This sets seeds for:
- Python's `random` module
- NumPy (if installed)
- PyTorch (CPU and CUDA)

### Deterministic Mode

For fully reproducible results, enable `deterministic: true`:

```yaml
seed:
  enabled: true
  value: 42
  deterministic: true
```

Note: Deterministic mode may reduce performance as it disables certain CUDA optimizations.

## Learning Rate Warmup

Warmup gradually increases the learning rate from 0 to the target value over a specified number of epochs. This can help training stability, especially with large batch sizes.

```yaml
scheduler:
  name: "cosine"
  T_max: 200
  warmup_epochs: 5  # 5 epochs of linear warmup
```

During warmup:
- Epoch 1: LR = base_lr * 0.2
- Epoch 2: LR = base_lr * 0.4
- Epoch 3: LR = base_lr * 0.6
- Epoch 4: LR = base_lr * 0.8
- Epoch 5: LR = base_lr * 1.0
- Epoch 6+: Normal scheduler (cosine/step)

## Mixed Precision Training (AMP)

Automatic Mixed Precision uses float16 for forward/backward passes while keeping float32 for parameter updates. This provides:
- 2-3x faster training on modern NVIDIA GPUs
- Reduced memory usage
- No accuracy loss (with proper scaling)

```yaml
amp:
  enabled: true  # Only works on CUDA devices
```

## TensorBoard Logging

Enable TensorBoard to visualize training metrics:

```yaml
tensorboard:
  enabled: true
  log_dir: "./runs"
```

Then run TensorBoard:

```bash
tensorboard --logdir=./runs
```

Logged metrics:
- Learning rate per epoch
- Training loss per epoch
- Training accuracy per epoch
- Validation/Test accuracy per epoch

## Validation

The framework supports splitting the training data into train/validation sets. This is useful for:
- Hyperparameter tuning without touching the test set
- Early stopping based on validation performance
- More reliable model selection

### Enable Validation Split

Add `val_split` to your dataset configuration:

```yaml
dataset:
  name: "cifar10"
  root: "./data"
  val_split: 0.1    # Use 10% of training data for validation
  seed: 42          # For reproducible splits
```

### How It Works

- When `val_split` is set, the training data is split into train and validation sets
- The validation set uses **test transforms** (no augmentation) for proper evaluation
- During training, the model is evaluated on the validation set instead of the test set
- After training completes, a final evaluation is run on the test set
- Checkpoints are saved based on validation accuracy (when enabled)

### Without Validation

If `val_split` is not set (default), the model is evaluated on the test set during training.

## Checkpoints

When checkpoints are enabled, the following files are saved in the experiment directory:

| File | Description |
|------|-------------|
| `epoch_0010.pt` | Periodic checkpoint at epoch 10 |
| `latest.pt` | Most recent epoch (always updated) |
| `best.pt` | Best model by validation/test accuracy |
| `config.yaml` | Saved configuration for reproducibility |

### Resume Training

To resume training from a checkpoint, add the `resume` field to your config:

```yaml
checkpoint:
  enabled: true
  dir: "./experiments"
  resume: "./experiments/resnet18_cifar10_20240115_143022/checkpoints/latest.pt"
```

### Checkpoint Contents

Each checkpoint contains:
- `model_state_dict` - Model weights
- `optimizer_state_dict` - Optimizer state
- `scheduler_state_dict` - Learning rate scheduler state
- `scaler_state_dict` - AMP scaler state (if using AMP)
- `epoch` - Epoch number
- `best_acc` - Best accuracy achieved
- `config` - Training configuration

## Examples

### Train ResNet-18 on CIFAR-10 with AMP and TensorBoard

```yaml
dataset:
  name: "cifar10"
  root: "./data"
  download: true
  num_workers: 8
  val_split: 0.1

model:
  name: "resnet18"

training:
  batch_size: 128
  epochs: 200

optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 5e-4

scheduler:
  name: "cosine"
  T_max: 200
  warmup_epochs: 5

amp:
  enabled: true

tensorboard:
  enabled: true
  log_dir: "./runs/resnet18_cifar10"

checkpoint:
  enabled: true
  dir: "./checkpoints/resnet18_cifar10"
  save_frequency: 20
  save_best: true
```

### Train VGG-16 on CIFAR-100

```yaml
dataset:
  name: "cifar100"
  root: "./data"
  download: true
  num_workers: 8

model:
  name: "vgg16"

training:
  batch_size: 128
  epochs: 200

optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 5e-4

scheduler:
  name: "cosine"
  T_max: 200
  warmup_epochs: 5

checkpoint:
  enabled: true
  dir: "./checkpoints/vgg16_cifar100"
```

### Train on MNIST

```yaml
dataset:
  name: "mnist"
  root: "./data"
  download: true
  num_workers: 8

model:
  name: "resnet18"
  # in_channels auto-detected as 1 for grayscale

training:
  batch_size: 256
  epochs: 50

optimizer:
  name: "adam"
  learning_rate: 0.001
  weight_decay: 0

scheduler:
  name: "cosine"
  T_max: 50

checkpoint:
  enabled: true
  dir: "./checkpoints/resnet18_mnist"
```

### Quick Test Run

```yaml
dataset:
  name: "cifar10"
  root: "./data"
  download: true
  num_workers: 4

model:
  name: "mobilenetv1"

training:
  batch_size: 128
  epochs: 5

optimizer:
  name: "adam"
  learning_rate: 0.001

scheduler:
  name: "none"

checkpoint:
  enabled: false
```

## Adding Custom Datasets

The framework uses a class-based design that makes it easy to add custom datasets. All datasets extend the `BaseDataset` abstract class.

### Step 1: Create Dataset Class

Create a new file `datasets/my_dataset.py`:

```python
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class MyCustomDataset(BaseDataset):
    """My custom dataset."""
    
    # Required: Dataset metadata
    name = "my_dataset"
    num_classes = 10
    in_channels = 3
    image_size = (32, 32)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    
    def _build_train_transform(self) -> transforms.Compose:
        """Optional: Override to customize training augmentations."""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def _build_test_transform(self) -> transforms.Compose:
        """Optional: Override to customize test transforms."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def _load_dataset(
        self,
        root: str,
        train: bool,
        download: bool,
        transform: transforms.Compose,
    ) -> Dataset:
        """Required: Load and return the PyTorch dataset."""
        return torchvision.datasets.MyDataset(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
```

### Step 2: Register the Dataset

Add your dataset to `datasets/__init__.py`:

```python
from .my_dataset import MyCustomDataset

DATASETS = {
    "cifar10": CIFAR10Dataset,
    "cifar100": CIFAR100Dataset,
    "mnist": MNISTDataset,
    "fashion_mnist": FashionMNISTDataset,
    "my_dataset": MyCustomDataset,  # Add your dataset here
}
```

Or use the decorator to auto-register:

```python
from datasets import register_dataset, BaseDataset

@register_dataset
class MyCustomDataset(BaseDataset):
    name = "my_dataset"
    # ... rest of the implementation
```

### Step 3: Use Your Dataset

```yaml
dataset:
  name: "my_dataset"
  root: "./data"
  download: true
  num_workers: 8

model:
  name: "resnet18"
  # num_classes and in_channels auto-detected from your dataset!
```

### BaseDataset API Reference

```python
class BaseDataset(ABC):
    # Required class attributes
    name: str                    # Dataset identifier (e.g., "cifar10")
    num_classes: int             # Number of output classes
    in_channels: int             # Number of input channels (1 or 3)
    image_size: Tuple[int, int]  # Image dimensions (height, width)
    mean: Tuple[float, ...]      # Normalization mean
    std: Tuple[float, ...]       # Normalization std
    
    # Required method
    def _load_dataset(self, root, train, download, transform) -> Dataset:
        """Load and return the PyTorch dataset."""
        pass
    
    # Optional methods (override to customize)
    def _build_train_transform(self) -> transforms.Compose:
        """Build training transforms (default: ToTensor + Normalize)."""
        pass
    
    def _build_test_transform(self) -> transforms.Compose:
        """Build test transforms (default: ToTensor + Normalize)."""
        pass
    
    # Available methods
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Returns (train_loader, test_loader)."""
        pass
```

## Adding New Models

1. Create a new file in `models/`, e.g., `models/my_model.py`:

```python
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        # Your model architecture here
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            # ... more layers
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.classifier(x)
        return x
```

2. Register it in `models/__init__.py`:

```python
from .mobilenet import MobileNetV1
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .vgg import VGG11, VGG13, VGG16, VGG19
from .my_model import MyModel

MODELS = {
    "mobilenetv1": MobileNetV1,
    "resnet18": ResNet18,
    # ... other models
    "mymodel": MyModel,
}
```

3. Use it in your config:

```yaml
model:
  name: "mymodel"
```

## Modular Architecture

The training framework uses a modular design with separate components for different concerns. This makes it easy to customize, extend, and test individual parts.

### Component Overview

| Component | File | Description |
|-----------|------|-------------|
| `OptimizerFactory` | `utils/optimizer.py` | Creates optimizers from config |
| `SchedulerFactory` | `utils/scheduler.py` | Creates schedulers with warmup support |
| `WarmupScheduler` | `utils/scheduler.py` | Wraps schedulers with linear warmup |
| `ExperimentManager` | `utils/experiment.py` | Manages experiment directories and checkpoints |
| `MetricsLogger` | `utils/logging.py` | Handles TensorBoard and console logging |
| `Trainer` | `utils/trainer.py` | Orchestrates training using all components |

### Using Components Independently

You can use individual components for custom training loops:

```python
from utils import (
    OptimizerFactory,
    SchedulerFactory,
    ExperimentManager,
    MetricsLogger,
)

# Create optimizer from config
optimizer = OptimizerFactory.create(model, config)

# Create scheduler with warmup
scheduler = SchedulerFactory.create(optimizer, config)

# Setup experiment directory and checkpoints
experiment = ExperimentManager.from_config(config)

# Setup logging
logger = MetricsLogger.from_config(config, experiment.get_tensorboard_dir())

# Custom training loop
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
    val_acc = evaluate(model, val_loader)
    
    scheduler.step()
    
    # Log metrics
    logger.log_epoch(epoch, epochs, scheduler.get_last_lr()[0], 
                     train_loss, train_acc, val_acc)
    
    # Save checkpoint
    if experiment.should_save(epoch, is_best=val_acc > best_acc):
        experiment.save_checkpoint(
            epoch, model, optimizer, scheduler, 
            best_acc, val_acc, is_best=val_acc > best_acc
        )

logger.close()
```

### OptimizerFactory

Supports the following optimizers:

| Optimizer | Config Name | Key Parameters |
|-----------|-------------|----------------|
| SGD | `sgd` | `momentum`, `nesterov`, `weight_decay` |
| Adam | `adam` | `betas`, `eps`, `weight_decay` |
| AdamW | `adamw` | `betas`, `eps`, `weight_decay` |

```python
from utils import OptimizerFactory

# List available optimizers
print(OptimizerFactory.available_optimizers())  # ['sgd', 'adam', 'adamw']

# Create optimizer
optimizer = OptimizerFactory.create(model, config)
```

### SchedulerFactory

Supports the following schedulers:

| Scheduler | Config Name | Key Parameters |
|-----------|-------------|----------------|
| Cosine Annealing | `cosine` | `T_max`, `eta_min` |
| Step LR | `step` | `step_size`, `gamma` |
| MultiStep LR | `multistep` | `milestones`, `gamma` |
| Exponential LR | `exponential` | `gamma` |
| Reduce on Plateau | `plateau` | `mode`, `factor`, `patience`, `min_lr` |
| None | `none` | - |

All schedulers can be combined with warmup:

```yaml
scheduler:
  name: "cosine"
  warmup_epochs: 5  # 5 epochs of linear warmup before cosine decay
  T_max: 200
```

```python
from utils import SchedulerFactory

# List available schedulers
print(SchedulerFactory.available_schedulers())

# Create scheduler (automatically wraps with warmup if configured)
scheduler = SchedulerFactory.create(optimizer, config)
```

### ExperimentManager

Handles experiment organization:

```python
from utils import ExperimentManager

# Create from config
experiment = ExperimentManager.from_config(config)

# Or create manually
experiment = ExperimentManager(
    config=config,
    enabled=True,
    base_dir="./experiments",
    experiment_name="my_experiment",  # Optional custom name
    save_frequency=10,
    save_best=True,
)

# Get directories
print(experiment.get_experiment_dir())   # Path to experiment root
print(experiment.get_tensorboard_dir())  # Path for TensorBoard logs

# Save checkpoint
experiment.save_checkpoint(
    epoch=10,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    best_acc=95.5,
    current_acc=94.2,
    scaler=scaler,  # Optional, for AMP
    is_best=False,
)

# Load checkpoint
start_epoch, best_acc = experiment.load_checkpoint(
    "path/to/checkpoint.pt",
    model, optimizer, scheduler, scaler, device
)

# Clean up old checkpoints (keep last 5)
experiment.cleanup_old_checkpoints(keep_last_n=5)
```

### MetricsLogger

Handles console and TensorBoard logging:

```python
from utils import MetricsLogger

# Create from config
logger = MetricsLogger.from_config(config, tensorboard_dir)

# Or create manually
logger = MetricsLogger(
    tensorboard_enabled=True,
    log_dir=Path("./runs/experiment"),
    console_enabled=True,
)

# Log training start
logger.log_training_start(
    model_name="resnet18",
    epochs=200,
    device="cuda",
    has_validation=True,
    use_amp=True,
    experiment_dir=Path("./experiments/my_exp"),
)

# Log epoch metrics
logger.log_epoch(
    epoch=10,
    total_epochs=200,
    lr=0.01,
    train_loss=0.5,
    train_acc=85.0,
    eval_acc=82.0,
    eval_name="Val",
)

# Log individual scalars
logger.log_scalar("custom_metric", 0.95, step=10)
logger.log_scalars({"metric1": 0.5, "metric2": 0.8}, step=10)

# Log final test accuracy
logger.log_final_test(accuracy=93.5, epoch=200)

# Log completion
logger.log_training_complete(best_acc=94.2, eval_name="Val")

# Close (important for TensorBoard)
logger.close()

# Or use as context manager
with MetricsLogger.from_config(config, tensorboard_dir) as logger:
    # ... training loop
    pass  # Automatically closes
```
