# FAT - Training Framework

A modular PyTorch training framework for image classification, with support for CIFAR-10/100, MNIST, and FashionMNIST datasets.

```{warning}
Python Version Tested with Python 3.12.3
```

```{toctree}
:hidden:
:includehidden:
:maxdepth: 2
:caption: Getting Started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:hidden:
:includehidden:
:maxdepth: 2
:caption: Framework Components

models/index
datasets/index
configuration/index
```

```{toctree}
:hidden:
:includehidden:
:maxdepth: 3
:caption: Fault Injection Documentation

fault_injection/overview
fault_injection/injector
fault_injection/wrapper
fault_injection/layers
fault_injection/config
fault_injection/statistics
fault_injection/functions
fault_injection/strategies/index
fault_injection/strategies/base
fault_injection/strategies/random
fault_injection/strategies/lsb_flip
fault_injection/strategies/msb_flip
fault_injection/strategies/full_flip
fault_injection/api
```

```{toctree}
:hidden:
:includehidden:
:maxdepth: 1
:caption: Other

benchmarks
```

## Features

- **Modular architecture** with pluggable components (optimizer, scheduler, loss, logging, checkpoints)
- Multiple model architectures (ResNet, VGG, MobileNet, CNV)
- **Quantization-aware training (QAT)** with Brevitas for all model architectures
- **Fault-aware training (FAT)** for training models robust to hardware faults
- **Fault injection framework** for evaluating model resilience to activation errors
- **Configurable loss functions** including cross-entropy, squared hinge, and more
- Multiple datasets with automatic configuration
- Reproducible training with seed control
- Learning rate warmup with multiple scheduler options
- Mixed precision training (AMP) for faster training on CUDA
- TensorBoard logging
- Organized experiment directories with auto-generated names
- Checkpoint saving and resuming
- Validation split support
- Progress bars with tqdm

## Quick Example

Create a YAML configuration file `configs/my_experiment.yaml`:

```yaml title="configs/my_experiment.yaml"
dataset:
  name: "cifar10"
  root: "./data"

model:
  name: "resnet20"

training:
  batch_size: 128
  epochs: 200

optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9

scheduler:
  name: "cosine"
  T_max: 200
  warmup_epochs: 5
```

Then run the training script:

```bash
python train.py --config configs/my_experiment.yaml
```

## Project Structure

```
fat/
├── configs/              # YAML configuration files
├── datasets/             # Dataset implementations
├── models/               # Model architectures
│   ├── resnet_cifar/     # ResNet for CIFAR (20, 32, 44, 56, 110)
│   ├── resnet_imagenet/  # ResNet for ImageNet (18, 34, 50, 101, 152)
│   └── quantized/        # Quantized models (Brevitas)
├── utils/                # Training utilities
│   ├── losses/           # Custom loss functions
│   ├── trainer.py        # Training loop
│   ├── optimizer.py      # Optimizer factory
│   ├── scheduler.py      # Scheduler factory
│   └── ...
├── experiments/          # Auto-generated experiment directories
└── train.py              # Main entry point
```

## Next Steps

- **[Installation](getting-started/installation.md)**: Install the framework and its dependencies  

- **[Quick Start](getting-started/quickstart.md)**: Train your first model in minutes  

- **[Models](models/index.md)**: Explore available model architectures

- **[Configuration](configuration/index.md)**: Learn about all configuration options

- **[API Reference](api/index.md)**: Complete API documentation


