# FAT - Training Framework

A modular PyTorch training framework for image classification, with support for standard and quantized models.

> [!WARNING]  
> Tested with Python 3.12.3

## Features

- Multiple architectures: ResNet, VGG, MobileNet, CNV
- Quantization-aware training (QAT) with Brevitas
- Multiple datasets: CIFAR-10/100, MNIST, Fashion-MNIST
- Configurable loss functions, optimizers, and schedulers
- Mixed precision training (AMP)
- TensorBoard logging
- Checkpoint saving and resuming
- Reproducible training with seed control

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd training-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Train with default configuration:

```bash
python train.py
```

Use a custom configuration:

```bash
python train.py --config configs/resnet20_cifar10.yaml
```

Example configuration:

```yaml
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
```

## Documentation

Full documentation is available in the `docs/` directory. To view it locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

### Documentation Contents

- [Getting Started](docs/getting-started/installation.md) - Installation and quick start guide
- [Models](docs/models/index.md) - Available architectures and how to add new ones
- [Datasets](docs/datasets/index.md) - Supported datasets and how to add new ones
- [Configuration](docs/configuration/index.md) - Complete configuration reference
- [API Reference](docs/api/index.md) - Python API documentation

## Project Structure

```
fat/
├── configs/          # YAML configuration files
├── datasets/         # Dataset implementations
├── models/           # Model architectures
│   ├── standard/         # Full-precision models
│   │   ├── resnet_cifar/     # ResNet for CIFAR (20, 32, 44, 56, 110)
│   │   └── resnet_imagenet/  # ResNet for ImageNet (18, 34, 50, 101, 152)
│   └── quantized/        # Quantized models (Brevitas)
├── utils/            # Training utilities
├── docs/             # MkDocs documentation
└── train.py          # Main entry point
```

## License

See [LICENSE](LICENSE) for details.
