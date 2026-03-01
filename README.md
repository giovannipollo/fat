# FAT - Training Framework

A modular PyTorch Fault-awaretraining (FAT) framework for image classification, with support for standard and quantized models.

> [!WARNING]  
> Tested with Python 3.12.3

## Features

- Multiple architectures: ResNet, MobileNet, CNV
- Quantization-aware training (QAT) with Brevitas
- Multiple datasets: CIFAR-10/100, MNIST, Fashion-MNIST, ImageNet
- Configurable loss functions, optimizers, and schedulers
- Mixed precision training (AMP)
- TensorBoard logging
- Checkpoint saving and resuming
- Reproducible training with seed control

## Installation

```bash
# Clone the repository
git clone https://github.com/giovannipollo/fat
cd fat

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
python train.py --config your_config.yaml
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

Full documentation is available in the `docs/` directory