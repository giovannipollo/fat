# Quick Start

This guide will help you train your first model using the training framework.

## Basic Training

Run training with the default configuration:

```bash
python train.py
```

Use a custom configuration file:

```bash
python train.py --config configs/resnet20_cifar10.yaml
```

## Your First Experiment

### 1. Create a Configuration File

Create a new file `configs/my_first_experiment.yaml`:

```yaml title="configs/my_first_experiment.yaml"
seed:
  enabled: true
  value: 42

dataset:
  name: "cifar10"
  root: "./data"
  download: true
  num_workers: 4

model:
  name: "resnet20"

training:
  batch_size: 128
  epochs: 100

optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0001

scheduler:
  name: "cosine"
  T_max: 100
  warmup_epochs: 5

checkpoint:
  enabled: true
  dir: "./experiments"
  save_best: true
```

### 2. Run Training

```bash
python train.py --config configs/my_first_experiment.yaml
```

### 3. Monitor Progress

You'll see output like:

```
Random seed: 42 (deterministic: False)
Using device: cuda
Dataset: cifar10 (10 classes, 3 channels)
Model: resnet20

Starting training resnet20 for 100 epochs...

Epoch 1/100: 100%|██████████| 391/391 [00:12<00:00, 31.2it/s, loss=1.5234, acc=45.23%]
Epoch [1/100] LR: 0.0200 | Train Loss: 1.5234 | Train Acc: 45.23% | Test Acc: 52.10%
...
```

## Common Configurations

### Fast Test Run

For quick testing with minimal epochs:

```yaml
model:
  name: "mobilenetv1"  # Lightweight model

training:
  batch_size: 128
  epochs: 5

scheduler:
  name: "none"

checkpoint:
  enabled: false
```

### High Accuracy Training

For best results on CIFAR-10:

```yaml
model:
  name: "resnet56"

training:
  batch_size: 128
  epochs: 200

optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0005

scheduler:
  name: "cosine"
  T_max: 200
  warmup_epochs: 5

amp:
  enabled: true  # Faster training on CUDA
```

### Quantized Model Training

For quantization-aware training:

```yaml
model:
  name: "quant_resnet20"  # Use quant_ prefix

quantization:
  weight_bit_width: 4
  act_bit_width: 4

loss:
  name: "sqr_hinge"  # Common for quantized networks

amp:
  enabled: false  # Disable AMP for quantized training
```

## Experiment Output

When checkpoints are enabled, the framework creates organized experiment directories:

```
experiments/
└── resnet20_cifar10_20240115_143022/
    ├── config.yaml           # Saved configuration
    ├── checkpoints/
    │   ├── epoch_0050.pt     # Periodic checkpoints
    │   ├── latest.pt         # Most recent
    │   └── best.pt           # Best accuracy
    └── tensorboard/          # TensorBoard logs
```

## TensorBoard Visualization

Enable TensorBoard in your config:

```yaml
tensorboard:
  enabled: true
```

Then view the logs:

```bash
tensorboard --logdir=./experiments
```

## Next Steps

- Learn about available [Models](../models/index.md)
- Explore [Configuration Options](../configuration/index.md)
