# Quick Start Guide

This guide will help you train your first model with the FAT training framework.

## Basic Training

Train a model with default configuration:

```bash
python train.py
```
This will use the default configuration file and training parameters.

## Custom Configuration

Specify a custom configuration file:

```bash
python train.py --config configs/resnet20_cifar10.yaml
```

## Configuration File

Create a configuration file to specify training parameters:

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

## Fault Injection Example

Enable fault injection during training:

```yaml
fault_injection:
  enabled: true
  probability: 5.0
  injection_type: "lsb_flip"
  apply_during: "train"
  track_statistics: true
```

For more information about fault injection, see the [Fault Injection Overview](../fault_injection/overview.md) documentation.

## Next Steps

Explore more configuration options in the Configuration section.
