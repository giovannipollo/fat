# Configuration

All training parameters are configured via YAML files. This section provides a complete reference for all available options.

## Configuration Sections

| Section | Description |
|---------|-------------|
| [Dataset](dataset.md) | Dataset selection and loading |
| [Model](model.md) | Model architecture |
| [Loss](loss.md) | Loss function |
| [Optimizer](optimizer.md) | Optimization algorithm |
| [Scheduler](scheduler.md) | Learning rate scheduling |
| [Training](training.md) | Training hyperparameters |
| [Checkpoints](checkpoints.md) | Saving and resuming |
| [Fault Injection](fault_injection.md) | Fault-aware training and evaluation |

## Quick Reference

```yaml title="Complete configuration example"
# Reproducibility
seed:
  enabled: true
  value: 42
  deterministic: false

# Dataset
dataset:
  name: "cifar10"
  root: "./data"
  download: true
  num_workers: 4
  val_split: 0.1

# Model
model:
  name: "resnet20"

# Quantization (for quant_* models)
quantization:
  weight_bit_width: 4
  act_bit_width: 4

# Loss function
loss:
  name: "cross_entropy"
  label_smoothing: 0.1

# Training
training:
  batch_size: 128
  epochs: 200

# Optimizer
optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0005

# Scheduler
scheduler:
  name: "cosine"
  T_max: 200
  warmup_epochs: 5

# Mixed precision
amp:
  enabled: true

# Fault Injection (for quantized models)
fault_injection:
  enabled: false
  probability: 5.0
  mode: "full_model"
  injection_type: "random"
  apply_during: "train"

# Logging
tensorboard:
  enabled: true

progress:
  enabled: true

# Checkpoints
checkpoint:
  enabled: true
  dir: "./experiments"
  save_frequency: 10
  save_best: true
```

## Seed Configuration

Controls reproducibility of training runs.

```yaml
seed:
  enabled: true          # Enable seed setting
  value: 42              # Random seed value
  deterministic: false   # Deterministic algorithms (slower)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Whether to set random seeds |
| `value` | int | `42` | Seed for random, numpy, torch |
| `deterministic` | bool | `false` | Use deterministic CUDA operations |

## AMP Configuration

Automatic Mixed Precision for faster training.

```yaml
amp:
  enabled: true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable mixed precision (CUDA only) |

!!! warning
    Disable AMP for quantized model training.

## TensorBoard Configuration

Training visualization.

```yaml
tensorboard:
  enabled: true
  log_dir: "./runs"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable TensorBoard logging |
| `log_dir` | string | `"./runs"` | Log directory |

## Progress Configuration

Progress bar display.

```yaml
progress:
  enabled: true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Show tqdm progress bars |

## Minimal Configuration

The minimal required configuration:

```yaml
dataset:
  name: "cifar10"

model:
  name: "resnet20"

training:
  batch_size: 128
  epochs: 100

optimizer:
  name: "sgd"
  learning_rate: 0.1
```

All other parameters use sensible defaults.

## Next Steps

Explore each configuration section in detail:

- [Dataset Configuration](dataset.md)
- [Model Configuration](model.md)
- [Loss Configuration](loss.md)
- [Optimizer Configuration](optimizer.md)
- [Scheduler Configuration](scheduler.md)
- [Training Configuration](training.md)
- [Checkpoint Configuration](checkpoints.md)
- [Fault Injection Configuration](fault_injection.md)
