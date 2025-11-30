# Dataset Configuration

Configure the dataset for training.

## Basic Configuration

```yaml
dataset:
  name: "cifar10"
  root: "./data"
  download: true
```

## Full Configuration

```yaml
dataset:
  name: "cifar10"        # Dataset name (required)
  root: "./data"         # Storage directory
  download: true         # Download if missing
  num_workers: 4         # DataLoader workers
  val_split: 0.1         # Validation split ratio
  seed: 42               # Split random seed
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | **required** | Dataset identifier |
| `root` | string | `"./data"` | Directory for dataset storage |
| `download` | bool | `true` | Automatically download dataset |
| `num_workers` | int | `4` | Number of data loading workers |
| `val_split` | float | `null` | Fraction for validation (0.0-1.0) |
| `seed` | int | `42` | Random seed for validation split |

## Available Datasets

| Name | Classes | Size | Channels |
|------|---------|------|----------|
| `cifar10` | 10 | 32×32 | 3 (RGB) |
| `cifar100` | 100 | 32×32 | 3 (RGB) |
| `mnist` | 10 | 28×28 | 1 (Gray) |
| `fashion_mnist` | 10 | 28×28 | 1 (Gray) |

## Examples

### CIFAR-10 with Validation

```yaml
dataset:
  name: "cifar10"
  root: "./data"
  download: true
  num_workers: 8
  val_split: 0.1
  seed: 42
```

### MNIST

```yaml
dataset:
  name: "mnist"
  root: "./data"
  download: true
  num_workers: 4
```

### Fast Loading

```yaml
dataset:
  name: "cifar10"
  root: "./data"
  num_workers: 16  # More workers for faster loading
```

## Validation Split

When `val_split` is set:

1. Training data is split into train and validation sets
2. Validation set uses test transforms (no augmentation)
3. Model is evaluated on validation during training
4. Final test evaluation after training completes

```yaml
dataset:
  name: "cifar10"
  val_split: 0.1    # 10% validation, 90% training
  seed: 42          # Reproducible split
```

## Auto-Detection

Model parameters are automatically inferred:

```yaml
dataset:
  name: "cifar100"  # 100 classes, 3 channels

model:
  name: "resnet50"
  # num_classes: 100  (auto-detected)
  # in_channels: 3    (auto-detected)
```

Override if needed:

```yaml
model:
  name: "resnet50"
  num_classes: 100   # Explicit override
  in_channels: 3
```

## See Also

- [Datasets Overview](../datasets/index.md)
- [Configuration Overview](index.md)
