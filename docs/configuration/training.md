# Training Configuration

Configure training hyperparameters.

## Basic Configuration

```yaml
training:
  batch_size: 128
  epochs: 200
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 128 | Samples per batch |
| `epochs` | int | **required** | Total training epochs |
| `test_frequency` | int | 10 | Test evaluation frequency |

## Batch Size

Number of samples per training batch.

```yaml
training:
  batch_size: 128
```

### Guidelines

| GPU Memory | Recommended Batch Size |
|------------|----------------------|
| 4 GB | 32-64 |
| 8 GB | 64-128 |
| 16 GB | 128-256 |
| 24+ GB | 256-512 |

### Large Batch Training

When increasing batch size, also adjust:

```yaml
training:
  batch_size: 256

optimizer:
  learning_rate: 0.2   # Scale LR with batch size

scheduler:
  warmup_epochs: 5     # Warmup helps stability
```

## Epochs

Total number of training epochs.

```yaml
training:
  epochs: 200
```

### Recommended Values

| Dataset | Model | Epochs |
|---------|-------|--------|
| CIFAR-10 | ResNet-20 | 200 |
| CIFAR-10 | ResNet-56 | 200-300 |
| CIFAR-100 | ResNet-50 | 200 |
| MNIST | Any | 50-100 |

## Test Frequency

When using validation split, how often to evaluate on test set.

```yaml
dataset:
  val_split: 0.1

training:
  batch_size: 128
  epochs: 200
  test_frequency: 10   # Test every 10 epochs
```

## Examples

### Quick Experiment

```yaml
training:
  batch_size: 128
  epochs: 10

checkpoint:
  enabled: false
```

### Standard Training

```yaml
training:
  batch_size: 128
  epochs: 200
```

### High Performance

```yaml
training:
  batch_size: 128
  epochs: 300

optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0005

scheduler:
  name: "cosine"
  T_max: 300
  warmup_epochs: 5

amp:
  enabled: true
```

### Memory Constrained

```yaml
training:
  batch_size: 32       # Smaller batch
  epochs: 200

amp:
  enabled: true        # Reduce memory with AMP
```

## See Also

- [Optimizer Configuration](optimizer.md)
- [Scheduler Configuration](scheduler.md)
