# Optimizer Configuration

Configure the optimization algorithm.

## Basic Configuration

```yaml
optimizer:
  name: "sgd"
  learning_rate: 0.1
```

## Available Optimizers

| Name | Description | Best For |
|------|-------------|----------|
| `sgd` | Stochastic Gradient Descent | CNNs, standard training |
| `adam` | Adaptive Moment Estimation | Quick convergence |
| `adamw` | Adam with decoupled weight decay | Transformers, regularization |

## SGD

Most common optimizer for CNN training.

```yaml
optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0005
  nesterov: false
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | **required** | Initial learning rate |
| `momentum` | float | 0.9 | Momentum factor |
| `weight_decay` | float | 0.0 | L2 regularization |
| `nesterov` | bool | false | Use Nesterov momentum |

### Recommended Settings

**CIFAR-10/100:**
```yaml
optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0005
```

**Fine-tuning:**
```yaml
optimizer:
  name: "sgd"
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
```

## Adam

Adaptive learning rate optimizer.

```yaml
optimizer:
  name: "adam"
  learning_rate: 0.001
  weight_decay: 0.0
  betas: [0.9, 0.999]
  eps: 1e-8
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | **required** | Initial learning rate |
| `weight_decay` | float | 0.0 | L2 regularization |
| `betas` | list | [0.9, 0.999] | Coefficients for running averages |
| `eps` | float | 1e-8 | Numerical stability |

### Recommended Settings

```yaml
optimizer:
  name: "adam"
  learning_rate: 0.001
  weight_decay: 0.0
```

## AdamW

Adam with decoupled weight decay (better regularization).

```yaml
optimizer:
  name: "adamw"
  learning_rate: 0.001
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | **required** | Initial learning rate |
| `weight_decay` | float | 0.01 | Decoupled weight decay |
| `betas` | list | [0.9, 0.999] | Coefficients for running averages |
| `eps` | float | 1e-8 | Numerical stability |

### Recommended Settings

```yaml
optimizer:
  name: "adamw"
  learning_rate: 0.001
  weight_decay: 0.01
```

## Examples

### High Accuracy Training

```yaml
optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0005
  nesterov: true

scheduler:
  name: "cosine"
  T_max: 200
  warmup_epochs: 5
```

### Quick Experiment

```yaml
optimizer:
  name: "adam"
  learning_rate: 0.001

scheduler:
  name: "none"
```

### Quantized Training

```yaml
optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0001  # Lower regularization
```

### Large Batch Training

```yaml
training:
  batch_size: 256

optimizer:
  name: "sgd"
  learning_rate: 0.2    # Scale with batch size
  momentum: 0.9
  weight_decay: 0.0005

scheduler:
  warmup_epochs: 5      # Warmup helps with large batches
```

## Learning Rate Guidelines

| Optimizer | Typical Range | Notes |
|-----------|---------------|-------|
| SGD | 0.01 - 0.1 | Higher with warmup |
| Adam | 0.0001 - 0.001 | Lower than SGD |
| AdamW | 0.0001 - 0.001 | Same as Adam |

## See Also

- [Scheduler Configuration](scheduler.md)
- [Training Configuration](training.md)
