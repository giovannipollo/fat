# Scheduler Configuration

Configure learning rate scheduling with optional warmup.

## Basic Configuration

```yaml
scheduler:
  name: "cosine"
  T_max: 200
```

## Available Schedulers

| Name | Description | Best For |
|------|-------------|----------|
| `cosine` | Cosine annealing | General training |
| `step` | Step decay | Simple schedules |
| `multistep` | Multi-step decay | Custom milestones |
| `exponential` | Exponential decay | Smooth decay |
| `plateau` | Reduce on plateau | Adaptive |
| `none` | No scheduling | Quick tests |

## Cosine Annealing

Smooth cosine decay from initial LR to minimum.

```yaml
scheduler:
  name: "cosine"
  T_max: 200           # Total epochs
  eta_min: 0           # Minimum LR
  warmup_epochs: 5     # Optional warmup
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `T_max` | int | epochs | Cycle length (usually = epochs) |
| `eta_min` | float | 0 | Minimum learning rate |
| `warmup_epochs` | int | 0 | Warmup epochs |

### Visualization

```
LR
│  ╭──╮
│ ╱    ╲
│╱      ╲
│        ╲
└──────────────→ Epoch
  warmup  cosine decay
```

## Step LR

Decay learning rate by factor every N epochs.

```yaml
scheduler:
  name: "step"
  step_size: 30        # Decay every 30 epochs
  gamma: 0.1           # Multiply LR by 0.1
  warmup_epochs: 5
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_size` | int | 30 | Epochs between decay |
| `gamma` | float | 0.1 | Decay factor |
| `warmup_epochs` | int | 0 | Warmup epochs |

## MultiStep LR

Decay at specific epochs.

```yaml
scheduler:
  name: "multistep"
  milestones: [60, 120, 160]
  gamma: 0.2
  warmup_epochs: 5
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `milestones` | list | [30, 60, 90] | Epochs to decay |
| `gamma` | float | 0.1 | Decay factor |
| `warmup_epochs` | int | 0 | Warmup epochs |

## Exponential LR

Decay by factor every epoch.

```yaml
scheduler:
  name: "exponential"
  gamma: 0.95          # LR = LR * 0.95 each epoch
  warmup_epochs: 5
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | float | 0.95 | Per-epoch decay factor |
| `warmup_epochs` | int | 0 | Warmup epochs |

## Reduce on Plateau

Reduce LR when metric stops improving.

```yaml
scheduler:
  name: "plateau"
  mode: "max"          # "max" for accuracy, "min" for loss
  factor: 0.1          # Reduce by 10x
  patience: 10         # Wait 10 epochs
  min_lr: 1e-6
  warmup_epochs: 5
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | "max" | "max" or "min" |
| `factor` | float | 0.1 | Reduction factor |
| `patience` | int | 10 | Epochs without improvement |
| `min_lr` | float | 1e-6 | Minimum learning rate |
| `warmup_epochs` | int | 0 | Warmup epochs |

## No Scheduler

Constant learning rate.

```yaml
scheduler:
  name: "none"
```

## Warmup

All schedulers support linear warmup:

```yaml
scheduler:
  name: "cosine"
  T_max: 200
  warmup_epochs: 5     # 5 epochs of linear warmup
```

### Warmup Behavior

- Epoch 1: LR = base_lr × 0.2
- Epoch 2: LR = base_lr × 0.4
- Epoch 3: LR = base_lr × 0.6
- Epoch 4: LR = base_lr × 0.8
- Epoch 5: LR = base_lr × 1.0
- Epoch 6+: Normal scheduler

## Examples

### Standard CIFAR Training

```yaml
optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0005

scheduler:
  name: "cosine"
  T_max: 200
  warmup_epochs: 5
```

### Step Decay Schedule

```yaml
optimizer:
  name: "sgd"
  learning_rate: 0.1

scheduler:
  name: "multistep"
  milestones: [100, 150]
  gamma: 0.1
```

### Adaptive Schedule

```yaml
optimizer:
  name: "adam"
  learning_rate: 0.001

scheduler:
  name: "plateau"
  mode: "max"
  factor: 0.5
  patience: 10
```

### Quick Test

```yaml
optimizer:
  name: "adam"
  learning_rate: 0.001

scheduler:
  name: "none"
```

## See Also

- [Optimizer Configuration](optimizer.md)
- [Training Configuration](training.md)
