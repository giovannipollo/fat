# Checkpoint Configuration

Configure model saving and experiment management.

## Basic Configuration

```yaml
checkpoint:
  enabled: true
  dir: "./experiments"
  save_best: true
```

## Full Configuration

```yaml
checkpoint:
  enabled: true              # Enable checkpointing
  dir: "./experiments"       # Base directory
  experiment_name: ""        # Custom name (optional)
  save_frequency: 10         # Save every N epochs
  save_best: true            # Save best model
  resume: ""                 # Resume path (optional)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable checkpoint saving |
| `dir` | string | "./experiments" | Base directory |
| `experiment_name` | string | auto | Custom experiment name |
| `save_frequency` | int | 10 | Epochs between saves |
| `save_best` | bool | true | Save best accuracy model |
| `resume` | string | null | Path to resume from |

## Experiment Directory Structure

When checkpoints are enabled:

```
experiments/
└── resnet20_cifar10_20240115_143022/
    ├── config.yaml           # Saved configuration
    ├── checkpoints/
    │   ├── epoch_0010.pt     # Periodic checkpoint
    │   ├── epoch_0020.pt
    │   ├── latest.pt         # Most recent
    │   └── best.pt           # Best accuracy
    └── tensorboard/          # TensorBoard logs
        └── events.out.tfevents.*
```

## Experiment Naming

### Automatic Naming

Default: `{model}_{dataset}_{timestamp}`

```yaml
checkpoint:
  enabled: true
  dir: "./experiments"
  # experiment_name: not set → auto-generated
```

### Custom Naming

```yaml
checkpoint:
  enabled: true
  dir: "./experiments"
  experiment_name: "my_experiment_v1"
```

## Checkpoint Contents

Each `.pt` file contains:

| Key | Description |
|-----|-------------|
| `model_state_dict` | Model weights |
| `optimizer_state_dict` | Optimizer state |
| `scheduler_state_dict` | Scheduler state |
| `scaler_state_dict` | AMP scaler (if enabled) |
| `epoch` | Current epoch number |
| `best_acc` | Best accuracy achieved |
| `config` | Training configuration |

## Resume Training

Resume from a checkpoint:

```yaml
checkpoint:
  enabled: true
  dir: "./experiments"
  resume: "./experiments/resnet20_cifar10_20240115_143022/checkpoints/latest.pt"
```

### Resume Behavior

1. Loads model weights
2. Restores optimizer state
3. Restores scheduler state
4. Restores AMP scaler (if applicable)
5. Continues from next epoch
6. Preserves best accuracy

## Examples

### Minimal Checkpointing

```yaml
checkpoint:
  enabled: true
  dir: "./experiments"
```

### Frequent Saves

```yaml
checkpoint:
  enabled: true
  dir: "./experiments"
  save_frequency: 5    # Every 5 epochs
  save_best: true
```

### Resume Experiment

```yaml
checkpoint:
  enabled: true
  dir: "./experiments"
  save_frequency: 10
  save_best: true
  resume: "./experiments/my_experiment/checkpoints/latest.pt"
```

### Disable Checkpointing

```yaml
checkpoint:
  enabled: false
```

## Loading Checkpoints Manually

```python
import torch
from models import get_model
from utils import load_config

# Load config and create model
config = load_config("experiments/my_exp/config.yaml")
model = get_model(config)

# Load checkpoint
checkpoint = torch.load("experiments/my_exp/checkpoints/best.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Use model for inference
model.eval()
```

## See Also

- [Configuration Overview](index.md)
- [Training Configuration](training.md)
