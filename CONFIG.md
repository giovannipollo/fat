# Configuration Reference

This document provides a complete reference for all available configuration parameters in the YAML config files.

## Table of Contents

- [Seed Configuration](#seed-configuration)
- [Dataset Configuration](#dataset-configuration)
- [Model Configuration](#model-configuration)
- [Quantization Configuration](#quantization-configuration)
- [Loss Configuration](#loss-configuration)
- [Training Configuration](#training-configuration)
- [Optimizer Configuration](#optimizer-configuration)
- [Scheduler Configuration](#scheduler-configuration)
- [AMP Configuration](#amp-configuration)
- [TensorBoard Configuration](#tensorboard-configuration)
- [Progress Configuration](#progress-configuration)
- [Checkpoint Configuration](#checkpoint-configuration)

---

## Seed Configuration

Controls reproducibility of training runs.

```yaml
seed:
  enabled: true          # Enable/disable seed setting
  value: 42              # Random seed value
  deterministic: false   # Use deterministic algorithms (slower but fully reproducible)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Whether to set random seeds for reproducibility |
| `value` | int | `42` | The seed value used for random, numpy, and torch |
| `deterministic` | bool | `false` | Enable CUDA deterministic mode (may reduce performance) |

---

## Dataset Configuration

Specifies the dataset and data loading parameters.

```yaml
dataset:
  name: "cifar10"        # Dataset name
  root: "./data"         # Download/storage directory
  download: true         # Download if not present
  num_workers: 4         # DataLoader worker processes
  val_split: 0.1         # Validation split ratio (optional)
  seed: 42               # Seed for reproducible splits (optional)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | **required** | Dataset identifier (see table below) |
| `root` | string | `"./data"` | Directory for dataset storage |
| `download` | bool | `true` | Automatically download dataset if missing |
| `num_workers` | int | `4` | Number of parallel data loading workers |
| `val_split` | float | `null` | Fraction of training data for validation (0.0-1.0) |
| `seed` | int | `42` | Random seed for validation split |

### Available Datasets

| Name | Classes | Image Size | Channels | Description |
|------|---------|------------|----------|-------------|
| `cifar10` | 10 | 32x32 | 3 (RGB) | CIFAR-10 image classification |
| `cifar100` | 100 | 32x32 | 3 (RGB) | CIFAR-100 fine-grained classification |
| `mnist` | 10 | 28x28 | 1 (Gray) | Handwritten digit recognition |
| `fashion_mnist` | 10 | 28x28 | 1 (Gray) | Fashion article classification |

---

## Model Configuration

Specifies the neural network architecture.

```yaml
model:
  name: "resnet20"       # Model architecture name
  num_classes: 10        # Number of output classes (optional, auto-detected)
  in_channels: 3         # Input channels (optional, auto-detected)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | **required** | Model architecture identifier (see tables below) |
| `num_classes` | int | auto | Number of output classes (detected from dataset) |
| `in_channels` | int | auto | Number of input channels (detected from dataset) |

### Standard Models

| Name | Parameters | Description |
|------|------------|-------------|
| `cnv` | ~1.5M | CNV (Compact Neural Vision) network |
| `mobilenetv1` | ~3.2M | MobileNetV1 with depthwise separable convolutions |
| `resnet20` | ~0.27M | ResNet-20 (CIFAR variant) |
| `resnet32` | ~0.46M | ResNet-32 (CIFAR variant) |
| `resnet44` | ~0.66M | ResNet-44 (CIFAR variant) |
| `resnet56` | ~0.85M | ResNet-56 (CIFAR variant) |
| `resnet110` | ~1.7M | ResNet-110 (CIFAR variant) |
| `resnet18` | ~11.2M | ResNet-18 (ImageNet variant, adapted) |
| `resnet34` | ~21.3M | ResNet-34 (ImageNet variant, adapted) |
| `resnet50` | ~23.5M | ResNet-50 with bottleneck blocks |
| `resnet101` | ~42.5M | ResNet-101 with bottleneck blocks |
| `resnet152` | ~58.1M | ResNet-152 with bottleneck blocks |
| `vgg11` | ~9.2M | VGG-11 |
| `vgg13` | ~9.4M | VGG-13 |
| `vgg16` | ~14.7M | VGG-16 |
| `vgg19` | ~20.0M | VGG-19 |

### Quantized Models (Brevitas)

All quantized models use the `quant_` prefix and require the `quantization` config section.

| Name | Base Model | Description |
|------|------------|-------------|
| `quant_cnv` | CNV | Quantized CNV |
| `quant_mobilenetv1` | MobileNetV1 | Quantized MobileNetV1 |
| `quant_resnet20` | ResNet-20 | Quantized ResNet-20 |
| `quant_resnet32` | ResNet-32 | Quantized ResNet-32 |
| `quant_resnet44` | ResNet-44 | Quantized ResNet-44 |
| `quant_resnet56` | ResNet-56 | Quantized ResNet-56 |
| `quant_resnet110` | ResNet-110 | Quantized ResNet-110 |
| `quant_resnet18` | ResNet-18 | Quantized ResNet-18 |
| `quant_resnet34` | ResNet-34 | Quantized ResNet-34 |
| `quant_resnet50` | ResNet-50 | Quantized ResNet-50 |
| `quant_resnet101` | ResNet-101 | Quantized ResNet-101 |
| `quant_resnet152` | ResNet-152 | Quantized ResNet-152 |
| `quant_vgg11` | VGG-11 | Quantized VGG-11 |
| `quant_vgg13` | VGG-13 | Quantized VGG-13 |
| `quant_vgg16` | VGG-16 | Quantized VGG-16 |
| `quant_vgg19` | VGG-19 | Quantized VGG-19 |

---

## Quantization Configuration

Settings for quantization-aware training (QAT) using Brevitas. Only used with `quant_*` models.

```yaml
quantization:
  weight_bit_width: 4    # Bit width for weights
  act_bit_width: 4       # Bit width for activations
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight_bit_width` | int | `8` | Bit width for weight quantization (1-8) |
| `act_bit_width` | int | `8` | Bit width for activation quantization (1-8) |

### Common Quantization Configurations

| Configuration | Weight Bits | Activation Bits | Use Case |
|--------------|-------------|-----------------|----------|
| INT8 | 8 | 8 | Standard quantization, minimal accuracy loss |
| INT4 | 4 | 4 | Aggressive compression, some accuracy loss |
| Binary | 1 | 1 | Maximum compression, significant accuracy loss |
| Mixed | 4 | 8 | Balance between compression and accuracy |

---

## Loss Configuration

Specifies the loss function for training.

```yaml
loss:
  name: "cross_entropy"  # Loss function name
  label_smoothing: 0.1   # Label smoothing factor (optional)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"cross_entropy"` | Loss function identifier (see table below) |

### Available Loss Functions

| Name | Description | Additional Parameters |
|------|-------------|----------------------|
| `cross_entropy` | Cross-entropy loss (default) | `label_smoothing`, `weight`, `ignore_index` |
| `nll` | Negative log likelihood loss | `weight`, `ignore_index` |
| `mse` | Mean squared error | - |
| `l1` | L1 / Mean absolute error | - |
| `smooth_l1` | Smooth L1 / Huber loss | `beta` |
| `bce` | Binary cross-entropy | - |
| `bce_with_logits` | BCE with logits | `pos_weight` |
| `kl_div` | KL divergence | `reduction`, `log_target` |
| `sqr_hinge` | Squared hinge loss | - |

### Loss-Specific Parameters

#### Cross-Entropy Loss
```yaml
loss:
  name: "cross_entropy"
  label_smoothing: 0.1   # Smoothing factor (0.0-1.0), default: 0.0
  weight: [1.0, 2.0]     # Class weights (optional)
  ignore_index: -100     # Index to ignore in loss computation (optional)
```

#### NLL Loss
```yaml
loss:
  name: "nll"
  weight: [1.0, 2.0]     # Class weights (optional)
  ignore_index: -100     # Index to ignore (optional)
```

#### Smooth L1 Loss
```yaml
loss:
  name: "smooth_l1"
  beta: 1.0              # Threshold for L1 vs L2 transition
```

#### KL Divergence Loss
```yaml
loss:
  name: "kl_div"
  reduction: "batchmean" # Reduction mode: "batchmean", "sum", "mean", "none"
  log_target: false      # Whether target is in log space
```

#### BCE with Logits Loss
```yaml
loss:
  name: "bce_with_logits"
  pos_weight: [1.0]      # Weight for positive class (optional)
```

#### Squared Hinge Loss
```yaml
loss:
  name: "sqr_hinge"
  # No additional parameters
  # Note: Expects targets as -1 or +1 (SVM-style)
```

---

## Training Configuration

Core training hyperparameters.

```yaml
training:
  batch_size: 128        # Batch size for training and evaluation
  epochs: 200            # Total number of training epochs
  test_frequency: 10     # Test evaluation frequency when using validation (optional)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | `128` | Number of samples per batch |
| `epochs` | int | **required** | Total training epochs |
| `test_frequency` | int | `10` | Epochs between test set evaluations (when using val_split) |

---

## Optimizer Configuration

Specifies the optimization algorithm and its hyperparameters.

```yaml
optimizer:
  name: "sgd"            # Optimizer name
  learning_rate: 0.1     # Initial learning rate
  momentum: 0.9          # Momentum factor (SGD only)
  weight_decay: 0.0005   # L2 regularization
  nesterov: false        # Nesterov momentum (SGD only)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | **required** | Optimizer identifier |
| `learning_rate` | float | **required** | Initial learning rate |
| `weight_decay` | float | `0.0` | L2 regularization coefficient |

### Available Optimizers

#### SGD (Stochastic Gradient Descent)
```yaml
optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9          # Momentum factor (default: 0.9)
  weight_decay: 0.0005
  nesterov: false        # Use Nesterov momentum (default: false)
```

#### Adam
```yaml
optimizer:
  name: "adam"
  learning_rate: 0.001
  weight_decay: 0.0
  betas: [0.9, 0.999]    # Coefficients for running averages
  eps: 1e-8              # Numerical stability term
```

#### AdamW (Adam with decoupled weight decay)
```yaml
optimizer:
  name: "adamw"
  learning_rate: 0.001
  weight_decay: 0.01     # Decoupled weight decay
  betas: [0.9, 0.999]
  eps: 1e-8
```

---

## Scheduler Configuration

Learning rate scheduling with optional warmup.

```yaml
scheduler:
  name: "cosine"         # Scheduler type
  warmup_epochs: 5       # Linear warmup epochs (optional)
  T_max: 200             # Scheduler-specific parameter
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"cosine"` | Scheduler identifier |
| `warmup_epochs` | int | `0` | Number of linear warmup epochs |

### Available Schedulers

#### Cosine Annealing
```yaml
scheduler:
  name: "cosine"
  T_max: 200             # Maximum iterations (usually = epochs - warmup_epochs)
  eta_min: 0             # Minimum learning rate (default: 0)
  warmup_epochs: 5       # Optional warmup
```

#### Step LR
```yaml
scheduler:
  name: "step"
  step_size: 30          # Epochs between LR decay
  gamma: 0.1             # Multiplicative factor (default: 0.1)
  warmup_epochs: 5       # Optional warmup
```

#### MultiStep LR
```yaml
scheduler:
  name: "multistep"
  milestones: [60, 120, 160]  # Epochs to decay LR
  gamma: 0.2             # Multiplicative factor (default: 0.1)
  warmup_epochs: 5       # Optional warmup
```

#### Exponential LR
```yaml
scheduler:
  name: "exponential"
  gamma: 0.95            # Multiplicative factor per epoch (default: 0.95)
  warmup_epochs: 5       # Optional warmup
```

#### Reduce on Plateau
```yaml
scheduler:
  name: "plateau"
  mode: "max"            # "min" or "max" (default: "max")
  factor: 0.1            # Factor to reduce LR (default: 0.1)
  patience: 10           # Epochs without improvement before reducing (default: 10)
  min_lr: 1e-6           # Minimum learning rate (default: 1e-6)
  warmup_epochs: 5       # Optional warmup
```

#### No Scheduler
```yaml
scheduler:
  name: "none"           # Constant learning rate
```

---

## AMP Configuration

Automatic Mixed Precision for faster training on CUDA GPUs.

```yaml
amp:
  enabled: true          # Enable/disable AMP
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable mixed precision training (CUDA only) |

> **Note:** AMP is automatically disabled for quantized models and non-CUDA devices.

---

## TensorBoard Configuration

TensorBoard logging settings.

```yaml
tensorboard:
  enabled: true          # Enable/disable TensorBoard logging
  log_dir: "./runs"      # Log directory (optional, uses experiment dir if checkpoints enabled)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable TensorBoard logging |
| `log_dir` | string | `"./runs"` | Directory for TensorBoard event files |

---

## Progress Configuration

Progress bar settings.

```yaml
progress:
  enabled: true          # Enable/disable progress bars
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Show tqdm progress bars during training |

---

## Checkpoint Configuration

Model checkpointing and experiment management.

```yaml
checkpoint:
  enabled: true          # Enable/disable checkpointing
  dir: "./experiments"   # Base directory for experiments
  experiment_name: ""    # Custom experiment name (optional)
  save_frequency: 10     # Save every N epochs
  save_best: true        # Save best model by validation/test accuracy
  resume: ""             # Path to checkpoint to resume from (optional)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable checkpoint saving |
| `dir` | string | `"./experiments"` | Base directory for experiment folders |
| `experiment_name` | string | auto | Custom name (default: `{model}_{dataset}_{timestamp}`) |
| `save_frequency` | int | `10` | Save checkpoint every N epochs |
| `save_best` | bool | `true` | Save best model checkpoint |
| `resume` | string | `null` | Path to checkpoint file to resume training |

### Checkpoint Contents

Each checkpoint (`.pt` file) contains:
- `model_state_dict` - Model weights
- `optimizer_state_dict` - Optimizer state
- `scheduler_state_dict` - Scheduler state
- `scaler_state_dict` - AMP scaler state (if using AMP)
- `epoch` - Current epoch number
- `best_acc` - Best accuracy achieved
- `config` - Training configuration

---

## Complete Example

```yaml
# Complete configuration example

seed:
  enabled: true
  value: 42
  deterministic: false

dataset:
  name: "cifar10"
  root: "./data"
  download: true
  num_workers: 8
  val_split: 0.1

model:
  name: "resnet20"

loss:
  name: "cross_entropy"
  label_smoothing: 0.1

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
  enabled: true

tensorboard:
  enabled: true

progress:
  enabled: true

checkpoint:
  enabled: true
  dir: "./experiments"
  save_frequency: 20
  save_best: true
```

## Quantized Model Example

```yaml
# Quantized CNV with 4-bit weights and activations

seed:
  enabled: true
  value: 42

dataset:
  name: "cifar10"
  root: "./data"
  download: true
  num_workers: 4

model:
  name: "quant_cnv"

quantization:
  weight_bit_width: 4
  act_bit_width: 4

loss:
  name: "sqr_hinge"  # Common for quantized networks

training:
  batch_size: 128
  epochs: 200

optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0001

scheduler:
  name: "cosine"
  T_max: 200
  warmup_epochs: 5

amp:
  enabled: false  # Disabled for quantized training

checkpoint:
  enabled: true
  dir: "./experiments"
  save_best: true
```
