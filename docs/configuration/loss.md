# Loss Configuration

Configure the loss function for training.

## Basic Configuration

```yaml
loss:
  name: "cross_entropy"
```

## Default Behavior

If no loss configuration is provided, `cross_entropy` is used by default.

## Available Loss Functions

| Name | Description | Use Case |
|------|-------------|----------|
| `cross_entropy` | Cross-entropy loss | Standard classification (default) |
| `nll` | Negative log likelihood | With log-softmax output |
| `mse` | Mean squared error | Regression tasks |
| `l1` | L1 / MAE loss | Regression tasks |
| `smooth_l1` | Smooth L1 / Huber | Robust regression |
| `bce` | Binary cross-entropy | Binary classification |
| `bce_with_logits` | BCE with logits | Binary (no sigmoid) |
| `kl_div` | KL divergence | Distribution matching |
| `sqr_hinge` | Squared hinge loss | Quantized networks |

## Cross-Entropy Loss

Standard loss for multi-class classification.

```yaml
loss:
  name: "cross_entropy"
```

### With Label Smoothing

Regularization technique that softens targets:

```yaml
loss:
  name: "cross_entropy"
  label_smoothing: 0.1   # 0.0 to 1.0
```

### With Class Weights

Handle imbalanced datasets:

```yaml
loss:
  name: "cross_entropy"
  weight: [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

### Ignore Index

Ignore specific class during loss computation:

```yaml
loss:
  name: "cross_entropy"
  ignore_index: -100
```

### All Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label_smoothing` | float | 0.0 | Smoothing factor (0.0-1.0) |
| `weight` | list | null | Per-class weights |
| `ignore_index` | int | null | Class index to ignore |

## Squared Hinge Loss

SVM-style loss, commonly used for quantized networks.

```yaml
loss:
  name: "sqr_hinge"
```

**Formula:** $L = \text{mean}(\max(0, 1 - y \cdot f(x))^2)$

**Note:** Automatically converts class indices to ±1 encoding.

### Recommended Configuration

```yaml
model:
  name: "quant_cnv"

loss:
  name: "sqr_hinge"

amp:
  enabled: false
```

## NLL Loss

Negative log likelihood, expects log-probabilities.

```yaml
loss:
  name: "nll"
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight` | list | null | Per-class weights |
| `ignore_index` | int | null | Class index to ignore |

## Smooth L1 Loss

Huber loss, less sensitive to outliers than MSE.

```yaml
loss:
  name: "smooth_l1"
  beta: 1.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | 1.0 | Threshold for L1/L2 transition |

## KL Divergence Loss

For matching probability distributions.

```yaml
loss:
  name: "kl_div"
  reduction: "batchmean"
  log_target: false
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reduction` | string | "batchmean" | Reduction method |
| `log_target` | bool | false | Target in log space |

## BCE with Logits

Binary cross-entropy with built-in sigmoid.

```yaml
loss:
  name: "bce_with_logits"
```

### With Positive Weight

Handle imbalanced binary classification:

```yaml
loss:
  name: "bce_with_logits"
  pos_weight: [2.0]  # Weight for positive class
```

## Examples

### Standard Training

```yaml
loss:
  name: "cross_entropy"
```

### Regularized Training

```yaml
loss:
  name: "cross_entropy"
  label_smoothing: 0.1
```

### Quantized Network

```yaml
model:
  name: "quant_resnet20"

loss:
  name: "sqr_hinge"
```

### Imbalanced Dataset

```yaml
loss:
  name: "cross_entropy"
  weight: [1.0, 5.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 3.0, 1.0]
```

## See Also

- [Configuration Overview](index.md)
- [Quantized Models](../models/quantized.md)
