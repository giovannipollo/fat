# Model Configuration

Configure the neural network architecture.

## Basic Configuration

```yaml
model:
  name: "resnet20"
```

## Full Configuration

```yaml
model:
  name: "resnet20"       # Model architecture (required)
  num_classes: 10        # Output classes (auto-detected)
  in_channels: 3         # Input channels (auto-detected)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | **required** | Model architecture identifier |
| `num_classes` | int | auto | Number of output classes |
| `in_channels` | int | auto | Number of input channels |

## Standard Models

### Compact Networks

```yaml
model:
  name: "cnv"          # ~1.5M params
```

```yaml
model:
  name: "mobilenetv1"  # ~3.2M params
```

### ResNet (CIFAR)

```yaml
model:
  name: "resnet20"     # ~0.27M params
```

```yaml
model:
  name: "resnet56"     # ~0.85M params
```

```yaml
model:
  name: "resnet110"    # ~1.7M params
```

### ResNet (ImageNet)

```yaml
model:
  name: "resnet18"     # ~11.2M params
```

```yaml
model:
  name: "resnet50"     # ~23.5M params
```

### VGG

```yaml
model:
  name: "vgg16"        # ~14.7M params
```

## Quantized Models

Use the `quant_` prefix for quantized versions:

```yaml
model:
  name: "quant_resnet20"

quantization:
  weight_bit_width: 4
  act_bit_width: 4
```

### Available Quantized Models

- `quant_cnv`
- `quant_mobilenetv1`
- `quant_resnet20`, `quant_resnet32`, `quant_resnet44`, `quant_resnet56`, `quant_resnet110`
- `quant_resnet18`, `quant_resnet34`, `quant_resnet50`, `quant_resnet101`, `quant_resnet152`
- `quant_vgg11`, `quant_vgg13`, `quant_vgg16`, `quant_vgg19`

## Quantization Configuration

For `quant_*` models:

```yaml
quantization:
  weight_bit_width: 4   # 1-8 bits
  act_bit_width: 4      # 1-8 bits
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight_bit_width` | int | 8 | Bit width for weights |
| `act_bit_width` | int | 8 | Bit width for activations |

## Auto-Detection

Model parameters are automatically detected from the dataset:

```yaml
dataset:
  name: "cifar100"      # 100 classes, 3 channels

model:
  name: "resnet50"      # Automatically uses 100 classes, 3 channels
```

## Examples

### CIFAR-10 Classification

```yaml
dataset:
  name: "cifar10"

model:
  name: "resnet56"
```

### MNIST Classification

```yaml
dataset:
  name: "mnist"         # 10 classes, 1 channel

model:
  name: "resnet18"      # Adapts to 1-channel input
```

### Quantized Training

```yaml
model:
  name: "quant_cnv"

quantization:
  weight_bit_width: 2
  act_bit_width: 2

loss:
  name: "sqr_hinge"

amp:
  enabled: false
```

## See Also

- [Models Overview](../models/index.md)
- [Standard Models](../models/standard.md)
- [Quantized Models](../models/quantized.md)
