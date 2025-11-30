# Quantized Models

Quantization-aware training (QAT) using [Brevitas](https://github.com/Xilinx/brevitas) for deployment on edge devices, FPGAs, and specialized hardware.

## Overview

Quantized models use low-precision arithmetic (1-8 bits) instead of 32-bit floating point, providing:

- **Smaller model size** (4-32x reduction)
- **Faster inference** on supported hardware
- **Lower power consumption**
- **FPGA/ASIC deployment** compatibility

## Available Models

All standard models have quantized versions with the `quant_` prefix:

| Standard | Quantized | Description |
|----------|-----------|-------------|
| `cnv` | `quant_cnv` | Quantized CNV |
| `mobilenetv1` | `quant_mobilenetv1` | Quantized MobileNet |
| `resnet20` | `quant_resnet20` | Quantized ResNet-20 |
| `resnet32` | `quant_resnet32` | Quantized ResNet-32 |
| `resnet44` | `quant_resnet44` | Quantized ResNet-44 |
| `resnet56` | `quant_resnet56` | Quantized ResNet-56 |
| `resnet110` | `quant_resnet110` | Quantized ResNet-110 |
| `resnet18` | `quant_resnet18` | Quantized ResNet-18 |
| `resnet34` | `quant_resnet34` | Quantized ResNet-34 |
| `resnet50` | `quant_resnet50` | Quantized ResNet-50 |
| `resnet101` | `quant_resnet101` | Quantized ResNet-101 |
| `resnet152` | `quant_resnet152` | Quantized ResNet-152 |
| `vgg11` | `quant_vgg11` | Quantized VGG-11 |
| `vgg13` | `quant_vgg13` | Quantized VGG-13 |
| `vgg16` | `quant_vgg16` | Quantized VGG-16 |
| `vgg19` | `quant_vgg19` | Quantized VGG-19 |

## Configuration

### Basic Usage

```yaml title="config.yaml"
model:
  name: "quant_resnet20"

quantization:
  weight_bit_width: 4
  act_bit_width: 4
```

### Quantization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight_bit_width` | int | 8 | Bit width for weights (1-8) |
| `act_bit_width` | int | 8 | Bit width for activations (1-8) |

## Common Configurations

### INT8 Quantization

Standard 8-bit quantization with minimal accuracy loss:

```yaml
model:
  name: "quant_resnet20"

quantization:
  weight_bit_width: 8
  act_bit_width: 8
```

### INT4 Quantization

Aggressive 4-bit quantization for maximum compression:

```yaml
model:
  name: "quant_resnet20"

quantization:
  weight_bit_width: 4
  act_bit_width: 4
```

### Mixed Precision

Different bit widths for weights and activations:

```yaml
model:
  name: "quant_resnet20"

quantization:
  weight_bit_width: 4
  act_bit_width: 8
```

### Binary/Ternary Networks

Extreme quantization (1-2 bits):

```yaml
model:
  name: "quant_cnv"

quantization:
  weight_bit_width: 2
  act_bit_width: 2
```

## Training Recommendations

### Loss Function

Squared hinge loss often works better for quantized networks:

```yaml
loss:
  name: "sqr_hinge"
```

### Disable AMP

Mixed precision training (AMP) should be disabled for QAT:

```yaml
amp:
  enabled: false
```

### Learning Rate

Slightly lower learning rates may help:

```yaml
optimizer:
  name: "sgd"
  learning_rate: 0.1  # or 0.01 for fine-tuning
  momentum: 0.9
  weight_decay: 0.0001
```

## Complete Example

```yaml title="configs/quant_cnv_4bit.yaml"
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
  name: "sqr_hinge"

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
  enabled: false

checkpoint:
  enabled: true
  dir: "./experiments"
  save_best: true
```

## Expected Accuracy

Approximate accuracy on CIFAR-10 for different bit widths:

| Model | FP32 | INT8 | INT4 | INT2 |
|-------|------|------|------|------|
| CNV | ~88% | ~87% | ~85% | ~80% |
| ResNet-20 | ~92% | ~91% | ~89% | ~85% |
| ResNet-56 | ~93% | ~92% | ~90% | ~86% |

!!! note
    Actual accuracy depends on training configuration and random seed.

## Export for Deployment

After training, quantized models can be exported using Brevitas export functions:

```python
from brevitas.export import export_qonnx

# Load trained model
model = ...

# Export to QONNX format
export_qonnx(model, input_shape=(1, 3, 32, 32), export_path="model.onnx")
```

## Next Steps

- [Loss Configuration](../configuration/loss.md) - Configure loss functions
- [Training Configuration](../configuration/training.md) - Training settings
- [Standard Models](standard.md) - Non-quantized versions
