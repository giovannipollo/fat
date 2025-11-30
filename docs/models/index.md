# Models

The framework provides a variety of neural network architectures for image classification, including both standard and quantized versions.

## Available Architectures

### Compact Networks

| Model | Name | Parameters | Best For |
|-------|------|------------|----------|
| CNV | `cnv` | ~1.5M | Edge deployment, FPGA |
| MobileNetV1 | `mobilenetv1` | ~3.2M | Mobile devices, efficiency |

### ResNet Family

Two variants are available:

- **CIFAR variants** (20, 32, 44, 56, 110): Optimized for 32x32 images
- **ImageNet variants** (18, 34, 50, 101, 152): Adapted for small images

| Model | Name | Parameters | Description |
|-------|------|------------|-------------|
| ResNet-20 | `resnet20` | ~0.27M | CIFAR variant, lightweight |
| ResNet-32 | `resnet32` | ~0.46M | CIFAR variant |
| ResNet-44 | `resnet44` | ~0.66M | CIFAR variant |
| ResNet-56 | `resnet56` | ~0.85M | CIFAR variant, good accuracy |
| ResNet-110 | `resnet110` | ~1.7M | CIFAR variant, deep |
| ResNet-18 | `resnet18` | ~11.2M | ImageNet variant, balanced |
| ResNet-34 | `resnet34` | ~21.3M | ImageNet variant |
| ResNet-50 | `resnet50` | ~23.5M | ImageNet variant, bottleneck |
| ResNet-101 | `resnet101` | ~42.5M | ImageNet variant, very deep |
| ResNet-152 | `resnet152` | ~58.1M | ImageNet variant, deepest |

### VGG Family

| Model | Name | Parameters | Description |
|-------|------|------------|-------------|
| VGG-11 | `vgg11` | ~9.2M | Classic VGG |
| VGG-13 | `vgg13` | ~9.4M | Deeper VGG |
| VGG-16 | `vgg16` | ~14.7M | Popular variant |
| VGG-19 | `vgg19` | ~20.0M | Deepest VGG |

## Quantized Models

All standard models have quantized versions using [Brevitas](https://github.com/Xilinx/brevitas). Use the `quant_` prefix:

| Standard | Quantized |
|----------|-----------|
| `cnv` | `quant_cnv` |
| `mobilenetv1` | `quant_mobilenetv1` |
| `resnet20` | `quant_resnet20` |
| `resnet56` | `quant_resnet56` |
| `vgg16` | `quant_vgg16` |
| ... | `quant_*` |

See [Quantized Models](quantized.md) for details.

## Usage

### Configuration

```yaml title="config.yaml"
model:
  name: "resnet20"  # Model name from tables above
```

Model parameters (`num_classes`, `in_channels`) are automatically detected from the dataset.

### Manual Override

```yaml
model:
  name: "resnet20"
  num_classes: 100    # Override auto-detection
  in_channels: 1      # For grayscale images
```

## Model Selection Guide

| Use Case | Recommended Model |
|----------|-------------------|
| Quick experiments | `mobilenetv1`, `resnet20` |
| Best CIFAR-10 accuracy | `resnet56`, `resnet110` |
| Best CIFAR-100 accuracy | `resnet50`, `resnet101` |
| Edge/FPGA deployment | `quant_cnv`, `quant_mobilenetv1` |
| Memory constrained | `resnet20`, `cnv` |
| Speed priority | `mobilenetv1` |

## Next Steps

- [Standard Models](standard.md) - Detailed architecture information
- [Quantized Models](quantized.md) - Quantization-aware training
