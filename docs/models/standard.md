# Standard Models

Detailed information about all standard (non-quantized) model architectures.

## CNV (Compact Neural Vision)

A compact convolutional network designed for efficient inference on edge devices and FPGAs.

```yaml
model:
  name: "cnv"
```

### Architecture

- 6 convolutional layers with batch normalization
- 2 max pooling layers
- 2 fully connected layers
- ~1.5M parameters

### Structure

```
Conv(3, 64) → BN → ReLU
Conv(64, 64) → BN → ReLU → MaxPool
Conv(64, 128) → BN → ReLU
Conv(128, 128) → BN → ReLU → MaxPool
Conv(128, 256) → BN → ReLU
Conv(256, 256) → BN → ReLU
FC(256, 512) → BN → ReLU
FC(512, 512) → BN → ReLU
FC(512, num_classes)
```

---

## MobileNetV1

Lightweight architecture using depthwise separable convolutions.

```yaml
model:
  name: "mobilenetv1"
```

### Architecture

- Depthwise separable convolutions
- Width multiplier: 1.0
- ~3.2M parameters

### Key Features

- Efficient for mobile deployment
- Good accuracy/speed trade-off
- Low memory footprint

---

## ResNet (CIFAR Variants)

ResNet architectures specifically designed for CIFAR-sized (32x32) images.

### Available Variants

| Model | Layers | Parameters | Config |
|-------|--------|------------|--------|
| ResNet-20 | 20 | ~0.27M | `resnet20` |
| ResNet-32 | 32 | ~0.46M | `resnet32` |
| ResNet-44 | 44 | ~0.66M | `resnet44` |
| ResNet-56 | 56 | ~0.85M | `resnet56` |
| ResNet-110 | 110 | ~1.7M | `resnet110` |

### Architecture

Uses BasicBlock (two 3x3 convolutions) with skip connections:

```
Input → Conv(3x3) → Layer1 → Layer2 → Layer3 → AvgPool → FC
```

Each layer has progressively more channels: 16 → 32 → 64

### Example

```yaml
model:
  name: "resnet56"  # Good balance of accuracy/speed
```

---

## ResNet (ImageNet Variants)

Standard ResNet architectures adapted for small images.

### Available Variants

| Model | Layers | Parameters | Block Type | Config |
|-------|--------|------------|------------|--------|
| ResNet-18 | 18 | ~11.2M | BasicBlock | `resnet18` |
| ResNet-34 | 34 | ~21.3M | BasicBlock | `resnet34` |
| ResNet-50 | 50 | ~23.5M | Bottleneck | `resnet50` |
| ResNet-101 | 101 | ~42.5M | Bottleneck | `resnet101` |
| ResNet-152 | 152 | ~58.1M | Bottleneck | `resnet152` |

### Architecture

- **BasicBlock**: Two 3x3 convolutions (ResNet-18, 34)
- **Bottleneck**: 1x1 → 3x3 → 1x1 convolutions (ResNet-50+)

### Adaptations for Small Images

- Smaller initial convolution (3x3 instead of 7x7)
- No initial max pooling
- Adjusted stride patterns

### Example

```yaml
model:
  name: "resnet50"  # Bottleneck architecture
```

---

## VGG

Classic VGG architectures adapted for CIFAR-sized images.

### Available Variants

| Model | Layers | Parameters | Config |
|-------|--------|------------|--------|
| VGG-11 | 11 | ~9.2M | `vgg11` |
| VGG-13 | 13 | ~9.4M | `vgg13` |
| VGG-16 | 16 | ~14.7M | `vgg16` |
| VGG-19 | 19 | ~20.0M | `vgg19` |

### Architecture

Stacked 3x3 convolutions with max pooling:

```
Conv layers → MaxPool → Conv layers → MaxPool → ... → FC → FC → Output
```

### Batch Normalization

All VGG variants include batch normalization after each convolution for:

- Faster training
- Better regularization
- Improved accuracy

### Example

```yaml
model:
  name: "vgg16"
```

---

## Feature Comparison

| Feature | CNV | MobileNet | ResNet-CIFAR | ResNet-ImageNet | VGG |
|---------|-----|-----------|--------------|-----------------|-----|
| Skip connections | ✗ | ✗ | ✓ | ✓ | ✗ |
| Batch normalization | ✓ | ✓ | ✓ | ✓ | ✓ |
| Depthwise separable | ✗ | ✓ | ✗ | ✗ | ✗ |
| Bottleneck blocks | ✗ | ✗ | ✗ | ✓ (50+) | ✗ |
| Global avg pooling | ✗ | ✓ | ✓ | ✓ | ✗ |

## Next Steps

- [Quantized Models](quantized.md) - Low-precision versions
- [Model Configuration](../configuration/model.md) - Configuration options
