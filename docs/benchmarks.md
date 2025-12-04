# Benchmark Results

Benchmark results for models trained on various datasets. Results show **mean accuracy ± standard deviation** across multiple runs.

## CIFAR-10

### Standard Models

| Model | Accuracy (%) | Epochs | Notes |
|-------|--------------|--------|-------|
| ResNet-20 | - | 200 | |
| ResNet-32 | - | 200 | |
| ResNet-44 | - | 200 | |
| ResNet-56 | - | 200 | |
| ResNet-110 | - | 200 | |
| ResNet-18 | - | 200 | |
| ResNet-34 | - | 200 | |
| ResNet-50 | - | 200 | |
| VGG-11 | - | 200 | |
| VGG-13 | - | 200 | |
| VGG-16 | - | 200 | |
| VGG-19 | - | 200 | |
| MobileNetV1 | - | 200 | |
| CNV | - | 200 | |

### Quantized Models

| Model | Bits (W/A) | Accuracy (%) | Epochs | Notes |
|-------|------------|--------------|--------|-------|
| QuantResNet-20 | 8/8 | - | 200 | |
| QuantResNet-20 | 4/4 | - | 200 | |
| QuantResNet-20 | 2/2 | - | 200 | |
| QuantCNV | 8/8 | - | 200 | |
| QuantCNV | 4/4 | - | 200 | |
| QuantCNV | 2/2 | - | 200 | |

## CIFAR-100

### Standard Models

| Model | Accuracy (%) | Epochs | Notes |
|-------|--------------|--------|-------|
| ResNet-20 | - | 200 | |
| ResNet-56 | - | 200 | |
| ResNet-110 | - | 200 | |
| ResNet-18 | - | 200 | |
| ResNet-50 | - | 200 | |
| ResNet-101 | - | 200 | |
| VGG-16 | - | 200 | |
| VGG-19 | - | 200 | |

### Quantized Models

| Model | Bits (W/A) | Accuracy (%) | Epochs | Notes |
|-------|------------|--------------|--------|-------|
| QuantResNet-20 | 8/8 | - | 200 | |
| QuantResNet-20 | 4/4 | - | 200 | |
| QuantResNet-50 | 8/8 | - | 200 | |
| QuantResNet-50 | 4/4 | - | 200 | |

## MNIST

### Standard Models

| Model | Accuracy (%) | Epochs | Notes |
|-------|--------------|--------|-------|
| ResNet-18 | - | 50 | |
| ResNet-20 | - | 50 | |
| MobileNetV1 | - | 50 | |
| CNV | - | 50 | |

### Quantized Models

| Model | Bits (W/A) | Accuracy (%) | Epochs | Notes |
|-------|------------|--------------|--------|-------|
| QuantCNV | 8/8 | - | 50 | |
| QuantCNV | 4/4 | - | 50 | |
| QuantCNV | 2/2 | - | 50 | |

## Fashion-MNIST

### Standard Models

| Model | Accuracy (%) | Epochs | Notes |
|-------|--------------|--------|-------|
| ResNet-18 | - | 50 | |
| ResNet-20 | - | 50 | |
| MobileNetV1 | - | 50 | |
| CNV | - | 50 | |

---

## Training Configuration

Default training configuration used for benchmarks:

```yaml
optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 5e-4

scheduler:
  name: "cosine"
  warmup_epochs: 5

training:
  batch_size: 128

seed:
  enabled: true
  value: 42
```

## Hardware

- **GPU**: 
- **CUDA Version**: 
- **PyTorch Version**: 

## Notes

- Results format: `mean ± std` (e.g., `93.45 ± 0.12`)
- Number of runs per experiment: 
- All models trained from scratch (no pre-training)
