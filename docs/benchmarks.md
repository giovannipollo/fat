# Benchmark Results

Benchmark results for models trained on various datasets. Results show **test_accuracies** across multiple runs. All models were trained with the `deterministic` setting enabled for reproducibility. The chosen seeds for each run are as follows:
- Run 1: 0
- Run 2: 1
- Run 3: 2

## CIFAR-10

### Standard Models

| Model | Mean (%) | Std (%) | Run 1 (%) | Run 2 (%) | Run 3 (%) | Config File |
|-------|----------:|----------:|----------:|---------:|--------:|-------------|
| MobileNetV1 | 90.91 | 0.31 | 90.85 | 91.25 | 90.64 |  `configs/benchmark/cifar10_mobilenetv1` |
| CNV | 91.09 | 0.15 | 91.19 | 91.16 | 90.91 |  `configs/benchmark/cifar10_cnv.yaml` |

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

## Hardware

- **GPU**: RTX 4060 Ti
- **CUDA Version**: 12.8
- **Driver Version**: 570.195.03
- **PyTorch Version**: 2.9.1
