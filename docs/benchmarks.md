# Benchmark Results

Benchmark results for models trained on various datasets. Results show **test_accuracies** across multiple runs. All models were trained with the `deterministic` setting enabled for reproducibility. The chosen seeds for each run are as follows:

- Run 1: 0
- Run 2: 1
- Run 3: 2

Deterministic training is not fully supported for all the quantized models; hence, some results may vary slightly between runs.

## MNIST

### Standard Models

| Model       | Accuracy (%) | Epochs | Notes |
| ----------- | ------------ | ------ | ----- |
| ResNet-18   | -            | 50     |       |
| ResNet-20   | -            | 50     |       |
| MobileNetV1 | -            | 50     |       |
| CNV         | -            | 50     |       |

### Quantized Models

| Model    | Bits (W/A) | Accuracy (%) | Epochs | Notes |
| -------- | ---------- | ------------ | ------ | ----- |
| QuantCNV | 8/8        | -            | 50     |       |
| QuantCNV | 4/4        | -            | 50     |       |
| QuantCNV | 2/2        | -            | 50     |       |

## Fashion-MNIST

### Standard Models

| Model       | Accuracy (%) | Epochs | Notes |
| ----------- | ------------ | ------ | ----- |
| ResNet-18   | -            | 50     |       |
| ResNet-20   | -            | 50     |       |
| MobileNetV1 | -            | 50     |       |
| CNV         | -            | 50     |       |

## CIFAR-10

### Standard Models

| Model       | Mean (%) | Std (%) | Run 1 (%) | Run 2 (%) | Run 3 (%) | Config File                             |
| ----------- | -------: | ------: | --------: | --------: | --------: | --------------------------------------- |
| MobileNetV1 |    90.91 |    0.31 |     90.85 |     91.25 |     90.64 | `configs/benchmark/cifar10_mobilenetv1` |
| CNV         |    91.09 |    0.15 |     91.19 |     91.16 |     90.91 | `configs/benchmark/cifar10_cnv.yaml`    |
| ResNet-20   |    92.96 |   0.05  |     93.02 |     92.92 |     92.93 | `configs/benchmark/cifar10_resnet20.yaml` |
| ResNet-32   |    xx.xx |   xx.xx |     93.77 |     93.61 |     93.76 | `configs/benchmark/cifar10_resnet32.yaml` |
| ResNet-44   |    xx.xx |   xx.xx |     93.82 |     94.14 |     94.19 | `configs/benchmark/cifar10_resnet44.yaml` |
| ResNet-56   |    xx.xx |   xx.xx |     94.32 |     94.27 |     94.41 | `configs/benchmark/cifar10_resnet56.yaml` |
| ResNet-110  |    xx.xx |   xx.xx |     94.25 |     94.21 |     94.61 | `configs/benchmark/cifar10_resnet110.yaml`|

### Quantized Models

| Model    | Bits (W/A) | Mean (%) | Std (%) | Run 1 (%) | Run 2 (%) | Run 3 (%) | Config File |
| -------- | ---------: | -------: | ------: | --------: | --------: | --------: | ----------- |
| QuantCNV |        8/8 |    91.43 |   0.19  |     91.52 |     91.56 |     91.219 | `configs/benchmark/cifar10_quant_cnv_w8a8.yaml`|
| QuantCNV |        4/4 |    xx.xx |   xx.xx |     91.28 |     91.29 |     91.29 | `configs/benchmark/cifar10_quant_cnv_w4a4.yaml`|
| QuantCNV |        2/2 |    xx.xx |   xx.xx |     90.43 |     90.43 |     90.57 | `configs/benchmark/cifar10_quant_cnv_w2a2.yaml`|

## CIFAR-100

### Standard Models

| Model       | Mean (%) | Std (%) | Run 1 (%) | Run 2 (%) | Run 3 (%) | Config File                             |
| ----------- | -------: | ------: | --------: | --------: | --------: | --------------------------------------- |
| MobileNetV1 |    xx.xx |   xx.xx |     69.30 |     69.08 |     68.76 | `configs/benchmark/cifar100_mobilenetv1` |
| CNV         |    xx.xx |   xx.xx |     63.88 |     64.01 |     63.50 | `configs/benchmark/cifar100_cnv.yaml`    |
| ResNet-20   |    xx.xx |   xx.xx |     69.09 |     69.48 |     69.86 | `configs/benchmark/cifar100_resnet20.yaml` |
| ResNet-32   |    xx.xx |   xx.xx |     71.26 |     70.28 |     71.07 | `configs/benchmark/cifar100_resnet32.yaml` |
| ResNet-44   |    xx.xx |   xx.xx |     xx.xx |     xx.xx |     xx.xx | `configs/benchmark/cifar100_resnet44.yaml` |
| ResNet-56   |    xx.xx |   xx.xx |     xx.xx |     xx.xx |     xx.xx | `configs/benchmark/cifar100_resnet56.yaml` |
| ResNet-110  |    xx.xx |   xx.xx |     xx.xx |     xx.xx |     xx.xx | `configs/benchmark/cifar100_resnet110.yaml` |

### Quantized Models

| Model    | Bits (W/A) | Mean (%) | Std (%) | Run 1 (%) | Run 2 (%) | Run 3 (%) | Config File |
| -------- | ---------: | -------: | ------: | --------: | --------: | --------: | ----------- |
| QuantCNV |        8/8 |    xx.xx |   xx.xx |     66.17 |     66.16 |     65.77 | `configs/benchmark/cifar100_quant_cnv_w8a8.yaml`|
| QuantCNV |        4/4 |    xx.xx |   xx.xx |     65.58 |     65.50 |     65.53 | `configs/benchmark/cifar100_quant_cnv_w4a4.yaml`|
| QuantCNV |        2/2 |    xx.xx |   xx.xx |     65.31 |     65.09 |     64.72 | `configs/benchmark/cifar100_quant_cnv_w2a2.yaml`|


## Hardware

- **CUDA Version**: 12.8
- **Driver Version**: 570.195.03
- **PyTorch Version**: 2.9.1
