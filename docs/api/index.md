# API Reference

Complete API documentation for the FAT training framework.

## Modules

### Datasets

Dataset implementations for image classification tasks.

| Module | Description |
|--------|-------------|
| [`datasets.base`](datasets/base.md) | Abstract base class for datasets |
| [`datasets.cifar10`](datasets/cifar10.md) | CIFAR-10 dataset |
| [`datasets.cifar100`](datasets/cifar100.md) | CIFAR-100 dataset |
| [`datasets.mnist`](datasets/mnist.md) | MNIST dataset |
| [`datasets.fashion_mnist`](datasets/fashion_mnist.md) | FashionMNIST dataset |

### Models

Neural network architectures for image classification.

#### Standard Models

| Module | Description |
|--------|-------------|
| [`models`](models/index.md) | Model registry and factory functions |
| [`models.standard.cnv`](models/cnv.md) | CNV (Compact Neural Vision) architecture |
| [`models.standard.mobilenet`](models/mobilenet.md) | MobileNetV1 architecture |
| [`models.standard.vgg`](models/vgg.md) | VGG architectures |
| [`models.standard.resnet_cifar`](models/resnet_cifar.md) | ResNet for CIFAR (20, 32, 44, 56, 110) |
| [`models.standard.resnet_imagenet`](models/resnet_imagenet.md) | ResNet for ImageNet (18, 34, 50, 101, 152) |

#### Quantized Models

| Module | Description |
|--------|-------------|
| [`models.quantized`](models/quantized.md) | Quantized models using Brevitas |

### Utilities

Training utilities and helper functions.

| Module | Description |
|--------|-------------|
| [`utils.trainer`](utils/trainer.md) | Training loop orchestration |
| [`utils.optimizer`](utils/optimizer.md) | Optimizer factory |
| [`utils.scheduler`](utils/scheduler.md) | Learning rate scheduler factory |
| [`utils.loss`](utils/loss.md) | Loss function factory |
| [`utils.experiment`](utils/experiment.md) | Experiment management and checkpoints |
| [`utils.logging`](utils/logging.md) | TensorBoard and console logging |
| [`utils.config`](utils/config.md) | Configuration loading |
| [`utils.device`](utils/device.md) | Device detection |
| [`utils.seed`](utils/seed.md) | Reproducibility utilities |
