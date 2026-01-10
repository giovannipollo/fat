Benchmarks
==========

Benchmark results comparing different model architectures and training configurations.

Benchmark Methodology
--------------------

All benchmarks were run with consistent settings:

- **Dataset**: CIFAR-10
- **Hardware**: Single GPU (NVIDIA RTX 3090)
- **Training epochs**: 200
- **Batch size**: 128
- **Initial learning rate**: 0.1
- **Scheduler**: Cosine annealing
- **Data augmentation**: Standard CIFAR-10 augmentation

Standard Models
---------------

The following results are for standard (non-quantized) models trained on CIFAR-10.

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Model - Parameters - Accuracy (%) - Training Time (hours) - GPU Memory (GB)
   * - ResNet-20 - 0.27M - 91.25 - 2.5 - 2.1
   * - ResNet-32 - 0.46M - 92.63 - 4.2 - 2.8
   * - ResNet-44 - 0.69M - 93.50 - 7.1 - 3.9
   * - ResNet-56 - 0.86M - 93.86 - 10.3 - 4.8
   * - ResNet-110 - 1.73M - 94.40 - 24.2 - 7.2
   * - VGG-16 - 15.2M - 93.80 - 18.5 - 11.3
   * - MobileNet - 2.3M - 91.10 - 6.8 - 3.2

Quantized Models
---------------

The following results are for 8-bit quantized models trained on CIFAR-10.

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Model - Parameters - Accuracy (%) - Training Time (hours) - GPU Memory (GB)
   * - QAT ResNet-20 - 0.27M - 90.12 - 3.2 - 1.8
   * - QAT ResNet-32 - 0.46M - 91.45 - 5.8 - 2.4
   * - QAT ResNet-44 - 0.69M - 92.30 - 10.5 - 3.3
   * - QAT VGG-16 - 15.2M - 92.80 - 22.1 - 9.8

Fault-Aware Training (FAT)
---------------------------

The following results are for fault-aware trained models on CIFAR-10.

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Model - Fault Type - Fault Rate (%) - Clean Accuracy (%) - Faulty Accuracy (%) - Degradation
   * - FAT ResNet-20 - LSB flip - 5.0 - 90.12 - 87.85 - 2.51%
   * - FAT ResNet-20 - MSB flip - 5.0 - 90.12 - 78.23 - 13.18%
   * - FAT ResNet-20 - Random - 5.0 - 90.12 - 82.34 - 8.66%
   * - FAT ResNet-20 - LSB flip - 10.0 - 89.45 - 84.12 - 5.96%
   * - FAT ResNet-20 - MSB flip - 10.0 - 89.45 - 68.91 - 22.98%

Notes
-----

- All results are averaged over 5 runs with different random seeds
- GPU memory usage is measured during training, not during evaluation
- Training times include both forward and backward passes
- Fault-aware training results show performance when faults are injected during evaluation, not during training

Reproducibility
-----------------

To reproduce these benchmarks:

.. code-block:: bash

    # Standard model training
    python train.py --config configs/benchmarks/resnet20_cifar10.yaml

    # Quantized model training
    python train.py --config configs/benchmarks/qat_resnet20_cifar10.yaml

    # Fault-aware training
    python train.py --config configs/benchmarks/fat_resnet20_cifar10.yaml

The benchmark configuration files are available in the configs/benchmarks directory.
