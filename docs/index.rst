.. FAT - Training Framework documentation master file

FAT - Training Framework
========================

A modular PyTorch training framework for image classification, with support for standard and quantized models and comprehensive fault injection capabilities for fault-aware training (FAT).

.. warning::
   Tested with Python 3.12.3

Features
--------

- Multiple architectures: ResNet, VGG, MobileNet, CNV
- Quantization-aware training (QAT) with Brevitas
- **Fault-aware training (FAT)** for training robust models
- **Comprehensive fault injection framework** for evaluating model resilience
- Multiple datasets: CIFAR-10/100, MNIST, Fashion-MNIST
- Configurable loss functions, optimizers, and schedulers
- Mixed precision training (AMP)
- TensorBoard logging
- Checkpoint saving and resuming
- Reproducible training with seed control

Quick Start
-----------

Train with default configuration::

    python train.py

Use a custom configuration::

    python train.py --config configs/resnet20_cifar10.yaml

Example configuration:

.. code-block:: yaml

    dataset:
      name: "cifar10"
      root: "./data"

    model:
      name: "resnet20"

    training:
      batch_size: 128
      epochs: 200

    optimizer:
      name: "sgd"
      learning_rate: 0.1
      momentum: 0.9

    scheduler:
      name: "cosine"
      T_max: 200

Building Documentation
--------------------

To build the HTML documentation locally:

.. code-block:: bash

    # Install dependencies
    pip install -r docs/requirements.txt

    # Build the documentation
    cd docs
    sphinx-build -b html . _build/html

    # Open in browser
    open _build/html/index.html  # macOS
    # or
    xdg-open _build/html/index.html  # Linux

Table of Contents
----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   getting-started/installation
   getting-started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Framework Components:

   models/index
   datasets/index
   configuration/index

.. toctree::
   :maxdepth: 3
   :caption: Fault Injection Documentation:

   fault_injection/overview
   fault_injection/injector
   fault_injection/wrapper
   fault_injection/layers
   fault_injection/config
   fault_injection/statistics
   fault_injection/functions
   fault_injection/strategies/index
   fault_injection/strategies/base
   fault_injection/strategies/random
   fault_injection/strategies/lsb_flip
   fault_injection/strategies/msb_flip
   fault_injection/strategies/full_flip
   fault_injection/api

.. toctree::
   :maxdepth: 1
   :caption: Other:

   benchmarks

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
