Installation
============

This guide explains how to install and set up the FAT training framework.

Prerequisites
-------------

The framework requires:

- **Python 3.12.3** or higher
- **PyTorch 2.0+** for deep learning
- **Git** for cloning the repository (if installing from source)

For GPU support, you will need appropriate CUDA drivers and a CUDA-enabled PyTorch installation.

Installation Steps
------------------

Clone the Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, clone the repository and navigate to the project directory:

.. code-block:: bash

    git clone <repository-url>
    cd training-framework

Create Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create and activate a Python virtual environment:

.. code-block:: bash

    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # or
    venv\Scripts\activate  # On Windows

Install Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

Install all required dependencies:

.. code-block:: bash

    pip install -r requirements.txt

This will install PyTorch, torchvision, Brevitas, and other required packages.

Verify Installation
^^^^^^^^^^^^^^^^^^^^^^

Verify that the installation is successful:

.. code-block:: python

    import torch
    import brevitas
    print(f"PyTorch version: {torch.__version__}")
    print(f"Brevitas version: {brevitas.__version__}")

Documentation Installation
------------------------

To build and view the documentation:

.. code-block:: bash

    # Install documentation dependencies
    pip install -r docs/requirements.txt

    # Build HTML documentation
    cd docs
    sphinx-build -b html . _build/html

    # Open in browser
    open _build/html/index.html  # macOS
    # or
    xdg-open _build/html/index.html  # Linux

Next Steps
----------

After installation, proceed to the Quick Start guide to train your first model.
