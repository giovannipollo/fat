FAT Documentation
=================

This directory contains the Sphinx documentation for the FAT training framework.

Building the Documentation
-------------------------

To build the HTML documentation:

1. Install dependencies:

.. code-block:: bash

    pip install -r requirements.txt

2. Build the documentation:

.. code-block:: bash

    cd docs
    sphinx-build -b html . _build/html

3. Open the documentation:

.. code-block:: bash

    open _build/html/index.html    # On macOS
    xdg-open _build/html/index.html # On Linux

Documentation Structure
---------------------

The documentation is organized into the following sections:

- **Getting Started**: Installation and quick start guides
- **Models**: Overview of available model architectures
- **Datasets**: Overview of supported datasets
- **Configuration**: Complete configuration reference
- **Fault Injection**: Comprehensive documentation of the fault injection framework
- **API Reference**: Auto-generated API documentation for all components

Fault Injection Documentation
----------------------------

The fault injection documentation includes extensive coverage of:

- High-level architecture and design decisions
- Component-level documentation for each major class
- Strategy descriptions for all fault types
- Usage examples and integration guides
- Mermaid diagrams for visualizing system behavior

Key fault injection documentation files:

- ``overview.rst``: Introduction to fault injection and system architecture
- ``injector.rst``: FaultInjector class for model transformation
- ``wrapper.rst``: _InjectionWrapper class for layer composition
- ``layers.rst``: QuantFaultInjectionLayer for runtime fault injection
- ``config.rst``: FaultInjectionConfig for parameter management
- ``statistics.rst``: FaultStatistics and LayerStatistics for tracking
- ``functions.rst``: FaultInjectionFunction for gradient handling
- ``strategies/index.rst``: Overview of fault injection strategies
- ``strategies/base.rst``: Base class for implementing custom strategies
- ``strategies/random.rst``: Random value replacement strategy
- ``strategies/lsb_flip.rst``: Least significant bit flipping strategy
- ``strategies/msb_flip.rst``: Most significant bit flipping strategy
- ``strategies/full_flip.rst``: Full bit inversion strategy
- ``api.rst``: Auto-generated API reference for all components

Customization
-------------

The documentation is built with:

- **Sphinx 7.0+**: Documentation generator
- **sphinx-rtd-theme**: ReadTheDocs theme for professional appearance
- **sphinxcontrib-mermaid**: Mermaid diagram support for visualizations
- **sphinx.ext.napoleon**: Google-style docstring support
- **sphinx.ext.autodoc**: Auto-generated API documentation

The configuration is in ``conf.py`` and can be customized as needed.

For more information about using Sphinx, see: https://www.sphinx-doc.org/
