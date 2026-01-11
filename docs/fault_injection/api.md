.. _fault_injection_api:

Fault Injection API Reference
==============================

This section provides the complete API reference for all fault injection components, auto-generated from the source code documentation.

Core Components
----------------

.. autoclass:: utils.fault_injection.FaultInjectionConfig
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: utils.fault_injection.FaultInjector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: utils.fault_injection.wrapper._InjectionWrapper
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: utils.fault_injection.layers.QuantFaultInjectionLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: utils.fault_injection.functions.FaultInjectionFunction
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: utils.fault_injection.statistics.FaultStatistics
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: utils.fault_injection.statistics.LayerStatistics
    :members:
    :undoc-members:
    :show-inheritance:

Injection Strategies
------------------

.. autoclass:: utils.fault_injection.strategies.base.InjectionStrategy
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: utils.fault_injection.strategies.random.RandomStrategy
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: utils.fault_injection.strategies.lsb_flip.LSBFlipStrategy
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: utils.fault_injection.strategies.msb_flip.MSBFlipStrategy
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: utils.fault_injection.strategies.full_flip.FullFlipStrategy
    :members:
    :undoc-members:
    :show-inheritance:

Module Contents
---------------

.. automodule:: utils.fault_injection
    :members:
    :undoc-members:

Functions
---------

.. autofunction:: utils.fault_injection.strategies.get_strategy
    :members:
    :show-inheritance:

.. autofunction:: utils.fault_injection.strategies.list_strategies
    :members:
    :show-inheritance:
