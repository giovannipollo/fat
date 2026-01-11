.. _fault_injector_module:

FaultInjector Class
==================

The FaultInjector class is responsible for transforming models at runtime to add fault injection layers. It walks the model graph, identifies target quantized layers, and inserts fault injection capability without modifying the original model definition.

Overview
--------

The FaultInjector serves as the primary entry point for adding fault injection capabilities to a neural network model. By analyzing the model structure and inserting fault injection layers at strategic positions, it enables fault-aware training and evaluation without requiring any changes to the model implementation.

This runtime transformation approach has several advantages:

- **Non-invasive**: Original model code remains unchanged
- **Flexible**: Injection can be enabled/disabled dynamically
- **Layer-specific**: Different parameters can be applied to different layers
- **Removable**: Injection layers can be completely removed to restore original model

Target Layers
------------

The injector specifically targets quantized activation layers that produce QuantTensor outputs. These are the ideal injection points because they represent intermediate activations in quantized neural network. The following layer types are targeted:

- **QuantIdentity**: Identity layers with quantization
- **QuantReLU**: ReLU activation with quantization
- **QuantHardTanh**: HardTanh activation with quantization

These layer types are identified because they are common points where quantized activations appear in Brevitas quantized models and they represent meaningful intermediate representations.

Injection Process
---------------

The injection process consists of several phases that transform a regular model into one with fault injection capabilities.

Model Transformation Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

    flowchart TD
        Start[Start: Load Model] --> Check{Fault<br/>Enabled?}
        Check -->|No| Return[Return Original Model]
        Check -->|Yes| Parse[Parse Configuration]
        Parse --> GetStrat[Get Injection Strategy]
        GetStrat --> Walk[Walk Model Graph]
        Walk --> Found{Is Target<br/>Layer?}
        Found -->|No| Recurse[Process Children]
        Recurse --> Walk
        Found -->|Yes| Create[Create Injection Layer]
        Create --> Wrap[Create Wrapper]
        Wrap --> Insert[Insert Wrapper]
        Insert --> Walk
        Walk --> Done[Return Transformed Model]
        
        style Found fill:#ffcccc
        style Create fill:#ccffcc
        style Done fill:#e6f3ff

The injector recursively walks the model graph starting from the top-level model. For each child module, it checks whether it is a target layer type. If it is a target, a new QuantFaultInjectionLayer is created and combined with the original layer using _InjectionWrapper. If it is not a target, the injector recurses into that module to continue the search.

This recursive approach ensures that fault injection layers are placed at every appropriate position in the model, regardless of how deeply nested the quantized layers are within the model hierarchy.

Wrapper Creation
^^^^^^^^^^^^^^^^^^^^^^

When a target layer is identified, the injector creates a wrapper that combines the original layer with a new fault injection layer. The wrapper maintains the interface of the original layer while adding fault injection functionality.

The wrapper is then inserted back into the parent module at the same position as the original layer. This effectively replaces the original layer with an equivalent layer that now has fault injection capabilities.

Key Implementation Details
-------------------------

The FaultInjector implementation includes several important design choices that enable flexible and robust fault injection.

Module Hierarchy Handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The injector must work with different types of module containers. PyTorch provides several ways to organize child modules:

- **Sequential containers**: Layers in a sequence
- **ModuleList containers**: Lists of modules
- **Regular modules**: Modules with named attributes

The injector handles all these cases by using named_children() to iterate over modules and setting children appropriately. For ModuleList, it uses index-based assignment. For regular modules, it uses attribute assignment. This ensures that the wrapper replacement works correctly regardless of how the model is structured.

Layer Tracking
^^^^^^^^^^^^^^^^^^^^^

The injector maintains a list of all created QuantFaultInjectionLayer instances. This tracking serves several purposes:

- Enables statistics to be attached to all layers
- Allows probability updates across all layers
- Supports enabling/disabling injection globally
- Provides counts for verification and debugging

Each layer is assigned a unique layer_id during creation, which is used for statistics reporting and layer-specific configuration.

Strategy Instantiation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The injection strategy is instantiated once during the inject() call and then shared across all layers. This ensures consistent fault behavior across the model and is more efficient than creating separate strategy instances for each layer. The strategy object is passed to each QuantFaultInjectionLayer during creation.

Runtime Control
--------------

After injection is complete, the injector provides several methods for controlling fault injection behavior during training and evaluation.

Probability Updates
^^^^^^^^^^^^^^^^^^^^^^

The injection probability can be updated at runtime, allowing for dynamic fault injection strategies. For example, you might want to:

- **Anneal probability**: Start with high probability and decrease during training
- **Curriculum learning**: Start with 0% and gradually increase
- **Phase-specific**: Use different probabilities during training vs. evaluation

The update_probability() method can target a specific layer or all layers, providing fine-grained control.

Enable/Disable
^^^^^^^^^^^^^^^^^^^^^^

Fault injection can be globally enabled or disabled without removing injection layers. This is useful for:

- **Ablation studies**: Compare performance with and without faults
- **Debugging**: Temporarily disable injection to isolate other issues
- **Evaluation phases**: Only inject faults during specific evaluation runs

The set_enabled() method toggles injection across all layers in a single operation.

Statistics Integration
^^^^^^^^^^^^^^^^^^^^^^^^^

When statistics tracking is enabled, the FaultStatistics object must be attached to all injection layers. The set_statistics() method performs this attachment across all layers in the model, ensuring that every injection event is recorded.

Layer Removal
-------------

The injector can completely remove all fault injection layers from a model, restoring it to its original state.

Removal Process
^^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

    flowchart LR
        Start[Start: Remove Injection] --> Iterate[Iterate Over<br/>All Children]
        Iterate --> CheckWrap{Is<br/>_InjectionWrapper?}
        CheckWrap -->|Yes| Unwrap[Unwrap to Get<br/>Original Layer]
        Unwrap --> Restore[Restore in<br/>Parent Module]
        CheckWrap -->|No| Skip[Skip - Not a Wrapper]
        Skip --> Next[Continue to<br/>Next Child]
        Restore --> Next
        Next --> Continue{More<br/>Children?}
        Continue -->|Yes| Iterate
        Continue -->|No| Clear[Clear Layer<br/>Tracking]
        Clear --> Done[Done: Model Restored]
        
        style Unwrap fill:#ffcccc
        style Restore fill:#ccffcc
        style Done fill:#e6f3ff

The removal process recursively walks the model and identifies all _InjectionWrapper instances. For each wrapper found, it extracts the original wrapped layer and restores it in the parent module at the same position. This completely removes fault injection capability and restores the model to its original architecture.

After unwrapping all wrappers, the injector clears its internal tracking lists. The returned model is now identical to the model before injection was applied.

Usage Examples
--------------

Basic Injection
^^^^^^^^^^^^^^^^^^

To inject faults into a model, create a FaultInjector instance and call the inject() method with a configuration:

.. code-block:: python

    from utils.fault_injection import FaultInjector, FaultInjectionConfig

    # Create configuration
    config = FaultInjectionConfig(
        enabled=True,
        probability=5.0,
        injection_type="lsb_flip",
        apply_during="train",
    )

    # Create injector and inject
    injector = FaultInjector()
    model = injector.inject(model, config)

This inserts fault injection layers at all target positions in the model. The model is modified in-place, so the returned model is the same object as the input.

Runtime Updates
^^^^^^^^^^^^^^^^^^

Update parameters during training:

.. code-block:: python

    # Update probability (all layers)
    injector.update_probability(model, probability=10.0)

    # Update specific layer
    injector.update_probability(model, probability=15.0, layer_id=5)

    # Disable injection temporarily
    injector.set_enabled(model, enabled=False)

These operations take effect immediately on subsequent forward passes.

Statistics Tracking
^^^^^^^^^^^^^^^^^^^^^^

Attach statistics tracker to all layers:

.. code-block:: python

    from utils.fault_injection import FaultStatistics

    # Create statistics tracker
    stats = FaultStatistics(num_layers=10)

    # Attach to model
    injector.set_statistics(model, stats)

    # Train model...

    # View statistics
    stats.print_report()

Removal
^^^^^^^^^^^^^^^^^^

Remove all injection layers to restore original model:

.. code-block:: python

    model = injector.remove(model)

This operation unwraps all wrappers and returns the model to its original structure.

API Reference
------------

.. autoclass:: utils.fault_injection.FaultInjector
    :members:
    :undoc-members:
    :show-inheritance:
