.. _injection_wrapper_module:

_InjectionWrapper Class
=====================

The _InjectionWrapper class combines a neural network layer with a fault injection layer, enabling fault injection to occur after the layer executes without modifying the layer itself. This wrapper pattern is essential for inserting fault injection capability into existing model structures.

Overview
--------

The wrapper serves as a simple container that holds two components:

- **Wrapped layer**: The original neural network layer that processes input normally
- **Injection layer**: A QuantFaultInjectionLayer that injects faults into the wrapped layer's output

By composing these two components in a single module, the wrapper presents the same interface as the original layer while adding fault injection functionality. This is particularly important for layers that are not part of a Sequential container, where we cannot simply insert a new layer after them.

Design Rationale
---------------

Why Wrapping is Necessary
~~~~~~~~~~~~~~~~~~~~~~~~~

In PyTorch models, layers can be organized in several ways:

- **Sequential**: Layers are executed in sequence, allowing easy insertion of new layers
- **ModuleList**: Layers stored in a list, still allowing insertion
- **Named attributes**: Layers as direct attributes of a module

For Sequential and ModuleList containers, we could theoretically insert a new fault injection layer directly into the sequence. However, for layers that are named attributes of their parent module, there is no existing "slot" where we can insert a new layer.

The wrapper pattern solves this problem by replacing the layer itself with a wrapper that contains both the original layer and the injection layer. This works regardless of how the layer is organized within the parent module.

Wrapper Structure
~~~~~~~~~~~~~~~~

The wrapper maintains a simple structure with two main components:

.. mermaid::

    classDiagram
        class _InjectionWrapper {
            +wrapped_layer: nn.Module
            +injection_layer: QuantFaultInjectionLayer
            +forward(x: Tensor) -> Tensor
            +__repr__() -> str
        }

        class nn.Module {
            <<abstract>>
            +forward(x)
        }

        class QuantFaultInjectionLayer {
            +layer_id: int
            +probability: float
            +strategy: InjectionStrategy
            +forward(x: QuantTensor) -> QuantTensor
        }

        _InjectionWrapper --> nn.Module: inherits
        _InjectionWrapper *-- QuantFaultInjectionLayer: contains
        _InjectionWrapper *-- nn.Module: wraps

The wrapper inherits from nn.Module, so it can be treated as a regular PyTorch layer. It contains references to both the wrapped layer and the injection layer.

Forward Pass Behavior
--------------------

The wrapper's forward method is straightforward: execute the wrapped layer, then pass the result through the injection layer.

Execution Flow
~~~~~~~~~~~~~~

.. mermaid::

    flowchart LR
        Input[Input Tensor] --> WL[Wrapped Layer]
        WL --> Output[Layer Output]
        Output --> IL[Injection Layer]
        IL --> Faulty[Faulty Output]
        Faulty --> Return[Return Result]
        
        style WL fill:#e6f3ff
        style IL fill:#ffe6e6
        style Faulty fill:#fff4e6

When the wrapper receives input during a forward pass, it first calls the wrapped layer's forward method with the input. This produces the layer's normal output. The wrapper then passes this output through the injection layer, which may inject faults based on the configuration. The final result is returned.

This two-step process ensures that the wrapped layer operates exactly as it would without fault injection, and only after its normal execution are faults applied to the output.

Preserving Layer Identity
~~~~~~~~~~~~~~~~~~~~~~~~~~

The wrapper maintains the wrapped layer's identity by:

- **Storing reference**: Keeping a reference to the original layer object
- **Transparent execution**: Calling the layer's forward method directly
- **Preserving parameters**: The wrapped layer's parameters remain accessible
- **Maintaining state**: Layer-specific state (like running statistics in batch norm) is preserved

This is important because the wrapped layer may have learned parameters that should not be altered by the wrapper's presence.

Integration with Parent Module
----------------------------

The wrapper is designed to integrate seamlessly with the parent module's structure.

Module Replacement
~~~~~~~~~~~~~~~~~

When the FaultInjector identifies a target layer, it creates a wrapper and replaces the layer in the parent module:

.. mermaid::

    flowchart TD
        Parent[Parent Module] --> Orig[Original Layer]
        Parent --> Wrapper[_InjectionWrapper]
        
        subgraph "Original State"
            Orig
        end
        
        subgraph "After Injection"
            Wrapper --> WL[Wrapped Layer]
            Wrapper --> IL[Injection Layer]
        end
        
        style Wrapper fill:#ffcccc
        style IL fill:#ffe6e6

The parent module's reference to the layer is updated to point to the wrapper instead. This is done using the appropriate method for the container type (attribute assignment for named attributes, index assignment for ModuleList).

From the perspective of the parent module and any code that uses it, the wrapper appears to be a drop-in replacement for the original layer. The interface is the same (forward method with same signature), and the behavior is similar but with fault injection capability.

Container Compatibility
~~~~~~~~~~~~~~~~~~~~~

The wrapper is designed to work with different container types. The FaultInjector handles the details of setting the wrapper as a child of the parent module correctly, whether the parent uses:

- **setattr()**: For layers stored as named attributes
- **__setitem__()**: For layers in ModuleList
- **Other methods**: As needed for custom container types

This flexibility ensures that the wrapper can be inserted into any model structure.

Backward Pass Behavior
----------------------

During the backward pass, gradients flow through both components of the wrapper.

Gradient Flow
~~~~~~~~~~~~~

.. mermaid::

    flowchart LR
        Loss[Loss] --> GradIL[Gradient to<br/>Injection Layer]
        GradIL --> BackIL[Injection Layer<br/>Backward]
        BackIL --> GradWL[Gradient to<br/>Wrapped Layer]
        GradWL --> BackWL[Wrapped Layer<br/>Backward]
        BackWL --> GradIn[Gradient to<br/>Wrapper]
        
        subgraph "Gradient Modification"
            BackIL --> Zero[Zero gradients at<br/>faulty positions]
            Zero --> GradWL
        end
        
        style Zero fill:#ffcccc

The injection layer's backward implementation (via FaultInjectionFunction) zeros gradients at positions where faults were injected. This prevents the model from learning to predict the faulty values. The wrapped layer then receives the modified gradients and updates its parameters accordingly.

The wrapper itself does not perform any gradient manipulation - it simply passes gradients through to both contained layers.

Advantages of the Wrapper Pattern
---------------------------------

The wrapper pattern provides several key advantages for the fault injection system.

Non-Invasive Integration
~~~~~~~~~~~~~~~~~~~~~~~

The most important advantage is that the original layer is never modified. This means:

- **Layer implementation remains unchanged**: No need to modify Brevitas or other library code
- **Layer functionality preserved**: All layer-specific features work as intended
- **Easy to remove**: The wrapper can be unwrapped to restore the original layer
- **Layer-specific behavior maintained**: Any layer-specific optimizations or methods remain available

This non-invasive approach is crucial for working with third-party libraries like Brevitas where we cannot modify the layer implementations.

Clean Separation of Concerns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The wrapper provides a clean separation between the layer's normal operation and fault injection logic. This separation has several benefits:

- **Layer logic isolated**: The layer's forward method does not need to know about faults
- **Injection logic isolated**: Fault injection code does not depend on layer implementation
- **Easier to maintain**: Changes to injection logic don't affect layer code
- **Easier to test**: Each component can be tested independently
- **Easier to debug**: Issues can be isolated to either layer or injection component

Modular and Reusable
~~~~~~~~~~~~~~~~~~~~~

The wrapper is a generic component that can wrap any layer. This means:

- **Works with any layer type**: Not limited to specific quantized layers
- **Same code for all layers**: No need for layer-specific wrapper implementations
- **Easy to extend**: Can add additional functionality to the wrapper if needed
- **Consistent behavior**: All wrapped layers behave identically from fault injection perspective

Limitations and Considerations
-----------------------------

While the wrapper pattern is effective for this use case, there are some limitations to be aware of.

Performance Overhead
^^^^^^^^^^^^^^^^^^^^^^

The wrapper adds a small overhead because it involves:

- **Additional function call**: The wrapper's forward method adds one level of indirection
- **Module lookup**: Accessing the wrapped and injection layers requires attribute access

However, this overhead is negligible compared to the actual layer computation and is acceptable given the flexibility it provides.

Access to Layer-Specific Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a layer is wrapped, layer-specific methods that are not part of the base nn.Module interface are no longer directly accessible through the wrapper. If you need to call layer-specific methods, you must access the wrapped_layer attribute directly.

For example, if a layer has a custom method:

.. code-block:: python

    # After wrapping, access the original layer
    wrapper.parent_module.some_layer.wrapped_layer.custom_method()

This is generally not an issue because the primary interaction with layers is through the forward method, which the wrapper preserves.

Memory Considerations
^^^^^^^^^^^^^^^^^^^^^^

The wrapper maintains references to both the wrapped layer and injection layer, which means both remain in memory. In large models with many layers, this can increase memory usage slightly. However, this is typically acceptable because:

- The increase is linear in the number of layers
- Each wrapper adds only a small amount of overhead
- The overhead is negligible compared to the model parameters themselves

Removal and Restoration
------------------------

The wrapper can be removed to completely restore the original model structure.

Unwrapping Process
^^^^^^^^^^^^^^^^^^^^^

To remove a wrapper and restore the original layer, the FaultInjector's remove() method:

1. Identifies all _InjectionWrapper instances in the model
2. Extracts the wrapped_layer from each wrapper
3. Replaces the wrapper in the parent module with the wrapped layer
4. Clears the wrapper reference

This process completely removes fault injection capability and returns the model to its original state.

After unwrapping, the model is identical to what it was before injection. All parameters, state, and functionality are restored.

API Reference
------------

.. autoclass:: utils.fault_injection.wrapper._InjectionWrapper
    :members:
    :undoc-members:
    :show-inheritance:
