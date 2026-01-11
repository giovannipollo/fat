.. _fault_functions_module:

FaultInjectionFunction Class
============================

The FaultInjectionFunction is a custom autograd function that implements fault injection during the forward pass and gradient zeroing during the backward pass. This function is essential for enabling fault-aware training.

Overview
--------

The FaultInjectionFunction extends PyTorch's autograd.Function class to provide custom forward and backward implementations. By controlling gradient flow, this function enables training of models that are robust to hardware faults.

The function serves two primary purposes:

1. **Forward pass**: Apply fault injection by selectively replacing activation values
2. **Backward pass**: Zero gradients at positions where faults were injected

The gradient zeroing is particularly important because it prevents the model from learning to compensate for artificially injected faults, instead training it to produce activations that are resilient to such errors.

Custom Autograd
----------------

PyTorch's autograd system automatically computes gradients during backpropagation. However, for certain operations, we need to customize how gradients flow. The autograd.Function class provides this capability.

When we want to customize gradient flow, we create a function that extends autograd.Function and implements two methods:

- **forward()**: Computation that happens during the forward pass
- **backward()**: Gradient computation that happens during the backward pass

The FaultInjectionFunction uses this mechanism to intercept gradient computation and modify it appropriately.

Forward Pass
--------------

During the forward pass, the function applies fault injection to the activation tensor.

Forward Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The forward() method is a static method that receives the activation tensor and fault-related information:

.. code-block:: python

    @staticmethod
    def forward(ctx, x: Tensor, faulty_values: Tensor, mask: Tensor) -> Tensor:

The parameters are:

- **x**: The original activation tensor from the layer
- **faulty_values**: A tensor containing faulty values to inject
- **mask**: A boolean tensor where True indicates positions to inject faults

The method performs a simple operation: it selects between the original value and the faulty value based on the mask:

.. code-block:: python

    output = torch.where(mask, faulty_values, x)

The torch.where() function creates an output tensor where:

- Positions where mask is True: Use faulty_values
- Positions where mask is False: Use original values from x

Context Saving
^^^^^^^^^^^^^^^^^^^^^^

The forward method saves information to the context for use in the backward pass:

.. code-block:: python

    ctx.save_for_backward(mask)

The context (ctx) is a special object that PyTorch passes from the forward pass to the backward pass. By saving the mask here, we can use it during the backward pass to determine which positions had faults injected.

Only the mask needs to be saved for the backward pass. The original values (x) and faulty values are not needed because the backward pass only needs to know which positions had faults in order to zero gradients at those positions.

Fault Injection Logic
^^^^^^^^^^^^^^^^^^^^^^^^^

The actual fault injection is performed by QuantFaultInjectionLayer, which computes the faulty_values tensor before calling this function. The FaultInjectionFunction is responsible only for applying these faulty values to the appropriate positions, not for computing what those faulty values should be.

This separation of concerns means that:

- **QuantFaultInjectionLayer**: Determines what faults to inject (which strategy, which values, what magnitude)
- **FaultInjectionFunction**: Applies those faults to the tensor and handles gradient flow

The layer computes the integer representation of activations, applies the injection strategy to create faulty values, and then converts these back to floating-point before calling the FaultInjectionFunction.

Backward Pass
-------------

During the backward pass, the function receives gradients from the loss and computes gradients for each input.

Backward Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The backward() method is also a static method that receives the context and the upstream gradient:

.. code-block:: python

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], None, None]:

The parameters are:

- **ctx**: The context object with saved tensors from the forward pass
- **grad_output**: The gradient tensor computed from the loss function

The method returns a tuple with one element for each input to the forward method:

- **grad_x**: The gradient for the original activation tensor (x)
- **None**: No gradient for the faulty_values tensor
- **None**: No gradient for the mask tensor

We return None for faulty_values and mask because we don't want to compute gradients for these. The faulty_values were artificially created, so we don't want to update any parameters based on them. The mask was created based on random selection, so there are no parameters to update for it.

Gradient Zeroing
^^^^^^^^^^^^^^^^^^^^^^^^^

The key operation in the backward pass is zeroing gradients at positions where faults were injected:

.. code-block:: python

    (mask,) = ctx.saved_tensors
    grad_x = torch.where(mask, torch.zeros_like(grad_output), grad_output)

This operation:

1. Retrieves the mask from the context
2. Uses torch.where() to select between zero gradient and the original gradient
3. At positions where mask is True (fault was injected): Use zero gradient
4. At positions where mask is False (no fault): Pass through the original gradient

The result (grad_x) is the gradient that flows back to the quantized layer, which will use it to update its parameters during the optimizer step.

Why Gradient Zeroing Matters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The gradient zeroing is the most important aspect of this function for fault-aware training. To understand why, consider what would happen without it:

Without gradient zeroing:
- Faulty values would produce gradients
- The model would learn to compensate for these specific faulty values
- The model would learn to "undo" the faults we're injecting
- The resulting model would not be robust to new faults at different positions

With gradient zeroing:
- Faulty values produce no gradients
- The model does not learn to compensate for specific faults
- The model learns to produce activations that are robust to faults
- The resulting model generalizes to new faults at different positions

In essence, by zeroing gradients at faulty positions, we tell the model: "these activations are unreliable, so don't update weights based on them." This encourages the model to produce activations that maintain performance even when some values are corrupted.

Backpropagation Flow
^^^^^^^^^^^^^^^^^^^^^^^^

The gradient flows through the entire backward pass as follows:

.. mermaid::

    flowchart LR
        Loss[Loss] --> GO[Grad to Output]
        GO --> FI[FaultInjectionFunction.backward]
        FI --> Mask[Check Saved Mask]
        Mask --> Zero{Mask True?}
        Zero -->|Yes| ZG[Zero Gradient]
        Zero -->|No| PG[Pass Gradient]
        ZG --> GQ[Gradient to QuantLayer]
        PG --> GQ
        GQ --> GW[Weight Update]
        
        style ZG fill:#ffcccc
        style PG fill:#ccffcc
        classDef fault fill:#ffcccc
        classDef clean fill:#ccffcc

The gradient flows from the loss, through the FaultInjectionFunction's backward method, to the quantized layer. At positions where faults were injected, the gradient is zeroed. The quantized layer receives this modified gradient and uses it to update its weights during the optimizer step.

Integration with Training Loop
------------------------------

The FaultInjectionFunction is automatically called by PyTorch's autograd system during backpropagation. There is no need to manually invoke it.

Automatic Invocation
^^^^^^^^^^^^^^^^^^^^^^^^^

When a forward pass uses the FaultInjectionFunction:

.. code-block:: python

    output_value = FaultInjectionFunction.apply(
        x.value,
        injected_float,
        condition_tensor,
    )

PyTorch automatically builds a computational graph that records this operation. During the backward pass (when loss.backward() is called), PyTorch automatically calls the backward() method of the FaultInjectionFunction to compute gradients.

This automatic integration means that the fault injection and gradient handling work seamlessly with the rest of the training loop. No special code is needed in the training script.

Gradient Accumulation
^^^^^^^^^^^^^^^^^^^^^^^^^

The function works correctly with gradient accumulation over multiple batches or micro-batches. Because the mask is saved separately for each forward pass, the gradient zeroing is applied independently to each batch's backward pass.

This means that if you accumulate gradients over several batches before updating weights, the gradients from all batches will have faults zeroed appropriately.

Performance Considerations
--------------------------

The FaultInjectionFunction is designed to have minimal overhead during both forward and backward passes.

Forward Pass Overhead
^^^^^^^^^^^^^^^^^^^^^^^^^^

During the forward pass, the function performs:

- **Mask retrieval**: Accessing the saved mask (trivial operation)
- **torch.where()**: A single element-wise operation

The torch.where() operation is highly optimized in PyTorch and runs efficiently on both CPU and GPU. The overhead is negligible compared to the actual layer computation that produces the activations.

Backward Pass Overhead
^^^^^^^^^^^^^^^^^^^^^^^^^^^

During the backward pass, the function performs:

- **Mask retrieval**: Accessing the saved mask from context
- **torch.zeros_like()**: Creating a zero tensor of the same shape
- **torch.where()**: Selecting between zero gradient and original gradient

These operations are also highly optimized and run efficiently on GPU. The backward pass overhead is small compared to the gradient computations for the actual neural network layers.

Memory Usage
^^^^^^^^^^^^^^^^^^^^^^

The function does not allocate significant additional memory. The only tensor that is created during the backward pass is the zero tensor for gradient zeroing, which has the same shape as the gradient tensor. This tensor is freed after the backward pass completes.

The mask tensor is saved in the context during the forward pass and retrieved during the backward pass, but this is just a reference, not a copy, so there is no additional memory allocation for the mask itself.

Numerical Stability
--------------------

The function is designed to be numerically stable and work correctly with different numerical representations.

Gradient Handling
^^^^^^^^^^^^^^^^^^^^^

The function handles gradients correctly even in edge cases:

- **Zero gradient**: If the upstream gradient is all zeros (which can happen in some loss functions), the zeroed gradient remains zero
- **NaN gradients**: If the upstream gradient contains NaN values (which can happen with certain operations), these NaN values are preserved at non-faulty positions and replaced with zeros at faulty positions
- **Infinite gradients**: Similarly, infinite gradients are preserved at clean positions and zeroed at faulty positions

This handling ensures that numerical issues in the gradient computation are not inadvertently caused or masked by the fault injection system.

Bitwise Precision
^^^^^^^^^^^^^^^^^^^^^

Because the function operates on floating-point tensors, it works correctly with the precision of the model. Whether the model uses float32, float16, or any other precision, the function will work correctly.

The fault injection itself occurs in integer space (within QuantFaultInjectionLayer), but by the time the FaultInjectionFunction sees the tensors, they have been converted back to floating-point. This means that the function operates with the same precision as the rest of the model.

API Reference
------------

.. autoclass:: utils.fault_injection.functions.FaultInjectionFunction
    :members:
    :undoc-members:
    :show-inheritance:
