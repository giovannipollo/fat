.. _strategy_base_module:

InjectionStrategy Base Class
=============================

The InjectionStrategy is the abstract base class for all fault injection strategies. It defines the interface that all strategies must implement, ensuring that different fault models can be used interchangeably.

Overview
--------

By defining a common interface, the base class enables the strategy pattern. This pattern allows the fault injection system to work with different fault models without needing to know the details of how each strategy operates.

All strategies inherit from InjectionStrategy and implement the inject() method. The fault injection layer calls this method whenever it needs to apply faults to activation values.

Abstract Method
---------------

The inject() method is the only method that subclasses must implement.

Method Signature
^^^^^^^^^^^^^^^^^^^^^

The inject() method has the following signature:

.. code-block:: python

    @abstractmethod
    def inject(
        self,
        int_tensor: Tensor,
        mask: Tensor,
        bit_width: int,
        signed: bool,
        device: torch.device,
    ) -> Tensor:

This method receives all information needed to apply faults and must return a tensor with faults applied.

Parameters
^^^^^^^^^^^^

The parameters provide all necessary information for applying faults:

- **int_tensor**: The quantized activation values as integers. This is the tensor that should have faults applied to it.

- **mask**: A boolean tensor where True indicates positions to modify and False indicates positions to leave unchanged. The mask has the same shape as int_tensor.

- **bit_width**: The number of bits used for quantization. This determines the range of valid values that can be represented.

- **signed**: A boolean indicating whether the quantization uses signed or unsigned integers. This affects the valid range of values.

- **device**: The PyTorch device (CPU or CUDA) where new tensors should be created. This ensures that the strategy creates tensors on the correct device.

Return Value
^^^^^^^^^^^^^^

The method must return a new integer tensor with the same shape as int_tensor. At positions where mask is True, the returned tensor should contain faulty values. At positions where mask is False, the returned tensor should contain the original values from int_tensor.

It is important that the method returns a new tensor, not modifying the input tensor in-place. This ensures that the original quantized values are preserved for statistics recording.

Helper Method
--------------

The base class provides a helper method for computing the valid value range for a given quantization scheme.

Value Range Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^

The _get_value_range() method computes the minimum, maximum, and range size for a quantization scheme:

.. code-block:: python

    def _get_value_range(self, bit_width: int, signed: bool) -> tuple[int, int, int]:

This method handles both signed and unsigned quantization.

Signed Quantization
^^^^^^^^^^^^^^^^^^^^^

For signed quantization, the method computes:

.. code-block:: python

    min_val = -(2 ** (bit_width - 1)) + 1
    max_val = (2 ** (bit_width - 1)) - 1

For example, with 8-bit signed quantization:

- **min_val**: -127 (not -128, which is reserved)
- **max_val**: 127
- **range_size**: 255 (127 - (-127) + 1)

The total number of representable values is 2^bit_width - 2.

Unsigned Quantization
^^^^^^^^^^^^^^^^^^^^^^^

For unsigned quantization, the method computes:

.. code-block:: python

    min_val = 0
    max_val = 2 ** bit_width

For example, with 8-bit unsigned quantization:

- **min_val**: 0
- **max_val**: 255
- **range_size**: 256 (255 - 0 + 1)

The total number of representable values is 2^bit_width.

Range Size Computation
^^^^^^^^^^^^^^^^^^^^^^^^^

The range_size is computed as:

.. code-block:: python

    range_size = max_val - min_val + 1

This represents the total number of valid quantized values. Strategies can use this value for:

- **Modular arithmetic**: Wrapping values that exceed the range back into valid range
- **Random value generation**: Ensuring that random values are within the valid range
- **Value clamping**: Ensuring that computed faulty values are valid

Implementation Guidelines
-----------------------

When implementing a custom strategy, there are several important considerations to keep in mind.

Correctness
^^^^^^^^^^^^^^

The strategy must correctly handle the quantization parameters:

- **Bit width**: Operations must be appropriate for the specified number of bits
- **Signedness**: The strategy must respect whether values are signed or unsigned
- **Value range**: All output values must be within the valid quantization range

For example, a strategy that uses bit manipulation should ensure that bit operations do not produce values outside of the valid range.

Performance
^^^^^^^^^^^^^^^

The strategy should be efficient, as it is called for every forward pass:

- **Use vectorized operations**: Prefer PyTorch operations that work on entire tensors at once
- **Avoid loops**: Do not iterate over tensor elements in Python
- **Minimize tensor allocation**: Reuse tensors when possible

For example, using XOR operations and torch.where() is much faster than iterating over elements and flipping bits individually.

Numerical Stability
^^^^^^^^^^^^^^^^^^^^^^

The strategy should handle edge cases and unusual inputs gracefully:

- **Empty mask**: If mask has no True values, strategy should efficiently return the original tensor
- **All True mask**: If mask has all True values, strategy should still work correctly
- **Zero tensors**: If int_tensor contains all zeros, strategy should still produce correct faulty values

These edge cases should not cause errors or produce invalid outputs.

Example Implementations
-----------------------

To understand how to implement a custom strategy, it can be helpful to examine the existing implementations.

Bit Flip Strategy
^^^^^^^^^^^^^^^^^^^^

The bit flip strategies (LSBFlipStrategy, MSBFlipStrategy, FullFlipStrategy) use bitwise XOR to flip specific bits:

.. code-block:: python

    flipped = int_tensor ^ bit_mask
    result = torch.where(mask, flipped, int_tensor)

This pattern can be adapted to flip any combination of bits.

Random Value Strategy
^^^^^^^^^^^^^^^^^^^^^^^^

The RandomStrategy uses modular arithmetic to add random values while staying within the valid range:

.. code-block:: python

    rand_tensor = torch.randint(low=min_val, high=max_val + 1, ...)
    modular_result = ((int_tensor + rand_tensor - min_val) % range_size) + min_val
    result = torch.where(mask, modular_result, int_tensor)

This pattern can be adapted for any strategy that needs to ensure results stay within a specific range.

Mask Application
^^^^^^^^^^^^^^^^^^^^^

All strategies use the same pattern for applying the mask:

.. code-block:: python

    result = torch.where(mask, faulty_tensor, int_tensor)

The torch.where() function selects between two tensors based on the boolean mask, applying the fault only at specified positions.

API Reference
------------

.. autoclass:: utils.fault_injection.strategies.base.InjectionStrategy
    :members:
    :undoc-members:
    :show-inheritance:
