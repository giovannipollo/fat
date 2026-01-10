.. _strategy_random_module:

RandomStrategy Class
===================

The RandomStrategy implements fault injection by replacing selected activation values with random integers within the valid quantization range. This models the scenario where complete value corruption occurs, such as a stuck bit or communication error.

Overview
--------

Random value injection is a common fault model for testing neural network robustness. It represents the situation where an activation value is completely replaced with an incorrect value, rather than having just one or a few bits flipped.

This type of error can occur in real hardware due to:

- **Stuck bits**: A memory cell where one or more bits are permanently stuck at 0 or 1
- **Communication errors**: Data corruption during transfer between memory and processing unit
- **Radiation-induced errors**: Alpha particles or cosmic rays causing random bit flips that result in completely different values
- **Complete cell failure**: An entire memory cell failing to read or write correct value

Algorithm
-----------

The random strategy uses modular arithmetic to ensure that faulty values stay within the valid quantization range.

Value Range Determination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, the strategy determines the valid range of values for the quantization parameters:

.. code-block:: python

    min_val, max_val, range_size = self._get_value_range(bit_width, signed)

For signed 8-bit quantization:
- **min_val**: -127
- **max_val**: 127
- **range_size**: 255

For unsigned 8-bit quantization:
- **min_val**: 0
- **max_val**: 255
- **range_size**: 256

Random Value Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The strategy generates a random integer for each element in the tensor:

.. code-block:: python

    rand_tensor = torch.randint(
        low=min_val,
        high=max_val + 1,
        size=int_tensor.shape,
        device=device,
        dtype=torch.int32,
    )

This creates a tensor of the same shape as the input, where each element is a random integer within the valid range. The use of torch.randint() ensures that:

- **Values are uniformly distributed**: Every value in the range has equal probability
- **Efficient generation**: The random values are generated in parallel for all elements
- **Correct device placement**: The random tensor is created on the same device as the input

Modular Arithmetic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The strategy then adds the random value to the original value and uses modular arithmetic to ensure the result stays within the valid range:

.. code-block:: python

    modular_result = ((int_tensor + rand_tensor - min_val) % range_size) + min_val

This operation works as follows:

1. **Subtract min_val**: Shift both tensors so that the minimum value becomes 0
2. **Add**: Add the random value to the original value
3. **Modulo**: Apply modulo by range_size to wrap around the range
4. **Add min_val**: Shift the result back to the original range

The modular arithmetic ensures that even if the addition exceeds the maximum value or goes below the minimum value, the result wraps around to stay within the valid range.

Mask Application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, the strategy applies the mask to only replace values at specified positions:

.. code-block:: python

    result = torch.where(mask, modular_result, int_tensor)

The torch.where() function selects between the faulty values and the original values based on the mask. At positions where the mask is True, the faulty values are used. At positions where the mask is False, the original values are preserved.

Characteristics
-------------

The random strategy has several characteristics that make it useful for different testing scenarios.

Error Magnitude
^^^^^^^^^^^^^^^^^^^^^^

The magnitude of errors produced by the random strategy is variable and depends on:

- **The original value**: Different values will have different random offsets
- **The random value generated**: This is uniformly distributed across the entire range
- **The quantization range**: Wider ranges (e.g., 16-bit vs 8-bit) allow for larger errors

In some cases, the error may be small (if the random value happens to be close to the original), while in other cases it may be large (if the random value is very different from the original). This variability makes the random strategy useful for testing general robustness to arbitrary value errors.

Uniform Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because torch.randint() uses a uniform distribution, every value in the valid range has equal probability of being selected as the faulty value. This means:

- **No bias toward specific values**: The strategy does not preferentially create certain types of errors
- **Predictable distribution**: The statistical properties of the injected values are known
- **Comprehensive testing**: Over many injections, the full range of possible errors will be explored

For testing purposes, the uniform distribution ensures that we are not inadvertently testing only a subset of possible error values.

Range Compliance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A key property of the random strategy is that it always produces values within the valid quantization range. This is achieved through the modular arithmetic step:

- **Wrapping**: Values that exceed the maximum wrap around to the minimum
- **Preserving signedness**: The strategy respects whether the quantization is signed or unsigned

This ensures that the injected values are always valid quantized values that the model would normally produce.

Use Cases
---------

The random strategy is appropriate for several testing and training scenarios.

General Robustness Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because the random strategy can produce any value in the quantization range, it is useful for testing whether the model is robust to arbitrary value errors. This is particularly relevant when:

- **The exact error distribution is unknown**: We want to test general resilience rather than a specific error type
- **Multiple error sources are possible**: Different types of hardware errors might produce different value changes
- **Worst-case analysis**: We want to understand how the model performs when any value could be corrupted

Fault-Aware Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When used for fault-aware training, the random strategy can help the model learn to be robust to general value corruption. The gradient zeroing mechanism ensures that the model does not learn to compensate for specific injected values, but rather learns to produce activations that maintain performance even when some values are completely replaced.

Comparison to Other Strategies
-----------------------------

The random strategy differs from bit-flip strategies in several important ways.

Versus LSB Flip
^^^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

    flowchart TD
        Random[Random Strategy]
        LSB[LSB Flip]
        
        subgraph "Random Strategy"
            R1[Any value in range]
            R2[Variable magnitude]
            R3[±1 to ±range]
        end
        
        subgraph "LSB Flip"
            L1[Flip bit 0 only]
            L2[Small magnitude]
            L3[Always ±1]
        end

The LSB flip strategy always changes a value by exactly ±1, flipping the least significant bit. In contrast, the random strategy can change a value by any amount within the quantization range. This means:

- **LSB flip**: Predictable, small error magnitude
- **Random**: Unpredictable, variable error magnitude

For testing minor bit errors that commonly occur in memory, LSB flip is more representative. For testing general value corruption, the random strategy is more appropriate.

Versus MSB Flip
^^^^^^^^^^^^^^^^^^^^^

The MSB flip strategy flips the most significant bit, which can cause a large magnitude change. The random strategy can also produce large magnitude changes, but:

- **MSB flip**: Always produces a large, specific change (approximately half the range)
- **Random**: May produce large or small changes depending on the random value

For testing the impact of catastrophic bit errors that dramatically change a value, the MSB flip strategy is more representative. For testing general resilience to value errors, the random strategy is more appropriate.

Versus Full Flip
^^^^^^^^^^^^^^^^^^^^^

The full flip strategy inverts all bits, producing the maximum possible change for the quantization range. The random strategy can produce a value that is the maximum possible distance from the original, but:

- **Full flip**: Always produces the bitwise complement, a specific type of maximum change
- **Random**: Can produce the maximum change, but may also produce smaller changes

For testing the impact of complete bit inversion errors, the full flip strategy is more representative.

Performance Considerations
--------------------------

The random strategy has slightly higher computational cost than the bit-flip strategies.

Computational Cost
^^^^^^^^^^^^^^^^^^^^^^

The random strategy performs the following operations:

1. **Range calculation**: Computing the value range (minor cost)
2. **Random value generation**: Creating a tensor of random integers
3. **Modular arithmetic**: Addition, subtraction, and modulo operations
4. **Mask application**: torch.where() to apply the mask

Compared to bit-flip strategies, the random strategy has the additional cost of random value generation and modular arithmetic. However, this cost is:

- **Vectorized**: All operations are applied to the entire tensor at once
- **Negligible for most models**: The cost is small compared to the actual layer computation
- **Only when faults are injected**: The cost is not incurred when no faults are to be injected

Numerical Stability
-------------------

The random strategy is numerically stable and handles edge cases correctly.

Edge Cases
^^^^^^^^^^^^^^^^^^^^^^

The strategy handles various edge cases:

- **Zero probability**: When no faults are to be injected, the mask is all False, and torch.where() returns the original tensor
- **All True mask**: When all elements are to be modified, all elements receive random values
- **Mixed types**: The strategy works correctly with different integer and floating-point types

The modular arithmetic ensures that results are always valid integers, and the mask application ensures that only the specified positions are modified.

API Reference
------------

.. autoclass:: utils.fault_injection.strategies.random.RandomStrategy
    :members:
    :undoc-members:
    :show-inheritance:
