.. _strategy_full_flip_module:

FullFlipStrategy Class
=====================

The FullFlipStrategy implements fault injection by inverting all bits of selected activation values. This computes the bitwise complement, representing the maximum possible change for a given quantization value.

Overview
--------

Full bit flip (or bitwise NOT) inverts every bit in a value's binary representation. This represents the most severe type of single-bit error, where the entire value is corrupted.

This type of error is particularly relevant because:

- **Maximum possible change**: The resulting value is as far as possible from the original
- **Complete corruption**: Every bit is affected, not just one or a few
- **Catastrophic impact**: A single full flip can completely destroy the meaning of an activation
- **Cell failure**: Models a scenario where a memory cell completely fails to read or write correctly

Algorithm
-----------

The full flip strategy uses bitwise XOR with a mask that has all bits set to 1.

Bit Mask Creation
^^^^^^^^^^^^^^^^^^^^

The strategy creates a bitmask where all bits up to bit_width are set to 1:

.. code-block:: python

    all_ones = (1 << bit_width) - 1

For example, with 8-bit quantization:
- **bit_width**: 8
- **all_ones**: (1 << 8) - 1 = 255 (binary: 1111 1111)

This mask has all bits in the quantization range set to 1.

Bit Inversion Operation
^^^^^^^^^^^^^^^^^^^^^^^^^

The strategy applies XOR operation to invert all bits:

.. code-block:: python

    flipped = int_tensor ^ all_ones

The XOR operation with all 1's inverts every bit:

- **If a bit is 0**: XOR with 1 makes it 1
- **If a bit is 1**: XOR with 1 makes it 0

This results in the bitwise complement of the original value.

Mask Application
^^^^^^^^^^^^^^^^^^^^^

Finally, the strategy applies the mask to only flip bits at specified positions:

.. code-block:: python

    result = torch.where(mask, flipped, int_tensor)

The torch.where() function selects between the flipped values and the original values based on the boolean mask.

Error Characteristics
--------------------

The full flip strategy produces errors with specific characteristics that make it the most severe of the available strategies.

Maximum Magnitude Change
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Full bit flip always produces the maximum possible change for the quantization range:

For signed N-bit quantization:
- **From minimum (-(2^(N-1)) + 1)**: Flips to maximum ((2^(N-1)) - 1)
- **From maximum ((2^(N-1)) - 1)**: Flips to minimum (-(2^(N-1)) + 1)

For unsigned N-bit quantization:
- **From minimum (0)**: Flips to maximum (2^N)
- **From maximum (2^N)**: Flips to minimum (0)

For example, with 8-bit signed quantization:
- **Original value**: 100 (binary: 0110 0100)
- **After full flip**: -29 (binary: 1110 0101)
- **Change**: -129

This is the maximum possible distance between two valid 8-bit signed values.

Complete Value Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike bit-flip strategies that only affect one bit, full flip affects every bit:

- **All bits inverted**: No information from the original value is preserved
- **Independent of original value**: The resulting value does not depend on the original, only on the bit width

This means that full flip errors completely destroy the information content of the activation value.

Use Cases
---------

The full flip strategy is appropriate for several testing and evaluation scenarios.

Worst-Case Analysis
^^^^^^^^^^^^^^^^^^^^^^^^

Because full flip produces the maximum possible change, it is useful for understanding the worst-case scenario. By evaluating model performance under full flip errors, you can determine:

- **Lower bound on performance**: The minimum accuracy the model can achieve under any single error
- **Catastrophic failure threshold**: At what error rate does the model completely fail
- **Resilience limits**: Whether the model can tolerate any full flip errors

Robustness Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When evaluating hardware for deployment, full flip testing helps establish the most stringent requirements for error resilience. If a model cannot tolerate any full flip errors, it may not be suitable for deployment in environments where such errors are possible.

Fault-Aware Training
^^^^^^^^^^^^^^^^^^^^^^^^^^

When used for fault-aware training, the full flip strategy encourages the model to learn representations that are robust to catastrophic errors. By training with occasional full flip errors, the model learns to maintain performance even when some activations are completely corrupted.

However, training with full flip errors can be very challenging because:

- **Complete information loss**: The model must learn to function despite some activations containing no useful information
- **Training instability**: The large magnitude changes can cause very different gradients, making training unstable
- **Over-conservative learning**: The model may learn overly robust representations that severely limit performance on clean data

Comparison to Other Strategies
-----------------------------

The full flip strategy is the most severe of all available strategies.

Error Magnitude Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

    flowchart TD
        LSB[LSB Flip<br/>±1 change]
        MSB[MSB Flip<br/>Large change]
        Random[Random<br/>Variable change]
        Full[Full Flip<br/>Maximum change]
        
        subgraph "Severity Level"
            LSB --> L1[Low]
            Random --> R1[Medium]
            MSB --> M1[High]
            Full --> F1[Very High]
        end
        
        style LSB fill:#ccffcc
        style MSB fill:#ffccaa
        style Full fill:#ff9999

The full flip always produces the maximum possible change for the quantization range. No other strategy can produce a larger change from a single injection event.

Information Loss
^^^^^^^^^^^^^^^^^^^^^

Different strategies preserve different amounts of information from the original value:

- **LSB flip**: Preserves almost all information, only the lowest-order bit changes
- **MSB flip**: Preserves most bits, only the highest-order bit changes
- **Random**: May preserve some bits if the random value happens to be similar
- **Full flip**: Preserves no information; every bit is inverted

This complete information loss is what makes full flip errors particularly catastrophic.

Realism
^^^^^^^^^^^^^^

While full flip errors are severe, they are realistic in certain scenarios:

- **Complete cell failure**: An entire memory cell may completely fail
- **Power surge**: A power event could corrupt all bits in a cell
- **Extreme radiation**: High-energy particle interactions can cause multiple simultaneous bit flips

However, full flip errors are the least common of all error types in well-functioning hardware.

Performance Considerations
--------------------------

The full flip strategy is highly efficient.

Computational Cost
^^^^^^^^^^^^^^^^^^^^^

The strategy performs a minimal number of operations:

1. **Bit mask creation**: Computing (1 << bit_width) - 1
2. **XOR operation**: One bitwise XOR per element
3. **Mask application**: One torch.where() to apply the mask

This makes full flip one of the fastest strategies, with performance essentially identical to LSB flip and MSB flip (which also use a single XOR operation).

Memory Overhead
^^^^^^^^^^^^^^^^^^^^^

The strategy creates one additional tensor (the flipped tensor) with the same size as the input. The bit mask is just an integer, not a tensor, so it has negligible memory footprint.

Numerical Stability
-------------------

The full flip strategy is numerically stable and handles all inputs correctly.

Edge Cases
^^^^^^^^^^^^^^^^^^^^^

The strategy handles various edge cases:

- **Zero values**: Flipping all bits of 0 produces the maximum value for the quantization range
- **All same values**: If all input values are the same, the same operation is applied to all
- **Boundary values**: Values at the minimum or maximum flip to the opposite boundary
- **Negative values**: For signed quantization, the bit inversion works identically for negative values

The XOR operation is well-defined for all integers, so the strategy does not encounter undefined or exceptional cases.

API Reference
------------

.. autoclass:: utils.fault_injection.strategies.full_flip.FullFlipStrategy
    :members:
    :undoc-members:
    :show-inheritance:
