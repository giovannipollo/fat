.. _strategy_msb_flip_module:

MSBFlipStrategy Class
====================

The MSBFlipStrategy implements fault injection by flipping the most significant bit of selected activation values. This models a severe type of hardware error where a critical bit is corrupted.

Overview
--------

The most significant bit (MSB) is the bit at the highest position in a binary representation, which contributes the most to the value's magnitude. Flipping this bit can cause dramatic changes to the activation value.

This type of error is particularly significant because:

- **Large magnitude impact**: The MSB contributes half of the total value range
- **Catastrophic effect**: A single bit flip can completely change the sign or magnitude
- **Low frequency but high impact**: MSB errors are less common but have severe consequences when they occur
- **Catastrophic errors**: Can cause complete misclassification or numerical instability

Algorithm
-----------

The MSB flip strategy uses bitwise XOR with a bit mask that has only the MSB position set.

Bit Mask Creation
^^^^^^^^^^^^^^^^^^^^

The strategy creates a bitmask with the MSB set:

.. code-block:: python

    msb_mask = 1 << (bit_width - 1)

For example, with 8-bit quantization:
- **bit_width**: 8
- **msb_mask**: 1 << 7 = 128 (binary: 1000 0000)

This mask has only the highest bit set to 1, and all other bits are 0.

Bit Flipping Operation
^^^^^^^^^^^^^^^^^^^^^^^^^

The strategy applies XOR operation to flip the MSB:

.. code-block:: python

    flipped = int_tensor ^ msb_mask

The XOR operation has the following effect:

- **If MSB is 0**: XOR with mask makes it 1 (adds the MSB's value to the number)
- **If MSB is 1**: XOR with mask makes it 0 (removes the MSB's value from the number)

All other bits are XORed with 0, so they remain unchanged.

Mask Application
^^^^^^^^^^^^^^^^^^^^^

Finally, the strategy applies the mask to only flip bits at specified positions:

.. code-block:: python

    result = torch.where(mask, flipped, int_tensor)

The torch.where() function selects between the flipped values and the original values based on the boolean mask.

Error Characteristics
--------------------

The MSB flip strategy produces errors with specific characteristics that distinguish it from other fault types.

Magnitude of Error
^^^^^^^^^^^^^^^^^^^^^^

Flipping the most significant bit can cause a large change to the activation value. The exact change depends on whether the quantization is signed or unsigned.

For signed N-bit quantization:
- **Positive values**: MSB flips to 1, changing the value from positive to negative
- **Negative values**: MSB flips to 0, changing the value from negative to positive
- **Magnitude change**: The value changes by approximately half of the quantization range

For example, with 8-bit signed quantization:
- **Original value**: 100 (binary: 0110 0100)
- **After MSB flip**: -28 (binary: 1110 0100)
- **Change**: -128 (the value contributed by the MSB bit)

For unsigned N-bit quantization:
- **MSB contributes 2^(N-1)**: Flipping it adds or subtracts this value
- **Magnitude change**: The value changes by exactly half of the total range

For example, with 8-bit unsigned quantization:
- **Original value**: 100 (binary: 0110 0100)
- **After MSB flip**: 228 (binary: 1110 0100)
- **Change**: +128 (the value contributed by the MSB bit)

Effect on Value Sign
^^^^^^^^^^^^^^^^^^^^^^^^^

For signed quantization, the MSB determines whether a value is positive or negative. Flipping it therefore changes the sign of the value:

- **Positive becomes negative**: The value goes from +something to -something else
- **Negative becomes positive**: The value goes from -something to +something else

This sign flip is particularly problematic for neural networks because it can completely reverse the meaning of an activation.

Use Cases
---------

The MSB flip strategy is appropriate for several testing and evaluation scenarios.

Catastrophic Error Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because MSB errors cause such large changes, they are useful for testing the worst-case scenario. By evaluating model performance under MSB flip errors, you can understand:

- **Maximum possible degradation**: How much accuracy can degrade from a single bit error
- **Model sensitivity**: Which layers are most affected by catastrophic errors
- **Failure modes**: Under what conditions does the model completely fail

Fault Resilience Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When evaluating hardware for deployment, MSB flip testing helps establish requirements for error resilience. If a model cannot tolerate occasional MSB errors, it may not be suitable for deployment on certain hardware platforms where such errors might occur.

Fault-Aware Training
^^^^^^^^^^^^^^^^^^^^^^^^^^

When used for fault-aware training, the MSB flip strategy encourages the model to learn representations that are robust to catastrophic errors. By training with occasional MSB flips, the model learns to maintain performance even when some activations change sign or magnitude dramatically.

However, training with MSB flip errors can be challenging because:

- **Large gradient changes**: The sign flip can produce very different gradients
- **Training instability**: The model may take longer to converge or require careful hyperparameter tuning
- **Over-robustness**: The model may learn overly conservative representations that limit performance on clean data

Comparison to Other Strategies
-----------------------------

The MSB flip strategy is one of the most severe strategies available.

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

The MSB flip produces errors with magnitude second only to the full flip strategy. While the full flip always produces the maximum possible change (the bitwise complement), the MSB flip produces a change that is approximately half of the quantization range.

Realism
^^^^^^^^^^^^^^^^^^^^^

While MSB errors are severe, they are realistic in the sense that:

- **Physical defects**: Manufacturing defects or aging can cause certain bits to be less reliable
- **Power supply issues**: Unstable power can affect higher-order bits more than lower-order bits
- **Radiation effects**: High-energy particles can cause multiple bit flips, including MSB

However, MSB errors are less common than LSB errors in well-functioning memory systems.

Performance Considerations
--------------------------

The MSB flip strategy is highly efficient.

Computational Cost
^^^^^^^^^^^^^^^^^^^^^^

The strategy performs a minimal number of operations:

1. **Bit mask creation**: Computing 1 << (bit_width - 1)
2. **XOR operation**: One bitwise XOR per element
3. **Mask application**: One torch.where() to apply the mask

This makes MSB flip one of the fastest strategies, with performance essentially identical to LSB flip and full flip (which also use a single XOR operation).

Memory Overhead
^^^^^^^^^^^^^^^^^^^^^^

The strategy creates one additional tensor (the flipped tensor) with the same size as the input. The bit mask is just an integer, not a tensor, so it has negligible memory footprint.

Numerical Stability
-------------------

The MSB flip strategy is numerically stable and handles all inputs correctly.

Edge Cases
^^^^^^^^^^^^^^^^^^^^^^

The strategy handles various edge cases:

- **Zero values**: Flipping the MSB of 0 changes it to 2^(bit_width - 1)
- **Boundary values**: Values at the maximum or minimum of the range change dramatically
- **All same values**: If all input values are the same, the same operation is applied to all

For signed quantization, the sign flip for boundary values can cause the value to jump from the positive maximum to the negative minimum or vice versa, which is the correct behavior for an MSB flip.

API Reference
------------

.. autoclass:: utils.fault_injection.strategies.msb_flip.MSBFlipStrategy
    :members:
    :undoc-members:
    :show-inheritance:
