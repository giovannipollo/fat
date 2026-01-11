.. _strategy_lsb_flip_module:

LSBFlipStrategy Class
====================

The LSBFlipStrategy implements fault injection by flipping the least significant bit of selected activation values. This models a common type of memory error where a single bit at the lowest significance position is corrupted.

Overview
--------

The least significant bit (LSB) is the bit that represents the value 1 in a binary number. Flipping this bit changes the value by exactly ±1, toggling between even and odd values.

This type of error is particularly relevant because:

- **Common in memory systems**: LSB errors occur frequently in DRAM and SRAM due to their physical implementation
- **Minimal impact**: The ±1 change typically has small effect on neural network output
- **Representative**: Models the most common type of bit-level hardware error

Algorithm
-----------

The LSB flip strategy uses bitwise XOR to flip the least significant bit.

Bit Mask Creation
^^^^^^^^^^^^^^^^^^^^^

The strategy creates a bitmask that has only the least significant bit set:

.. code-block:: python

    bit_mask = 1

This mask represents the binary value 0000...0001, where only the LSB is set to 1. When XORed with another value, this mask will flip the LSB and leave all other bits unchanged.

Bit Flipping Operation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The strategy applies the XOR operation to flip the LSB:

.. code-block:: python

    flipped = int_tensor ^ bit_mask

The XOR operation has the following effect:

- **If LSB is 0**: XOR with 1 makes it 1 (value increases by 1)
- **If LSB is 1**: XOR with 1 makes it 0 (value decreases by 1)

All other bits are XORed with 0, so they remain unchanged.

Mask Application
^^^^^^^^^^^^^^^^^^^^^

Finally, the strategy applies the mask to only flip bits at specified positions:

.. code-block:: python

    result = torch.where(mask, flipped, int_tensor)

The torch.where() function selects between the flipped values and the original values based on the boolean mask.

Error Characteristics
--------------------

The LSB flip strategy produces errors with specific characteristics.

Magnitude of Error
^^^^^^^^^^^^^^^^^^^^^^

Flipping the least significant bit always changes a value by exactly ±1:

.. code-block:: python

    # If original value is even (LSB = 0):
    # Flipping LSB makes it odd (value + 1)

    # If original value is odd (LSB = 1):
    # Flipping LSB makes it even (value - 1)

This predictability makes LSB flip errors relatively mild compared to other fault types.

Effect on Different Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The effect of flipping the LSB depends on the original value:

- **For even values**: LSB is 0, flipping it to 1 increases the value by 1
- **For odd values**: LSB is 1, flipping it to 0 decreases the value by 1

This means that the error is symmetric in the sense that values alternate between even and odd.

Influence on Higher Bits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The LSB flip has no effect on higher-order bits. For example, consider the 8-bit value 0010 1100 (44 in decimal):

- **Original**: 0010 1100 (44)
- **After LSB flip**: 0010 1101 (45)

Only the LSB changed, and the value increased from 44 to 45. All other bits remained the same.

This limited effect is what makes LSB errors less severe than MSB or full flip errors.

Use Cases
---------

The LSB flip strategy is appropriate for several testing and training scenarios.

Modeling Common Memory Errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In real memory systems, the least significant bits are most vulnerable to errors due to:

- **Physical layout**: Memory cells for LSBs may have different physical characteristics
- **Capacitance**: Lower bits often have smaller capacitance, making them more susceptible to noise
- **Power delivery**: Power delivery to LSB cells may be less stable

When testing robustness to realistic hardware errors, LSB flip is an appropriate strategy because it models the most common type of memory error.

Baseline Robustness Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because LSB errors are common but have minimal impact, testing with LSB flip provides a baseline for model robustness. If a model is not robust to LSB errors, it is likely to be very sensitive to more severe errors.

Fault-Aware Training
^^^^^^^^^^^^^^^^^^^^^^^^^^

When used for fault-aware training, the LSB flip strategy encourages the model to learn representations that are robust to minor bit errors. Because the gradient zeroing mechanism prevents the model from learning to compensate for specific injected errors, the model instead learns to produce activations that maintain performance even when some values are off by ±1.

Comparison to Other Strategies
-----------------------------

The LSB flip strategy is the mildest of the available strategies.

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
            MSB --> M1[High]
            Random --> R1[Medium]
            Full --> F1[Very High]
        end
        
        style LSB fill:#ccffcc
        style MSB fill:#ffccaa
        style Full fill:#ff9999

The LSB flip produces the smallest magnitude changes of any strategy. The MSB flip and full flip strategies can produce changes that are many times larger.

Realism
^^^^^^^^^^^^^^^^^^^^^

While LSB errors are common, the other strategies also model real hardware phenomena:

- **MSB flip**: Less common but can occur due to power supply issues or manufacturing defects
- **Random**: Can model stuck bits, communication errors, or radiation effects
- **Full flip**: Can model complete cell failure or extreme radiation events

When testing for deployment on specific hardware, research the error characteristics of that hardware to choose the most realistic strategies.

Performance Characteristics
--------------------------

The LSB flip strategy is highly efficient.

Computational Cost
^^^^^^^^^^^^^^^^^^^^^^

The strategy performs a minimal number of operations:

1. **Bit mask creation**: Creating the value 1 (trivial)
2. **XOR operation**: One bitwise XOR per element
3. **Mask application**: One torch.where() to apply the mask

This makes LSB flip one of the fastest strategies, with performance essentially identical to MSB flip and full flip (which also use a single XOR operation).

Memory Overhead
^^^^^^^^^^^^^^^^^^^^^^

The strategy creates one additional tensor (the flipped tensor) with the same size as the input. The bit_mask is just an integer, not a tensor, so it has negligible memory footprint.

Numerical Stability
-------------------

The LSB flip strategy is numerically stable and handles all inputs correctly.

Edge Cases
^^^^^^^^^^^^^^^^^^^^^

The strategy handles various edge cases:

- **Zero values**: Flipping the LSB of 0 changes it to 1
- **All same values**: If all input values are the same, the same operation is applied to all
- **Negative values**: For signed quantization, flipping the LSB works identically for negative values

The XOR operation is well-defined for all integers, so the strategy does not encounter undefined or exceptional cases.

API Reference
------------

.. autoclass:: utils.fault_injection.strategies.lsb_flip.LSBFlipStrategy
    :members:
    :undoc-members:
    :show-inheritance:
