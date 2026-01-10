.. _fault_strategies_overview:

Fault Injection Strategies
========================

The fault injection system uses a strategy pattern to implement different fault models. Each strategy represents a different type of hardware error that might occur in quantized neural networks.

Overview
--------

A fault injection strategy is responsible for determining exactly how to modify activation values when faults are injected. The strategy receives an integer tensor of quantized activation values, a mask indicating which values to modify, and quantization parameters (bit width and signedness). It returns a new integer tensor with faults applied at the specified positions.

The strategy pattern provides several advantages:

- **Extensibility**: New fault types can be added by creating new strategy classes
- **Modularity**: Different fault models are isolated from each other
- **Interchangeability**: Strategies can be swapped at runtime without changing other components
- **Testability**: Each strategy can be tested independently

Strategy Selection
-----------------

The strategy to use is specified through the configuration. FaultInjectionConfig has an injection_type parameter that accepts the following values:

- **random**: RandomStrategy replaces values with random integers within the valid quantization range
- **lsb_flip**: LSBFlipStrategy flips the least significant bit
- **msb_flip**: MSBFlipStrategy flips the most significant bit
- **full_flip**: FullFlipStrategy inverts all bits (bitwise NOT)

Each strategy models a different type of hardware error:

.. mermaid::

    flowchart TD
        Random[Random Value<br/>Replacement]
        LSB[LSB Flip<br/>Bit 0]
        MSB[MSB Flip<br/>Highest Bit]
        Full[Full Flip<br/>All Bits]
        
        subgraph "Fault Severity"
            Random --> M1[Medium]
            LSB --> M2[Low]
            MSB --> M3[High]
            Full --> M4[Very High]
        end
        
        subgraph "Error Magnitude"
            Random --> E1[Variable]
            LSB --> E2[Change of 1]
            MSB --> E3[Large Change]
            Full --> E4[Maximum Change]
        end
        
        style LSB fill:#ccffcc
        style MSB fill:#ffccaa
        style Full fill:#ff9999

The strategies differ in both the severity of errors they produce and the frequency with which such errors might occur in real hardware:

- **LSB flip**: Common in memory systems, but typically has minimal impact
- **MSB flip**: Less common but can have catastrophic impact on model accuracy
- **Random value**: Models complete value corruption, which might occur due to stuck bits or communication errors
- **Full flip**: Represents extreme corruption, such as complete bit inversion

Choosing a Strategy
--------------------

The appropriate strategy depends on the hardware characteristics you want to model and the research questions you're addressing.

Modeling Real Hardware
^^^^^^^^^^^^^^^^^^^^^^^^^

If you're trying to model a specific type of hardware error, choose the strategy that best represents that error type:

- **Memory bit errors**: LSB flip is most common type of error in DRAM and SRAM
- **Stuck bits**: Random value replacement can model bits that are permanently stuck at 0 or 1
- **Complete cell failure**: Full flip can model complete corruption of a memory cell

Research Use Cases
^^^^^^^^^^^^^^^^^^^^^^

Different strategies are useful for different research questions:

- **Robustness testing**: Use LSB flip to test if model is robust to common, minor errors
- **Worst-case analysis**: Use MSB flip or full flip to understand maximum possible degradation
- **General robustness**: Use random value injection to test resilience to arbitrary errors
- **Comparative analysis**: Test with all strategies to compare model's sensitivity to different error types

Training vs. Evaluation
-------------------------

Strategies work identically during training and evaluation. The difference in fault behavior between training and evaluation is controlled by the configuration's apply_during parameter, not by the strategy implementation.

This means that the same fault model is applied consistently, whether faults are being used for fault-aware training or for evaluating a trained model's resilience.

Strategy Registry
-----------------

All strategies are registered in a central registry that enables factory-style instantiation:

.. code-block:: python

    _STRATEGIES: Dict[str, Type[InjectionStrategy]] = {
        "random": RandomStrategy,
        "lsb_flip": LSBFlipStrategy,
        "msb_flip": MSBFlipStrategy,
        "full_flip": FullFlipStrategy,
    }

The get_strategy() factory function retrieves a strategy by name:

.. code-block:: python

    strategy = get_strategy("lsb_flip")

This factory pattern provides several benefits:

- **String-based selection**: Strategies can be selected from configuration files
- **Type safety**: The factory returns an instance of InjectionStrategy base class
- **Error handling**: Invalid strategy names produce a clear error message

Implementing Custom Strategies
-----------------------------

You can extend the system with new fault models by implementing a custom strategy.

Implementation Steps
^^^^^^^^^^^^^^^^^^^^^^

To implement a new strategy:

1. Create a new class that inherits from InjectionStrategy
2. Implement the inject() method with the required signature
3. Register the strategy in the _STRATEGIES dictionary
4. Update the configuration validation to accept the new strategy name

The inject() method must have the following signature:

.. code-block:: python

    def inject(
        self,
        int_tensor: Tensor,
        mask: Tensor,
        bit_width: int,
        signed: bool,
        device: torch.device,
    ) -> Tensor:

The parameters are:

- **int_tensor**: The quantized activation values as integers
- **mask**: A boolean tensor where True indicates positions to modify
- **bit_width**: The number of bits used for quantization
- **signed**: Whether the quantization uses signed or unsigned integers
- **device**: The device (CPU or CUDA) to create new tensors on

The method must return:

- A new integer tensor with the same shape as int_tensor, with faults applied at positions where mask is True

Example: Bit-Flip Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's a simplified example of implementing a bit-flip strategy:

.. code-block:: python

    class MyBitFlipStrategy(InjectionStrategy):
        """Flip a specific bit."""

        def inject(
                self,
                int_tensor: Tensor,
                mask: Tensor,
                bit_width: int,
                signed: bool,
                device: torch.device,
        ) -> Tensor:
            # Create bitmask for the bit to flip
            bit_position = 2  # Flip bit 2
            bit_mask = 1 << bit_position

            # XOR with the bitmask to flip the bit
            flipped = int_tensor ^ bit_mask

            # Apply mask: only flip where mask is True
            result = torch.where(mask, flipped, int_tensor)

            return result

This strategy would flip bit 2 (the third least significant bit) in all selected activation values.

Value Range Handling
-------------------

Different quantization schemes use different value ranges. Strategies must handle these ranges correctly to produce valid quantized values.

Signed Quantization
^^^^^^^^^^^^^^^^^^^^^^

For signed N-bit quantization, the valid range is:

- **Minimum**: -(2^(N-1)) + 1
- **Maximum**: (2^(N-1)) - 1

For example, with 8-bit signed quantization:

- **Minimum**: -127
- **Maximum**: 127

The total number of representable values is 2^N - 2 (excluding -128 which is reserved for special purposes in many quantization schemes).

Unsigned Quantization
^^^^^^^^^^^^^^^^^^^^^^^^

For unsigned N-bit quantization, the valid range is:

- **Minimum**: 0
- **Maximum**: 2^N

For example, with 8-bit unsigned quantization:

- **Minimum**: 0
- **Maximum**: 255

The total number of representable values is 2^N.

Helper Method
^^^^^^^^^^^^^^^^^^^^^^^

The InjectionStrategy base class provides a helper method for computing the valid range:

.. code-block:: python

    min_val, max_val, range_size = self._get_value_range(bit_width, signed)

This method returns:

- **min_val**: The minimum valid value
- **max_val**: The maximum valid value
- **range_size**: The total number of valid values (max_val - min_val + 1)

Strategies can use these values to ensure that any computed faulty values are within the valid quantization range.

For example, the RandomStrategy uses the range_size to implement modular arithmetic:

.. code-block:: python

    modular_result = ((int_tensor + rand_tensor - min_val) % range_size) + min_val

This ensures that random additions always produce values within the valid range by wrapping around when the sum exceeds the maximum.

Comparing Strategies
--------------------

Different strategies have different characteristics that make them suitable for different purposes.

Error Severity
^^^^^^^^^^^^^^^^^^^^^

Strategies vary in how severely they affect activation values:

- **LSB flip**: Changes value by ±1, minimal severity
- **MSB flip**: Changes value by approximately half the quantization range, high severity
- **Random**: Variable severity, depends on the random value added
- **Full flip**: Maximum possible change for the quantization range, very high severity

The severity of errors you want to test should guide strategy selection. For testing general robustness, you might start with low-severity errors (LSB flip) and progress to higher-severity errors (MSB flip, full flip).

Probability of Real Occurrence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In real hardware, different error types occur with different probabilities:

- **LSB errors**: More common in memory systems
- **MSB errors**: Less common but more severe
- **Random value corruption**: Variable frequency depending on the hardware
- **Full bit inversion**: Rare, typically indicates a complete cell failure

When evaluating model resilience for a specific hardware target, research the error characteristics of that hardware and choose strategies that match.

Performance Impact
-------------------

Different strategies have different computational costs.

Bit Flip Operations
^^^^^^^^^^^^^^^^^^^^^^

Bit flip strategies (LSB, MSB, full) are computationally efficient:

.. code-block:: python

    flipped = int_tensor ^ bit_mask
    result = torch.where(mask, flipped, int_tensor)

These operations require:

- **One XOR operation**: A single bitwise operation per element
- **One torch.where()**: To apply the mask

This is extremely efficient and runs quickly on both CPU and GPU.

Random Value Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The random value strategy is slightly more expensive:

.. code-block:: python

    rand_tensor = torch.randint(low=min_val, high=max_val + 1, ...)
    modular_result = ((int_tensor + rand_tensor - min_val) % range_size) + min_val
    result = torch.where(mask, modular_result, int_tensor)

This requires:

- **Random number generation**: Creating a tensor of random integers
- **Modular arithmetic**: Addition, subtraction, and modulo operations
- **One torch.where()**: To apply the mask

The additional operations for random number generation and modular arithmetic make this strategy slightly slower than bit flip operations, but the difference is negligible for most models.

Strategy Switching
-------------------

Strategies can be switched at runtime by updating the configuration.

Dynamic Strategy Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During training or evaluation, you can change the injection strategy:

.. code-block:: python

    # Update configuration
    config.injection_type = "msb_flip"

    # Re-inject to update strategy
    model = injector.remove(model)
    model = injector.inject(model, config)

This allows for:

- **Curriculum learning**: Starting with mild errors and progressing to severe errors
- **Multi-strategy evaluation**: Testing model with multiple fault types
- **Phase-based strategies**: Using different strategies during different training phases

Note that updating the strategy requires removing and re-injecting the fault injection layers, because the strategy object is created during the inject() call and then shared across all layers.

Next Steps
----------

For detailed information about specific strategies, see:

- :doc:`base` - Base class for implementing custom strategies
- :doc:`random` - Random value replacement strategy
- :doc:`lsb_flip` - Least significant bit flipping strategy
- :doc:`msb_flip` - Most significant bit flipping strategy
- :doc:`full_flip` - Full bit inversion strategy
