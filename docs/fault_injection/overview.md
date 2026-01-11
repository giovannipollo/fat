.. _fault_injection_overview:

Fault Injection System Overview
=============================

This document provides a high-level overview of the fault injection system for quantized neural networks. The fault injection framework enables both fault-aware training (FAT) and fault resilience evaluation by simulating hardware errors in neural network activations.

Introduction to Fault Injection
-------------------------------

Fault injection is a technique used to simulate hardware errors that may occur in deployed neural networks, particularly in quantized models running on specialized hardware. By introducing controlled errors during training and evaluation, we can:

- **Train robust models** that are resilient to hardware faults
- **Evaluate model resilience** to different types of errors
- **Identify vulnerable layers** in neural networks
- **Quantify performance degradation** under fault conditions

In quantized neural networks, activations are represented using a limited number of bits, making them particularly susceptible to bit-flip errors that can significantly impact model accuracy.

System Architecture
-------------------

The fault injection system consists of several interconnected components that work together to inject faults into model activations during the forward pass while maintaining proper gradient flow during backpropagation.

Architecture Diagram
^^^^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

    flowchart TB
        subgraph "Configuration Layer"
            Config[FaultInjectionConfig]
        end

        subgraph "Model Transformation"
            Injector[FaultInjector]
            Wrapper[_InjectionWrapper]
        end

        subgraph "Runtime Execution"
            Layer[QuantFaultInjectionLayer]
            Function[FaultInjectionFunction]
        end

        subgraph "Strategies"
            Base[InjectionStrategy]
            Random[RandomStrategy]
            LSB[LSBFlipStrategy]
            MSB[MSBFlipStrategy]
            Full[FullFlipStrategy]
        end

        subgraph "Statistics"
            Stats[FaultStatistics]
            LayerStats[LayerStatistics]
        end

        Config --> Injector
        Injector --> Wrapper
        Wrapper --> Layer
        Layer --> Function

        Base --> Random
        Base --> LSB
        Base --> MSB
        Base --> Full

        Config --> Base
        Layer --> Stats
        Stats --> LayerStats

        style Config fill:#e1f5ff
        style Stats fill:#fff4e6
        style Base fill:#f0f0f0

Component Responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Configuration Layer**

The configuration system defines how faults are injected through parameters such as probability, injection type, and phase selection (training vs. evaluation). This provides a single source of truth for fault injection behavior across the entire system.

**Model Transformation**

The FaultInjector walks the model graph and inserts fault injection layers at strategic positions. It wraps existing quantized layers with injection wrappers without modifying the original model definition, allowing for easy enable/disable of fault injection.

**Runtime Execution**

During the forward pass, QuantFaultInjectionLayer intercepts activations, determines where to inject faults based on the probability, applies the injection strategy, and ensures proper gradient flow during backpropagation through FaultInjectionFunction.

**Injection Strategies**

Different fault types are modeled through a strategy pattern. Each strategy implements a different way of modifying activation values, such as random value replacement or bit flipping at specific positions.

**Statistics Tracking**

The statistics system records metrics about injection behavior, including the actual injection rate, error magnitudes (RMSE), and output similarity (cosine similarity), enabling detailed analysis of fault impact.

Data Flow
-----------

Understanding how data flows through the fault injection system is essential for understanding its behavior and debugging issues.

Forward Pass Flow
^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

    flowchart LR
        Input[Input Tensor] --> Q[Quantized Layer]
        Q --> QTensor[QuantTensor]
        QTensor --> Wrap[_InjectionWrapper]
        Wrap --> Layer[QuantFaultInjectionLayer]
        
        subgraph "Layer Processing"
            Layer --> Extract[Extract Quant Parameters]
            Extract --> MaskGen[Generate Fault Mask]
            MaskGen --> Apply[Apply Strategy]
            Apply --> Grad[FaultInjectionFunction]
        end
        
        Grad --> Output[Output QuantTensor]
        
        style Q fill:#ffe6e6
        style Layer fill:#e6f3ff
        style Output fill:#e6ffe6

The forward pass begins when a QuantTensor from a quantized layer enters the wrapper. The injection layer extracts quantization parameters, determines which activations should be modified based on the configured probability, applies the selected injection strategy, and returns the modified QuantTensor.

Backward Pass Flow
^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

    flowchart LR
        GradLoss[Gradient from Loss] --> FI[FaultInjectionFunction.backward]
        FI --> Check[Check Fault Mask]
        Check -->|At Faulty Positions| Zero[Zero Gradient]
        Check -->|At Clean Positions| Pass[Pass Gradient]
        Zero --> GradLayer[Gradient for Layer]
        Pass --> GradLayer
        GradLayer --> Update[Weight Update]
        
        style Zero fill:#ffcccc
        style Pass fill:#ccffcc
        
        classDef fault fill:#ffcccc
        classDef clean fill:#ccffcc

During backpropagation, gradients at positions where faults were injected are zeroed. This prevents the model from learning to predict the faulty values, instead training it to be robust to such errors. This is the key mechanism that enables fault-aware training.

Fault Types and Strategies
--------------------------

The system supports several fault models, each representing different types of hardware errors that may occur in quantized neural networks.

Random Value Injection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Random value injection replaces selected activation values with random integers within the valid quantization range. This models complete value corruption and is useful for testing the worst-case scenario where an entire activation is corrupted.

Least Significant Bit (LSB) Flip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LSB flipping changes the least significant bit of selected values, toggling between even and odd values. This is the least severe fault type and models minor bit errors that commonly occur in memory systems.

Most Significant Bit (MSB) Flip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MSB flipping changes the most significant bit, causing large magnitude changes in the activation value. This represents a severe fault that can significantly impact model output.

Full Bit Flip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Full bit flipping inverts all bits of the activation value, computing the bitwise complement. This represents the most extreme fault type and is used to evaluate model resilience to catastrophic errors.

Use Cases
----------

The fault injection system supports two primary use cases:

Fault-Aware Training (FAT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During training, faults are injected based on the configuration parameters. The model learns to maintain performance despite these errors, resulting in a robust model that can tolerate hardware faults in deployment. This is achieved through the gradient zeroing mechanism in the backpropagation.

Fault Resilience Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After training, fault injection can be used to evaluate model resilience. By injecting faults with different probabilities and strategies, we can quantify how much model accuracy degrades under various fault scenarios. This helps identify potential issues before deployment and provides metrics for hardware requirements.

Configuration and Control
-------------------------

The fault injection system provides multiple levels of control:

- **Global enable/disable**: Master switch to turn fault injection on or off
- **Phase control**: Inject during training, evaluation, or both
- **Probability tuning**: Adjust injection rate from 0% to 100%
- **Strategy selection**: Choose between different fault models
- **Per-layer control**: Fine-tune behavior for specific layers
- **Statistics tracking**: Monitor injection behavior and impact

This flexibility allows researchers and practitioners to experiment with different fault scenarios and find the optimal fault-aware training configuration for their specific use case.

Integration with Training Loop
------------------------------

The fault injection system integrates seamlessly with the training loop:

1. **Model Preparation**: Before training, FaultInjector injects layers into the model
2. **Configuration**: FaultInjectionConfig specifies injection parameters
3. **Training**: During forward passes, faults are automatically injected based on config
4. **Statistics**: FaultStatistics tracks injection behavior if enabled
5. **Analysis**: After training, statistics can be analyzed to understand fault impact

No changes to the training loop are required - fault injection is completely transparent once the model is wrapped.

Next Steps
----------

For detailed information about specific components, see:

- :doc:`injector` - Model transformation and layer injection
- :doc:`wrapper` - Wrapper mechanism for combining layers
- :doc:`layers` - Core fault injection layer implementation
- :doc:`config` - Configuration parameters and validation
- :doc:`statistics` - Statistics tracking and reporting
- :doc:`strategies/index` - Fault injection strategies
