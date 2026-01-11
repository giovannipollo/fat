.. _fault_config_module:

FaultInjectionConfig Class
=========================

The FaultInjectionConfig class is a dataclass that manages all fault injection parameters for the system. It provides a single source of truth for how, when, and where faults are injected into the model.

Overview
--------

The configuration class encapsulates all parameters that control fault injection behavior. By centralizing these parameters in a single object, the system can consistently apply the same fault injection settings across the entire model and throughout training or evaluation process.

The configuration supports initialization from:

- **Direct instantiation**: Creating an instance with specified parameters
- **Dictionary loading**: Creating from a dictionary, typically from a YAML configuration file
- **Validation**: Automatically validating parameters to ensure they are correct

Configuration Parameters
---------------------

The configuration includes several parameters that control different aspects of fault injection.

enabled
^^^^^^^^^^

The enabled parameter is a master switch for the entire fault injection system. When set to False, no faults are injected regardless of other parameter values.

This parameter is useful for:

- **Ablation studies**: Comparing model performance with and without faults
- **Debugging**: Disabling faults while keeping other aspects of the system active
- **Conditional injection**: Enabling faults only for specific training runs or phases

probability
^^^^^^^^^^^^^^

The probability parameter specifies the percentage of activations that should have faults injected. This value ranges from 0.0 to 100.0, where:

- 0.0: No faults are injected
- 5.0: Approximately 5% of activations are modified
- 100.0: All activations are modified

The actual number of injections for a given forward pass may vary slightly due to the discrete nature of sampling. For example, with a probability of 5.0% and a tensor with 1000 elements, approximately 50 elements would be modified, but the exact number may be 48, 49, 50, 51, or 52 depending on the random selection.

injection_type
^^^^^^^^^^^^^^^^

The injection_type parameter selects which fault model to use when injecting faults. The supported values are:

- **random**: Replace values with random integers within valid quantization range
- **lsb_flip**: Flip the least significant bit, toggling between even and odd values
- **msb_flip**: Flip the most significant bit, causing large magnitude changes
- **full_flip**: Invert all bits, computing the bitwise complement

Different fault types represent different kinds of hardware errors that might occur in quantized neural network hardware:

- **Random value errors**: Complete corruption of an activation value
- **LSB errors**: Minor bit-level errors that commonly occur in memory systems
- **MSB errors**: Severe bit errors that cause large value changes
- **Full bit errors**: Catastrophic errors where all bits are flipped

apply_during
^^^^^^^^^^^^^^^^^

The apply_during parameter controls when faults are injected. The supported values are:

- **train**: Inject faults only during the training phase
- **eval**: Inject faults only during the evaluation or testing phase
- **both**: Inject faults during both training and evaluation

This parameter provides flexibility for different use cases:

- **Fault-aware training**: Setting to "train" enables training with faults, resulting in a robust model
- **Resilience evaluation**: Setting to "eval" injects faults only during testing, measuring the model's inherent resilience
- **Comprehensive testing**: Setting to "both" evaluates both training-time robustness and inference-time resilience

track_statistics
^^^^^^^^^^^^^^^^^^^^^

The track_statistics parameter enables or disables statistics tracking during fault injection. When enabled, the system records:

- Number of injections per layer
- Actual injection rate versus configured probability
- Error metrics such as RMSE between clean and faulty outputs
- Similarity metrics such as cosine similarity between clean and faulty outputs

Statistics tracking is useful for:

- **Analyzing injection behavior**: Understanding how faults are distributed
- **Evaluating fault impact**: Measuring how much faults affect model outputs
- **Debugging**: Verifying that injection is working as expected

When statistics are not needed, disabling tracking can provide a small performance improvement.

verbose
^^^^^^^^^^

The verbose parameter controls whether to print detailed information about fault injection as it occurs. When enabled, the system prints:

- Which layers have injection layers inserted
- Number of faults injected per forward pass
- Actual injection rate versus configured probability

Verbose output is useful for:

- **Debugging**: Verifying that injection is working correctly
- **Understanding behavior**: Seeing exactly how many faults are being injected
- **Educational purposes**: Learning how the fault injection system works

Configuration Validation
----------------------

The configuration class includes built-in validation to ensure that all parameters are correct.

Probability Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The probability parameter is validated to be within the valid range:

.. code-block:: python

    if self.probability < 0.0 or self.probability > 100.0:
        raise ValueError(
            f"probability must be between 0 and 100, got {self.probability}"
        )

This validation prevents invalid configurations where the probability is outside of the meaningful range.

Injection Type Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The injection_type parameter is validated against the list of supported types:

.. code-block:: python

    if self.injection_type not in self._VALID_INJECTION_TYPES:
        raise ValueError(
            f"injection_type must be one of {self._VALID_INJECTION_TYPES}, "
            f"got '{self.injection_type}'"
        )

This validation prevents typos or invalid fault types from being used, ensuring that only the implemented strategies are used.

Apply During Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The apply_during parameter is validated against the list of supported phases:

.. code-block:: python

    if self.apply_during not in self._VALID_APPLY_DURING:
        raise ValueError(
            f"apply_during must be one of {self._VALID_APPLY_DURING}, "
            f"got '{self.apply_during}'"
        )

This validation ensures that only the supported phase options are used.

Loading from Dictionary
-----------------------

The configuration can be loaded from a dictionary, which is particularly useful for YAML configuration files.

Dictionary Structure
^^^^^^^^^^^^^^^^^^^^^^

When loading from a dictionary, the following keys are recognized:

.. code-block:: yaml

    fault_injection:
      enabled: true
      probability: 5.0
      injection_type: "lsb_flip"
      apply_during: "train"
      track_statistics: false
      verbose: false

All keys are optional. If a key is not provided in the dictionary, the default value for that parameter is used.

Loading Process
^^^^^^^^^^^^^^^^^^^^^^

The from_dict() class method handles loading:

.. code-block:: python

    config = FaultInjectionConfig.from_dict(yaml_config["fault_injection"])

This method extracts each parameter from the dictionary using .get() method, which returns None if the key is not present. The method then uses the default value for any missing parameters.

This approach allows for partial configurations where only the parameters that need to be different from defaults are specified.

Runtime Query Methods
---------------------

The configuration provides convenience methods to query whether faults should be injected during different phases.

Training Query
^^^^^^^^^^^^^^^^^

The should_inject_during_training() method checks if faults should be applied during the training phase:

.. code-block:: python

    def should_inject_during_training(self) -> bool:
        return self.enabled and self.apply_during in ("train", "both")

This method checks both the enabled flag and the apply_during setting, returning True only if faults are both enabled and configured to occur during training.

Evaluation Query
^^^^^^^^^^^^^^^^^^^^^

The should_inject_during_eval() method checks if faults should be applied during the evaluation phase:

.. code-block:: python

    def should_inject_during_eval(self) -> bool:
        return self.enabled and self.apply_during in ("eval", "both")

This method is similar to the training query but checks for the evaluation phase instead.

These convenience methods are useful because they encapsulate the logic for determining when to inject, preventing the training code from having to implement this logic directly.

Configuration Export
------------------

The configuration can be exported to a dictionary for serialization or logging.

Dictionary Export
^^^^^^^^^^^^^^^^^^^^^

The to_dict() method creates a dictionary representation of the configuration:

.. code-block:: python

    config_dict = config.to_dict()

This is useful for:

- **Logging**: Recording the exact configuration used for a training run
- **Serialization**: Saving configuration to a file or database
- **Comparison**: Comparing different configurations
- **Reconstruction**: Recreating a configuration object from saved data

The exported dictionary contains all of the configuration parameters with their current values, providing a complete representation of the fault injection settings.

Usage Examples
--------------

Direct Instantiation
^^^^^^^^^^^^^^^^^^^^^^^^

Create a configuration by directly specifying parameters:

.. code-block:: python

    from utils.fault_injection import FaultInjectionConfig

    config = FaultInjectionConfig(
        enabled=True,
        probability=5.0,
        injection_type="lsb_flip",
        apply_during="train",
        track_statistics=True,
        verbose=True,
    )

Loading from Dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Load a configuration from a YAML file:

.. code-block:: python

    import yaml

    # Load YAML configuration
    with open("config.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)

    # Create fault injection configuration
    config = FaultInjectionConfig.from_dict(yaml_config["fault_injection"])

Runtime Querying
^^^^^^^^^^^^^^^^^^^^^^

Check if faults should be injected during the current phase:

.. code-block:: python

    # Before training loop
    model.train()
    if config.should_inject_during_training():
        print("Faults will be injected during training")

    # Before evaluation
    model.eval()
    if config.should_inject_during_eval():
        print("Faults will be injected during evaluation")

Configuration Update
^^^^^^^^^^^^^^^^^^^^^^^

Modify configuration at runtime:

.. code-block:: python

    # Change probability during training
    config.probability = 10.0

    # Disable faults for specific evaluation
    config.enabled = False

    # Change fault type
    config.injection_type = "msb_flip"

These changes take effect immediately on the next forward pass, allowing for dynamic fault injection strategies.

API Reference
------------

.. autoclass:: utils.fault_injection.config.FaultInjectionConfig
    :members:
    :undoc-members:
    :show-inheritance:
