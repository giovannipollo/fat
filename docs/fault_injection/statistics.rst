.. _fault_statistics_module:

Fault Statistics Tracking
======================

The fault statistics system provides comprehensive tracking and reporting of fault injection behavior. It records metrics about injections, computes error statistics, and enables detailed analysis of how faults affect model outputs.

Overview
--------

Statistics tracking serves several important purposes in the fault injection system:

- **Verification**: Confirm that faults are being injected at the expected rate
- **Analysis**: Understand the impact of faults on model outputs
- **Debugging**: Identify patterns or anomalies in fault injection behavior
- **Reporting**: Provide metrics for papers, presentations, or model evaluation
- **Comparison**: Compare different fault injection strategies or configurations

The statistics system works by recording information about each injection event and then aggregating and summarizing this information in a human-readable format.

LayerStatistics Class
---------------------

The LayerStatistics class tracks statistics for a single fault injection layer.

Tracked Metrics
^^^^^^^^^^^^^^^^^^^

Each LayerStatistics instance maintains several metrics:

- **total_activations**: Total number of activation values that have been processed
- **injected_count**: Number of activation values that had faults injected
- **injection_rate**: Percentage of activations that were modified (computed as property)
- **rmse_sum**: Sum of root mean squared error across all samples
- **avg_rmse**: Average RMSE across all samples (computed as property)
- **cosine_similarity_sum**: Sum of cosine similarity values across all samples
- **avg_cosine_similarity**: Average cosine similarity (computed as property)
- **sample_count**: Number of forward passes (samples) that were recorded

Injection Rate
^^^^^^^^^^^^^^^^^^^

The injection_rate property calculates the actual percentage of activations that were modified:

.. code-block:: python

    @property
    def injection_rate(self) -> float:
        if self.total_activations == 0:
            return 0.0
        return 100.0 * self.injected_count / self.total_activations

This metric is useful for verifying that the configured probability matches the actual injection behavior. For example, if you set a probability of 5.0%, the injection_rate should be approximately 5.0% across many forward passes.

Due to the discrete nature of sampling, the actual rate will vary slightly around the target. As more activations are processed, the actual rate should converge toward the configured probability.

Root Mean Squared Error (RMSE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The RMSE measures the magnitude of errors introduced by fault injection. It is computed as:

.. code-block:: python

    clean_diff = clean_int[different_mask].float()
    faulty_diff = faulty_int[different_mask].float()
    diff = clean_diff - faulty_diff
    rmse = torch.sqrt(torch.mean(diff**2)).item()

RMSE is only computed for positions where faults were injected (where clean and faulty values differ). This provides a measure of how much the faults are changing the activation values.

Higher RMSE values indicate that the injected faults are causing larger changes to the activations. This can help assess the severity of different fault injection strategies:

- **LSB flip**: Typically results in small RMSE (changes of 1-2 in value)
- **MSB flip**: Typically results in large RMSE (changes of half the quantization range)
- **Random**: Variable RMSE depending on the range of random values
- **Full flip**: Maximum possible RMSE for the quantization range

Cosine Similarity
^^^^^^^^^^^^^^^^^^^^^^^^^

Cosine similarity measures the overall similarity between clean and faulty activation tensors. It is computed as:

.. code-block:: python

    clean_flat = clean_int.flatten().float()
    faulty_flat = faulty_int.flatten().float()

    clean_norm = torch.norm(clean_flat)
    faulty_norm = torch.norm(faulty_flat)

    if clean_norm > 0 and faulty_norm > 0:
        cos_sim = torch.dot(clean_flat, faulty_flat) / (
            clean_norm * faulty_norm
        )
        stats.cosine_similarity_sum += cos_sim.item()
    else:
        stats.cosine_similarity_sum += 1.0

Cosine similarity ranges from -1 to 1, where:

- **1.0**: Vectors are identical (no injection or injections that don't change the tensor)
- **0.0**: Vectors are orthogonal (completely different)
- **-1.0**: Vectors are opposite in direction

For fault injection, cosine similarity typically decreases as more activations are modified and as the magnitude of modifications increases. This metric is useful because it captures the overall similarity of the entire activation tensor, not just the individual values that were modified.

Sample Count
^^^^^^^^^^^^^^^^^^

The sample_count tracks how many forward passes have been recorded. This is important because both RMSE and cosine similarity are computed per forward pass and then averaged across all passes.

The average values (avg_rmse and avg_cosine_similarity) are computed as:

.. code-block:: python

    def avg_rmse(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.rmse_sum / self.sample_count

This means that if one forward pass had a very high RMSE and another had a very low RMSE, the average would reflect both. This provides a more stable metric than looking at any single forward pass.

FaultStatistics Class
--------------------

The FaultStatistics class aggregates statistics across all fault injection layers in the model.

Multi-Layer Tracking
^^^^^^^^^^^^^^^^^^^^^

While LayerStatistics tracks metrics for a single layer, FaultStatistics maintains a dictionary of LayerStatistics objects, one for each fault injection layer in the model:

.. code-block:: python

    self.layer_stats: Dict[int, LayerStatistics] = {
        i: LayerStatistics(layer_id=i) for i in range(num_layers)
    }

This allows for analyzing fault injection behavior at different layers in the network. For example, early layers might have different fault characteristics than later layers due to differences in activation distributions and magnitudes.

The class is initialized with the number of layers, which is typically obtained from the FaultInjector after it has completed injection:

.. code-block:: python

    num_layers = injector.get_num_layers(model)
    stats = FaultStatistics(num_layers=num_layers)

Recording Process
~~~~~~~~~~~~~~~~~~

During a forward pass, each QuantFaultInjectionLayer can record its injection statistics:

.. code-block:: python

    stats.record(
        clean_int=int_tensor,
        faulty_int=injected_int,
        mask=condition_tensor,
        layer_id=self.layer_id,
    )

The FaultStatistics.record() method handles the recording:

1. Verifies that statistics tracking is enabled
2. Looks up the LayerStatistics for the specified layer_id
3. Calls the appropriate recording methods on that LayerStatistics instance

If a layer_id is provided that does not exist in the layer_stats dictionary, the method automatically creates a new LayerStatistics instance for that layer. This provides flexibility for dynamic layer additions or for recording statistics before all layers are known.

Aggregation and Reporting
------------------------

After training or evaluation is complete, the FaultStatistics class provides several methods for viewing and analyzing the collected statistics.

Print Report
~~~~~~~~~~~~~~

The print_report() method generates a formatted table showing statistics for all layers:

.. code-block:: python

    stats.print_report()

The report includes:

- **Layer ID**: Identifier for each layer
- **Injected count**: Number of activations that were modified
- **Total activations**: Total number of activations processed
- **Rate**: Actual injection rate as percentage
- **RMSE**: Average root mean squared error across all samples
- **Cos Sim**: Average cosine similarity as percentage

The report also includes a summary row with totals across all layers, providing an overall view of fault injection behavior.

Export to Dictionary
~~~~~~~~~~~~~~~~~~~~

The to_dict() method exports all statistics to a dictionary structure:

.. code-block:: python

    stats_dict = stats.to_dict()

The exported dictionary has the following structure:

.. code-block:: python

    {
        "summary": {
            "total_injected": 12500,
            "total_activations": 250000,
            "overall_injection_rate": 5.0,
            "num_layers": 10
        },
        "layers": {
            "0": {
                "layer_id": 0,
                "total_activations": 25000,
                "injected_count": 1250,
                "injection_rate": 5.0,
                "avg_rmse": 1.5,
                "avg_cosine_similarity": 0.98,
                "sample_count": 10
            },
            # ... other layers ...
        }
    }

This dictionary can be used for:

- **JSON serialization**: Saving statistics to a file
- **Logging**: Recording statistics for training runs
- **Analysis**: Further processing or visualization of statistics
- **Comparison**: Comparing statistics from different configurations

Save to File
~~~~~~~~~~~~~~

The save_to_file() method writes statistics to a JSON file:

.. code-block:: python

    stats.save_to_file("fault_stats.json")

This is convenient for archiving statistics from training or evaluation runs. The JSON format ensures that the statistics can be easily loaded later for analysis or for comparison across runs.

Load from File
~~~~~~~~~~~~~~

The load_from_file() classmethod creates a FaultStatistics instance from a saved JSON file:

.. code-block:: python

    stats = FaultStatistics.load_from_file("fault_stats.json")

This allows for reloading and analyzing previously collected statistics. The loaded statistics object has all of the same methods as a newly created object, enabling report generation, comparison, and further analysis.

Note that when loading from a file, only the aggregated averages are available. The individual sums (rmse_sum, cosine_similarity_sum) are not preserved, because only the averaged values are saved to the file.

Get Summary
~~~~~~~~~~~~

The get_summary() method returns a dictionary with overall statistics across all layers:

.. code-block:: python

    summary = stats.get_summary()
    # Returns:
    # {
    #     "total_injected": 12500,
    #     "total_activations": 250000,
    #     "overall_injection_rate": 5.0,
    #     "num_layers": 10
    # }

The overall_injection_rate property provides convenient access to this summary value:

.. code-block:: python

    rate = stats.overall_injection_rate

This is useful for quick checks of how closely the actual injection rate matches the configured probability.

Reset Statistics
-----------------

The statistics can be reset to start fresh collection.

Reset Individual Layer
~~~~~~~~~~~~~~~~~~~~

Each LayerStatistics instance can be reset:

.. code-block:: python

    layer_stats.reset()

This sets all counters and accumulated values back to zero, while preserving the layer_id.

Reset All Layers
~~~~~~~~~~~~~~~~~~

The entire FaultStatistics instance can be reset:

.. code-block:: python

    stats.reset()

This resets all LayerStatistics instances in the layer_stats dictionary, clearing all collected statistics. This is useful for:

- **Starting new evaluation**: Resetting statistics before running a new evaluation
- **Phase-based training**: Resetting statistics at the start of a new training phase
- **Ablation studies**: Resetting to collect fresh statistics for a different configuration

Performance Considerations
--------------------------

The statistics system is designed to have minimal overhead during training and evaluation.

Recording Overhead
~~~~~~~~~~~~~~~~~~~~

During each forward pass, the statistics recording involves:

- **Counting activations**: Adding tensor size to total
- **Counting injections**: Summing the mask
- **Computing RMSE**: If any injections occurred
- **Computing cosine similarity**: If any injections occurred

These operations use PyTorch operations on the GPU (if using CUDA), so the overhead is:

- **Vectorized**: Operations are applied to all elements simultaneously
- **Minimal compared to forward pass**: The computational cost is small relative to the layer computation itself
- **Optional**: Can be disabled completely by setting track_statistics=False

Memory Overhead
~~~~~~~~~~~~~~

The statistics system stores only summary metrics, not the raw activation tensors. This means:

- **Small memory footprint**: Only a few floating-point numbers per layer
- **No persistent tensors**: Activation tensors are freed after recording
- **Scalable**: Memory usage grows linearly with the number of layers, not with the number of activations

For a model with many layers and many forward passes, the memory usage remains small because only the aggregated statistics are stored.

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~~~

Enable and attach statistics tracking:

.. code-block:: python

    from utils.fault_injection import FaultInjector, FaultInjectionConfig, FaultStatistics

    # Create configuration with statistics enabled
    config = FaultInjectionConfig(
        enabled=True,
        probability=5.0,
        track_statistics=True,
    )

    # Inject faults into model
    injector = FaultInjector()
    model = injector.inject(model, config)

    # Create and attach statistics tracker
    stats = FaultStatistics(num_layers=injector.get_num_layers(model))
    injector.set_statistics(model, stats)

    # Train model...

    # View report
    stats.print_report()

Saving and Loading
~~~~~~~~~~~~~~~~~~

Save statistics to a file:

.. code-block:: python

    # After evaluation
    stats.save_to_file("evaluation_stats.json")

Load statistics for later analysis:

.. code-block:: python

    # Load previously saved statistics
    loaded_stats = FaultStatistics.load_from_file("evaluation_stats.json")

    # Generate report
    loaded_stats.print_report()

Resetting for New Runs
~~~~~~~~~~~~~~~~~~~~~~~~~

Reset statistics between different evaluations:

.. code-block:: python

    # First evaluation
    model.eval()
    for batch in test_loader:
        outputs = model(batch)

    # Print first evaluation results
    stats.print_report()

    # Reset for second evaluation with different config
    stats.reset()
    config.injection_type = "msb_flip"

    # Second evaluation
    for batch in test_loader:
        outputs = model(batch)

    # Print second evaluation results
    stats.print_report()

API Reference
------------

.. autoclass:: utils.fault_injection.statistics.FaultStatistics
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: utils.fault_injection.statistics.LayerStatistics
    :members:
    :undoc-members:
    :show-inheritance:
