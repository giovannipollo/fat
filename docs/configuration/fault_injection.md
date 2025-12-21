# Fault Injection

This page provides comprehensive documentation for the fault injection framework, which enables Fault-Aware Training (FAT) and fault resilience evaluation for quantized neural networks.

## Overview

### What is Fault Injection?

Fault injection is a technique for simulating hardware errors in neural network computations. When quantized models are deployed on edge devices or specialized hardware accelerators, they may experience various types of faults due to:

- **Soft errors**: Caused by cosmic rays or alpha particles flipping bits in memory
- **Voltage fluctuations**: Leading to incorrect computations
- **Aging effects**: Degradation of hardware over time
- **Manufacturing defects**: Permanent faults in specific memory cells

By injecting controlled faults during training or evaluation, we can:

1. **Train robust models** that maintain accuracy even when faults occur (Fault-Aware Training)
2. **Evaluate resilience** of existing models to various fault scenarios
3. **Identify vulnerable layers** that are most sensitive to errors
4. **Compare architectures** for their inherent fault tolerance

### How It Works

The fault injection framework operates at the **activation level** of quantized neural networks. It intercepts the output of quantized activation layers (such as `QuantIdentity`, `QuantReLU`, `QuantHardTanh` from Brevitas) and modifies a percentage of values according to a specified fault model.

```
┌─────────────────┐     ┌─────────────────────────┐     ┌─────────────────┐
│  Previous Layer │ ──▶ │  Fault Injection Layer  │ ──▶ │    Next Layer   │
│  (QuantReLU)    │     │  - Generate fault mask  │     │    (QuantConv)  │
│                 │     │  - Apply fault strategy │     │                 │
└─────────────────┘     │  - Track statistics     │     └─────────────────┘
                        └─────────────────────────┘
```

The key insight is that faults are injected **after quantization** but **before the next operation**, simulating errors that would occur in the quantized integer representation on hardware.

---

## Quick Start

### Basic Configuration

Add the `fault_injection` section to your YAML configuration:

```yaml
fault_injection:
  enabled: true
  probability: 5.0
  injection_type: "random"
  apply_during: "train"
  track_statistics: true
  verbose: true
```

### Training with Fault Injection

```bash
python train.py --config configs/fault_injection_example.yaml
```

### Evaluating Fault Resilience

```bash
python evaluate.py --config configs/your_config.yaml \
    --checkpoint path/to/checkpoint.pth \
    --probability 10.0
```

---

## Configuration Reference

### Complete Configuration Example

```yaml
fault_injection:
  # Master switch - enables/disables all fault injection
  enabled: true
  
  # Probability of fault injection (0-100%)
  probability: 5.0
  
  # Injection mode: "full_model" or "layer"
  mode: "full_model"
  
  # Layer index for "layer" mode (-1 = random selection)
  injection_layer: -1
  
  # Fault model: "random", "lsb_flip", "msb_flip", "full_flip"
  injection_type: "random"
  
  # When to inject: "train", "eval", or "both"
  apply_during: "train"
  
  # Epoch interval (1 = every epoch)
  epoch_interval: 1
  
  # Step interval (0-1): probability per training step
  step_interval: 0.5
  
  # Random seed for reproducibility (null = random)
  seed: null
  
  # Enable statistics tracking
  track_statistics: true
  
  # Verbose output for debugging
  verbose: false
  
  # Hardware-aware periodic fault pattern (simulates real FPGA/ASIC behavior)
  hw_mask: false
  
  # Hardware parallelism factor for hw_mask (number of parallel MAC units)
  frequency_value: 1024
  
  # Gradient mode for backpropagation: "ste" or "zero_faulty"
  gradient_mode: "ste"
```

### Configuration Parameters

#### enabled

| Property | Value |
|----------|-------|
| Type | `boolean` |
| Default | `false` |
| Required | No |

Master switch that enables or disables fault injection. When `false`, no fault injection layers are added to the model, resulting in zero overhead.

```yaml
fault_injection:
  enabled: true  # Enable fault injection
```

---

#### probability

| Property | Value |
|----------|-------|
| Type | `float` |
| Range | `0.0` to `100.0` |
| Default | `5.0` |
| Required | No |

The percentage of activation values that will be modified when fault injection is active. This is the **per-element** probability, meaning each activation value independently has this probability of being corrupted.

**Example values:**

| Probability | Use Case |
|------------|----------|
| `1.0` | Light fault injection for initial experiments |
| `5.0` | Typical FAT training |
| `10.0` | Aggressive fault injection |
| `20.0+` | Stress testing / worst-case evaluation |

```yaml
fault_injection:
  probability: 5.0  # 5% of activations will be corrupted
```

!!! tip "Starting Point"
    Start with low probabilities (1-5%) and increase gradually. High probabilities (>20%) can prevent model convergence during training.

---

#### injection_type

| Property | Value |
|----------|-------|
| Type | `string` |
| Options | `"random"`, `"lsb_flip"`, `"msb_flip"`, `"full_flip"` |
| Default | `"random"` |
| Required | No |

The fault model (injection strategy) determines how activation values are corrupted:

##### `"random"` - Random Value Replacement

Adds a random integer value using modular arithmetic, resulting in a uniformly distributed random value within the valid quantization range.

**Mathematical operation:**
```
faulty_int = ((original_int + random_int - min_val) % range_size) + min_val
```

**Characteristics:**

- Most general-purpose fault model
- Simulates unpredictable memory corruption
- Recommended for FAT training

```yaml
fault_injection:
  injection_type: "random"
```

##### `"lsb_flip"` - Least Significant Bit Flip

Flips the least significant bit of the quantized integer value.

**Mathematical operation:**
```
faulty_int = original_int XOR 1
```

**Characteristics:**

- Minimal value change (±1 in integer domain)
- Models small precision errors
- Lower impact on model accuracy

```yaml
fault_injection:
  injection_type: "lsb_flip"
```

##### `"msb_flip"` - Most Significant Bit Flip

Flips the most significant bit of the quantized integer value.

**Mathematical operation:**
```
faulty_int = original_int XOR (1 << (bit_width - 1))
```

**Characteristics:**

- Maximum value change for a single bit flip
- Models severe single-bit errors
- High impact on model accuracy (stress test)

```yaml
fault_injection:
  injection_type: "msb_flip"
```

##### `"full_flip"` - All Bits Flip

Flips all bits in the quantized integer value (bitwise NOT).

**Mathematical operation:**
```
faulty_int = original_int XOR ((1 << bit_width) - 1)
```

**Characteristics:**

- Complete value inversion
- Models worst-case corruption
- Useful for extreme stress testing

```yaml
fault_injection:
  injection_type: "full_flip"
```

##### Comparison of Injection Types

| Type | Impact | Use Case |
|------|--------|----------|
| `random` | Variable | General FAT, realistic fault simulation |
| `lsb_flip` | Minimal | Precision error simulation |
| `msb_flip` | High | Single-bit stress testing |
| `full_flip` | Maximum | Extreme stress testing |

---

#### apply_during

| Property | Value |
|----------|-------|
| Type | `string` |
| Options | `"train"`, `"eval"`, `"both"` |
| Default | `"train"` |
| Required | No |

Controls when fault injection is active:

##### `"train"` - Training Only

Faults are injected only during the training forward pass. The model sees clean data during validation/testing.

**Use case:** Fault-Aware Training (FAT) where you want to train with faults but evaluate on clean data to measure true accuracy.

```yaml
fault_injection:
  apply_during: "train"
```

##### `"eval"` - Evaluation Only

Faults are injected only during evaluation (validation/test). The model trains on clean data.

**Use case:** Evaluating fault resilience of pre-trained models.

```yaml
fault_injection:
  apply_during: "eval"
```

##### `"both"` - Training and Evaluation

Faults are injected during both training and evaluation.

**Use case:** Research scenarios where you want consistent fault exposure, or when measuring FAT effectiveness under faults.

```yaml
fault_injection:
  apply_during: "both"
```

---

#### verbose

| Property | Value |
|----------|-------|
| Type | `boolean` |
| Default | `false` |
| Required | No |

Enable detailed logging of fault injection operations. Useful for debugging and understanding injection behavior.

```yaml
fault_injection:
  verbose: true
```

**Example verbose output:**
```
Fault injection enabled: 12 injection layers added
  Mode: full_model
  Probability: 5.0%
  Injection type: random
  Apply during: train
Randomly selected layer 7 for injection
```

---


## Target Layers

Fault injection is applied after these Brevitas quantized activation layers:

| Layer Type | Description |
|------------|-------------|
| `QuantIdentity` | Identity function with quantization |
| `QuantReLU` | ReLU activation with quantization |
| `QuantHardTanh` | HardTanh activation with quantization |

The `FaultInjector` automatically traverses the model graph and identifies these layers at runtime. No modifications to model definitions are required.

### How Injection Works

1. **Model Analysis**: The injector counts all target layers in the model
2. **Layer Wrapping**: Each target layer is wrapped with a `QuantFaultInjectionLayer`
3. **Runtime Injection**: During forward pass, the injection layer:
   - Receives the `QuantTensor` output
   - Generates a fault mask based on probability
   - Applies the injection strategy to masked values
   - Returns the modified `QuantTensor`

### Backpropagation Support

The fault injection layer supports configurable gradient modes for backpropagation, controlled by the `gradient_mode` parameter:

**STE Mode (default):**
```python
# Forward: Apply injection
output_value = torch.where(mask, faulty_values, x)

# Backward: Straight-through (all gradients pass unchanged)
grad_x = grad_output  # [1, 1, 1, 1]
```

**Zero-Faulty Mode:**
```python
# Forward: Apply injection  
output_value = torch.where(mask, faulty_values, x)

# Backward: Zero gradients at faulty positions
grad_x = torch.where(mask, zeros, grad_output)  # [1, 0, 1, 1]
```

Both modes allow Fault-Aware Training to work correctly with standard optimizers. See the [`gradient_mode`](#gradient_mode) parameter for details.

---

## Evaluation Script

The framework includes a dedicated evaluation script for assessing fault resilience.

### Basic Usage

```bash
python evaluate.py --config configs/your_config.yaml \
    --checkpoint path/to/checkpoint.pth
```

### Command-Line Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--config` | string | Yes | Path to configuration file |
| `--checkpoint` | string | Yes | Path to model checkpoint |
| `--probability` | float | No | Override fault probability (0-100) |
| `--injection-type` | string | No | Override injection type |
| `--mode` | string | No | Override injection mode |
| `--sweep` | string | No | Comma-separated probabilities for sweep |
| `--num-runs` | int | No | Number of runs per probability |
| `--output` | string | No | Output file path (JSON) |
| `--no-progress` | flag | No | Disable progress bars |

### Single Probability Evaluation

Evaluate with a specific fault probability:

```bash
python evaluate.py --config configs/quant_cnv_w2a2_cifar10.yaml \
    --checkpoint experiments/quant_cnv/best.pth \
    --probability 10.0
```

**Output:**
```
Using device: cuda
Dataset: cifar10 (157 test batches)
Model: quant_cnv
Loading checkpoint: experiments/quant_cnv/best.pth
Checkpoint accuracy: 87.45%

Fault Injection Configuration:
  Injection layers: 12
  Mode: full_model
  Injection type: random
  Probability: 10.0%

============================================================
Evaluating baseline (no faults)...
Baseline accuracy: 87.45%

============================================================
Evaluating with 10.0% fault probability...

Results:
  Baseline accuracy:   87.45%
  Fault accuracy:      82.31%
  Accuracy degradation: +5.14%

------------------------------------------------------------
FAULT INJECTION STATISTICS
============================================================
Layer    Injected     Total          Rate (%)     RMSE         Cos Sim (%) 
------------------------------------------------------------
0        12543        125430         10.0000      2.3451       97.8234     
1        8721         87210          10.0000      1.9876       98.1234     
...
------------------------------------------------------------
TOTAL    156432       1564320        10.0000     
============================================================

Results saved to: experiments/quant_cnv/fault_eval_random.json
```

### Probability Sweep

Evaluate across multiple fault probabilities to find the resilience threshold:

```bash
python evaluate.py --config configs/quant_cnv_w2a2_cifar10.yaml \
    --checkpoint experiments/quant_cnv/best.pth \
    --sweep 0,1,2,5,10,20,50
```

**Output:**
```
============================================================
Evaluating baseline (no faults)...
Baseline accuracy: 87.45%

============================================================

Evaluating with probability 0%...
  Accuracy: 87.45% (std: 0.00%)

Evaluating with probability 1%...
  Accuracy: 86.92% (std: 0.00%)

Evaluating with probability 2%...
  Accuracy: 86.21% (std: 0.00%)

Evaluating with probability 5%...
  Accuracy: 84.56% (std: 0.00%)

Evaluating with probability 10%...
  Accuracy: 82.31% (std: 0.00%)

Evaluating with probability 20%...
  Accuracy: 75.43% (std: 0.00%)

Evaluating with probability 50%...
  Accuracy: 52.18% (std: 0.00%)

============================================================
Sweep Summary:
 Probability |   Accuracy |  Degradation
----------------------------------------
        0.0% |     87.45% |       +0.00%
        1.0% |     86.92% |       +0.53%
        2.0% |     86.21% |       +1.24%
        5.0% |     84.56% |       +2.89%
       10.0% |     82.31% |       +5.14%
       20.0% |     75.43% |      +12.02%
       50.0% |     52.18% |      +35.27%

Results saved to: experiments/quant_cnv/fault_eval_random.json
```

### Multiple Runs for Statistical Significance

Run multiple evaluations with different random seeds:

```bash
python evaluate.py --config configs/quant_cnv_w2a2_cifar10.yaml \
    --checkpoint experiments/quant_cnv/best.pth \
    --sweep 5,10,20 \
    --num-runs 5
```

This runs 5 evaluations at each probability level and reports mean ± standard deviation.

---

## Programmatic Usage

### Basic Usage

```python
from utils.fault_injection import (
    FaultInjectionConfig,
    FaultInjector,
    FaultStatistics,
)

# Create configuration
config = FaultInjectionConfig(
    enabled=True,
    probability=5.0,
    mode="full_model",
    injection_type="random",
    apply_during="train",
)

# Validate configuration
config.validate()

# Create injector and inject layers into model
injector = FaultInjector()
model = injector.inject(model, config)

# Get number of injection layers
num_layers = injector.get_num_layers(model)
print(f"Injected {num_layers} fault injection layers")
```

### Statistics Tracking

```python
# Create statistics tracker
stats = FaultStatistics(num_layers=num_layers)

# Attach statistics to injection layers
injector.set_statistics(model, stats)

# Run inference...
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)

# Print statistics report
stats.print_report()

# Save to file
stats.save_to_file("fault_stats.json")

# Access statistics programmatically
for layer_id, layer_stats in stats.layer_stats.items():
    print(f"Layer {layer_id}:")
    print(f"  Injection rate: {layer_stats.injection_rate:.2f}%")
    print(f"  Average RMSE: {layer_stats.avg_rmse:.4f}")
    print(f"  Cosine similarity: {layer_stats.avg_cosine_similarity:.4f}")
```

### Training Loop Integration

```python
# Training loop with fault injection
for epoch in range(num_epochs):
    # Update epoch for all injection layers
    injector.update_epoch(model, epoch)
    
    # Reset iteration counters
    injector.reset_counters(model)
    
    # Set up step-level injection schedule
    if config.step_interval > 0:
        num_iterations = len(train_loader)
        injector.set_condition_injector(
            model, num_iterations, config.step_interval
        )
    
    # Enable/disable based on apply_during
    if config.apply_during in ("train", "both"):
        injector.set_enabled(model, True)
    else:
        injector.set_enabled(model, False)
    
    # Training loop
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    if config.apply_during in ("eval", "both"):
        injector.set_enabled(model, True)
    else:
        injector.set_enabled(model, False)
    
    model.eval()
    # ... evaluation code ...
```

### Dynamic Probability Adjustment

```python
# Update probability during training
injector.update_probability(model, probability=10.0)

# Update probability for specific layer only
injector.update_probability(model, probability=15.0, layer_id=3)
```

### Removing Injection Layers

```python
# Remove all fault injection layers
model = injector.remove(model)

# Now model is back to original structure
```

### Loading Configuration from YAML

```python
import yaml

with open("configs/fault_injection_example.yaml") as f:
    full_config = yaml.safe_load(f)

fi_config = full_config.get("fault_injection", {})
config = FaultInjectionConfig.from_dict(fi_config)
```

---

## Statistics Output

### Console Report

When `track_statistics: true`, a detailed report is printed:

```
==========================================================================================
FAULT INJECTION STATISTICS
==========================================================================================
Layer    Injected     Total          Rate (%)     RMSE         Cos Sim (%) 
------------------------------------------------------------------------------------------
0        12543        125430         10.0000      2.3451       97.8234     
1        8721         87210          10.0000      1.9876       98.1234     
2        15432        154320         10.0000      2.5678       97.5432     
3        9876         98760          10.0000      2.1234       97.9876     
4        11234        112340         10.0000      2.2345       97.8765     
------------------------------------------------------------------------------------------
TOTAL    57806        578060         10.0000     
==========================================================================================
```

### JSON Output

Statistics are saved to `fault_injection_stats.json`:

```json
{
  "num_layers": 5,
  "total_injections": 57806,
  "total_activations": 578060,
  "overall_injection_rate": 10.0,
  "layers": {
    "0": {
      "layer_id": 0,
      "total_activations": 125430,
      "injected_count": 12543,
      "injection_rate": 10.0,
      "avg_rmse": 2.3451,
      "avg_cosine_similarity": 0.978234,
      "sample_count": 157
    },
    "1": {
      "layer_id": 1,
      "total_activations": 87210,
      "injected_count": 8721,
      "injection_rate": 10.0,
      "avg_rmse": 1.9876,
      "avg_cosine_similarity": 0.981234,
      "sample_count": 157
    }
  }
}
```

### Metrics Explained

| Metric | Description |
|--------|-------------|
| **Injected** | Number of activation values that were corrupted |
| **Total** | Total number of activation values processed |
| **Rate (%)** | Actual injection rate (Injected / Total × 100) |
| **RMSE** | Root Mean Squared Error between original and corrupted values |
| **Cos Sim (%)** | Cosine similarity between original and corrupted activation tensors |

---

## Example Configurations

### Fault-Aware Training (FAT)

Standard FAT configuration for training robust models:

```yaml
# configs/fat_training.yaml
seed:
  enabled: true
  value: 42

dataset:
  name: "cifar10"
  root: "./data"

model:
  name: "quant_cnv"

quantization:
  weight_bit_width: 2
  act_bit_width: 2

training:
  batch_size: 64
  epochs: 300

optimizer:
  name: "adam"
  learning_rate: 0.01

scheduler:
  name: "cosine"
  T_max: 300

fault_injection:
  enabled: true
  probability: 5.0
  mode: "full_model"
  injection_type: "random"
  apply_during: "train"
  epoch_interval: 1
  step_interval: 0.5
  track_statistics: false  # Disable for training speed

checkpoint:
  enabled: true
  dir: "./experiments"
  save_best: true
```

### Fault Evaluation

Configuration for evaluating pre-trained model resilience:

```yaml
# configs/fault_eval.yaml
seed:
  enabled: true
  value: 42
  deterministic: true  # Reproducible evaluation

dataset:
  name: "cifar10"
  root: "./data"

model:
  name: "quant_cnv"

quantization:
  weight_bit_width: 2
  act_bit_width: 2

training:
  batch_size: 64
  epochs: 1  # Not used for evaluation

fault_injection:
  enabled: true
  probability: 10.0
  mode: "full_model"
  injection_type: "random"
  apply_during: "eval"
  seed: 42  # Reproducible faults
  track_statistics: true
  verbose: true
```

### Layer Sensitivity Analysis

Configuration for analyzing which layers are most vulnerable:

```yaml
# configs/layer_analysis.yaml
fault_injection:
  enabled: true
  probability: 20.0
  mode: "layer"
  injection_layer: -1  # Random layer each run
  injection_type: "random"
  apply_during: "eval"
  track_statistics: true
```

Run multiple times and compare per-layer accuracy degradation.

### Curriculum Fault Training

Gradually increase fault exposure during training:

```yaml
# Start with no faults, then increase
fault_injection:
  enabled: true
  probability: 10.0
  mode: "full_model"
  injection_type: "random"
  apply_during: "train"
  epoch_interval: 5  # Faults every 5th epoch initially
  step_interval: 0.3  # Only 30% of steps
```

Then modify during training or use a callback to increase `step_interval` and decrease `epoch_interval` as training progresses.

### Bit-Flip Stress Testing

Configuration for worst-case fault analysis:

```yaml
fault_injection:
  enabled: true
  probability: 5.0
  mode: "full_model"
  injection_type: "msb_flip"  # Maximum impact per fault
  apply_during: "eval"
  track_statistics: true
```

---

## Best Practices

### 1. Start with Low Probability

Begin with 1-5% injection probability and increase gradually:

```yaml
# Initial experiments
fault_injection:
  probability: 1.0  # Start low
```

High probabilities (>20%) can prevent model convergence during FAT.

### 2. Use step_interval for Gradual Training

Mixing clean and noisy training steps often works better than constant fault exposure:

```yaml
fault_injection:
  probability: 10.0
  step_interval: 0.5  # 50% clean, 50% noisy steps
```

### 3. Establish Baseline First

Always evaluate without faults first to establish baseline accuracy:

```bash
# Baseline (no faults)
python evaluate.py --config config.yaml --checkpoint model.pth --probability 0

# With faults
python evaluate.py --config config.yaml --checkpoint model.pth --probability 10
```

### 4. Use Probability Sweeps

Find the fault tolerance threshold using sweeps:

```bash
python evaluate.py --config config.yaml --checkpoint model.pth \
    --sweep 0,1,2,5,10,15,20,30,50
```

### 5. Track Statistics During Development

Enable statistics tracking to understand injection behavior:

```yaml
fault_injection:
  track_statistics: true
  verbose: true  # During development only
```

Disable `verbose` for production training runs.

### 6. Use Reproducible Seeds for Comparisons

When comparing different models or configurations:

```yaml
seed:
  enabled: true
  value: 42
  deterministic: true

fault_injection:
  seed: 42  # Same fault pattern
```

### 7. Consider Quantization Bit Width

Lower bit widths have fewer possible values, so faults have proportionally larger impact:

| Bit Width | Value Range (signed) | Fault Impact |
|-----------|---------------------|--------------|
| 2-bit | -1 to 1 | Very High |
| 4-bit | -7 to 7 | High |
| 8-bit | -127 to 127 | Moderate |

Adjust probability accordingly.

### 8. Validate with Multiple Injection Types

Test resilience against different fault models:

```bash
for type in random lsb_flip msb_flip full_flip; do
    python evaluate.py --config config.yaml --checkpoint model.pth \
        --injection-type $type --probability 10
done
```

---

## Troubleshooting

### Model Not Converging with FAT

**Symptoms:** Training loss stays high or accuracy doesn't improve.

**Solutions:**

1. Reduce `probability`:
   ```yaml
   fault_injection:
     probability: 1.0  # Start very low
   ```

2. Reduce `step_interval`:
   ```yaml
   fault_injection:
     step_interval: 0.2  # Only 20% of steps have faults
   ```

3. Use `epoch_interval` to start clean:
   ```yaml
   fault_injection:
     epoch_interval: 10  # Faults only every 10th epoch initially
   ```

### No Injection Layers Added

**Symptoms:** "Injected 0 fault injection layers" message.

**Solutions:**

1. Verify model uses Brevitas quantized layers:
   ```python
   for name, module in model.named_modules():
       print(name, type(module).__name__)
   ```

2. Check that model is a quantized model (e.g., `quant_cnv`, not `cnv`)

3. Ensure `enabled: true` in configuration

### Statistics Show 0% Injection Rate

**Symptoms:** Statistics report shows no injections despite `enabled: true`.

**Solutions:**

1. Check `apply_during` matches current phase:
   ```yaml
   fault_injection:
     apply_during: "eval"  # Make sure you're evaluating, not training
   ```

2. Verify `epoch_interval` allows injection at current epoch

3. Check `step_interval` is not 0.0

### Different Results Between Runs

**Symptoms:** Accuracy varies significantly between evaluation runs.

**Solutions:**

1. Set deterministic seed:
   ```yaml
   seed:
     enabled: true
     value: 42
     deterministic: true
   
   fault_injection:
     seed: 42
   ```

2. Use `--num-runs` in evaluation for statistical averaging:
   ```bash
   python evaluate.py --config config.yaml --checkpoint model.pth \
       --probability 10 --num-runs 10
   ```

---

## Architecture Details

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         YAML Config                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FaultInjectionConfig                         │
│  - Parses and validates configuration                           │
│  - Provides typed access to parameters                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FaultInjector                             │
│  - Analyzes model structure                                     │
│  - Identifies target layers (QuantIdentity, QuantReLU, etc.)    │
│  - Inserts QuantFaultInjectionLayer at runtime                  │
│  - Manages epoch/step updates                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 QuantFaultInjectionLayer                        │
│  - Receives QuantTensor from previous layer                     │
│  - Generates fault mask based on probability                    │
│  - Applies injection strategy                                   │
│  - Tracks statistics (optional)                                 │
│  - Returns modified QuantTensor                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    InjectionStrategy                            │
│  - RandomStrategy: Modular random addition                      │
│  - LSBFlipStrategy: XOR with 1                                  │
│  - MSBFlipStrategy: XOR with MSB                                │
│  - FullFlipStrategy: XOR with all 1s                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FaultStatistics                              │
│  - Per-layer injection counts                                   │
│  - RMSE computation                                             │
│  - Cosine similarity computation                                │
│  - Report generation                                            │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
utils/fault_injection/
├── __init__.py          # Public API exports
├── config.py            # FaultInjectionConfig dataclass
├── strategies.py        # Injection strategy implementations
├── layers.py            # QuantFaultInjectionLayer
├── statistics.py        # FaultStatistics tracking
└── injector.py          # FaultInjector (model transformer)
```

---

## API Reference

### FaultInjectionConfig

```python
@dataclass
class FaultInjectionConfig:
    enabled: bool = False
    probability: float = 5.0
    injection_type: str = "random"
    apply_during: str = "eval"
    track_statistics: bool = False
    verbose: bool = False
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FaultInjectionConfig": ...
    
    def validate(self) -> None: ...
```

### FaultInjector

```python
class FaultInjector:
    QUANT_TARGET_LAYERS: Set[str]  # {"QuantIdentity", "QuantReLU", "QuantHardTanh"}
    
    def inject(self, model: nn.Module, config: FaultInjectionConfig) -> nn.Module: ...
    def remove(self, model: nn.Module) -> nn.Module: ...
    def update_probability(self, model: nn.Module, probability: float, layer_id: Optional[int] = None) -> None: ...
    def set_enabled(self, model: nn.Module, enabled: bool) -> None: ...
    def set_statistics(self, model: nn.Module, statistics: FaultStatistics) -> None: ...
    def get_num_layers(self, model: nn.Module) -> int: ...
```

### FaultStatistics

```python
class FaultStatistics:
    def __init__(self, num_layers: int) -> None: ...
    def record(self, clean_int: Tensor, faulty_int: Tensor, mask: Tensor, layer_id: int) -> None: ...
    def reset(self) -> None: ...
    def print_report(self) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
    def save_to_file(self, path: str) -> None: ...
```

### InjectionStrategy

```python
class InjectionStrategy(ABC):
    @abstractmethod
    def inject(self, int_tensor: Tensor, mask: Tensor, bit_width: int, signed: bool, device: torch.device) -> Tensor: ...

class RandomStrategy(InjectionStrategy): ...
class LSBFlipStrategy(InjectionStrategy): ...
class MSBFlipStrategy(InjectionStrategy): ...
class FullFlipStrategy(InjectionStrategy): ...

def get_strategy(name: str) -> InjectionStrategy: ...
```

---

## References

- **Brevitas**: [https://github.com/Xilinx/brevitas](https://github.com/Xilinx/brevitas)
- **Fault-Aware Training**: Research on training neural networks to be robust against hardware faults
- **Quantization-Aware Training**: Training with simulated quantization for deployment on integer-only hardware
