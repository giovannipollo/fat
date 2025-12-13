# Fault Injection Framework - Implementation Plan

> **Status: COMPLETED** (December 2025)
> 
> All phases of this implementation plan have been completed. See the documentation at `docs/configuration/fault_injection.md` for usage instructions.

## Overview

This document outlines the implementation plan for adding a fault injection framework to the PyTorch training codebase. The framework enables runtime injection of faults into quantized neural network activations for:

1. **Fault-Aware Training (FAT)**: Training models to be robust against hardware faults
2. **Fault Evaluation**: Assessing model resilience to activation errors

The design follows the existing implementation in `fat/src/err_inj_layer/` while providing a cleaner, more modular architecture.

---

## Architecture

### Directory Structure

```
utils/
├── fault_injection/
│   ├── __init__.py              # Public API exports
│   ├── config.py                # FaultInjectionConfig dataclass
│   ├── layers.py                # Fault injection layer implementations
│   ├── injector.py              # Model transformer (runtime layer insertion)
│   ├── strategies.py            # Injection strategies (random, bit-flip, etc.)
│   └── statistics.py            # Fault statistics tracking and reporting
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         YAML Config                              │
│  fault_injection:                                                │
│    enabled: true                                                 │
│    probability: 5.0                                              │
│    ...                                                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FaultInjectionConfig                          │
│  - Parses and validates YAML configuration                       │
│  - Provides typed access to all parameters                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FaultInjector                              │
│  - Analyzes model structure                                      │
│  - Identifies injection points (after QuantIdentity, etc.)       │
│  - Inserts FaultInjectionLayer instances at runtime              │
│  - Updates parameters per epoch/iteration                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FaultInjectionLayer                           │
│  - Receives QuantTensor from previous layer                      │
│  - Applies fault injection based on probability                  │
│  - Uses selected InjectionStrategy                               │
│  - Tracks statistics (optional)                                  │
│  - Returns modified QuantTensor                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    InjectionStrategy                             │
│  - RandomStrategy: Random value in valid range                   │
│  - LSBFlipStrategy: Flip least significant bit                   │
│  - MSBFlipStrategy: Flip most significant bit                    │
│  - FullFlipStrategy: Flip all bits                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FaultStatistics                               │
│  - Tracks per-layer injection counts                             │
│  - Computes RMSE between clean/faulty outputs                    │
│  - Computes cosine similarity                                    │
│  - Generates reports                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Configuration (`utils/fault_injection/config.py`)

A dataclass that holds all fault injection parameters with validation.

```python
@dataclass
class FaultInjectionConfig:
    """Configuration for fault injection.
    
    Attributes:
        enabled: Master switch for fault injection.
        probability: Injection probability as percentage (0-100).
        mode: Injection mode - "full_model" or "layer".
        injection_layer: Specific layer index for "layer" mode (-1 for random).
        injection_type: Type of fault - "random", "lsb_flip", "msb_flip", "full_flip".
        apply_during: When to inject - "train", "eval", or "both".
        epoch_interval: Re-generate fault mask every N epochs (training only).
        step_interval: Probability of injection per batch step (0-1).
        seed: Random seed for reproducible fault patterns.
        track_statistics: Enable statistics tracking (RMSE, cosine similarity).
        verbose: Print injection details.
    """
    enabled: bool = False
    probability: float = 0.0
    mode: str = "full_model"
    injection_layer: int = -1
    injection_type: str = "random"
    apply_during: str = "eval"
    epoch_interval: int = 1
    step_interval: float = 0.5
    seed: Optional[int] = None
    track_statistics: bool = False
    verbose: bool = False
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FaultInjectionConfig":
        """Create config from dictionary (YAML)."""
        
    def validate(self) -> None:
        """Validate configuration values."""
```

**YAML Schema:**

```yaml
fault_injection:
  # Master switch
  enabled: false
  
  # Injection probability (0-100%)
  probability: 0.0
  
  # Injection mode:
  # - "full_model": Inject faults in all layers
  # - "layer": Inject faults in a specific layer
  mode: "full_model"
  
  # Layer index for "layer" mode (-1 = random layer selection)
  injection_layer: -1
  
  # Injection type:
  # - "random": Replace with random value in valid range
  # - "lsb_flip": Flip least significant bit
  # - "msb_flip": Flip most significant bit  
  # - "full_flip": Flip all bits
  injection_type: "random"
  
  # When to apply injection:
  # - "train": Only during training
  # - "eval": Only during evaluation
  # - "both": During both training and evaluation
  apply_during: "eval"
  
  # Training-specific settings (for Fault-Aware Training)
  epoch_interval: 1        # Re-generate fault mask every N epochs
  step_interval: 0.5       # Probability of injection per batch (0-1)
  
  # Reproducibility
  seed: null               # Random seed (null = non-deterministic)
  
  # Statistics and logging
  track_statistics: false  # Track RMSE, cosine similarity per layer
  verbose: false           # Print detailed injection info
```

---

### 2. Fault Injection Layers (`utils/fault_injection/layers.py`)

Two layer implementations for quantized models:

#### 2.1 `QuantFaultInjectionLayer`

For use after layers that output `QuantTensor` (e.g., `QuantReLU`, `QuantIdentity`).

```python
class QuantFaultInjectionLayer(nn.Module):
    """Fault injection layer for Brevitas QuantTensor outputs.
    
    This layer intercepts QuantTensor activations and injects faults
    based on the configured probability and injection strategy.
    
    Attributes:
        layer_id: Unique identifier for this injection layer.
        probability: Injection probability (0-100).
        injection_layer: Target layer for "layer" mode.
        num_layers: Total number of injection layers in model.
        strategy: Injection strategy instance.
        epoch: Current training epoch.
        counter: Iteration counter within epoch.
        statistics: Optional statistics tracker.
    """
    
    def __init__(
        self,
        layer_id: int = 0,
        probability: float = 0.0,
        injection_layer: int = 0,
        num_layers: int = 0,
        strategy: "InjectionStrategy" = None,
        config: FaultInjectionConfig = None,
    ):
        ...
    
    def forward(self, x: QuantTensor) -> QuantTensor:
        """Forward pass with optional fault injection.
        
        Args:
            x: Input QuantTensor from previous layer.
            
        Returns:
            QuantTensor with injected faults (if enabled).
        """
        # Skip injection if probability is 0
        if self.probability == 0.0:
            return x
        
        # Check if this layer should inject (based on mode)
        if not self._should_inject():
            return x
        
        # Extract quantization parameters
        scale = x.scale
        zero_point = x.zero_point
        bit_width = x.bit_width
        signed = x.signed
        
        # Generate fault mask based on mode (train/eval)
        fault_mask = self._generate_fault_mask(x.shape)
        
        # Apply injection strategy
        faulty_values = self.strategy.inject(x, fault_mask, bit_width, signed)
        
        # Track statistics if enabled
        if self.statistics is not None:
            self.statistics.record(x, faulty_values, self.layer_id)
        
        # Reconstruct QuantTensor
        return QuantTensor(
            value=faulty_values,
            scale=scale,
            zero_point=zero_point,
            bit_width=bit_width,
            signed=signed,
            training=x.training,
        )
    
    def _should_inject(self) -> bool:
        """Check if this layer should inject faults."""
        if self.mode == "full_model":
            return True
        elif self.mode == "layer":
            return self.layer_id == self.injection_layer
        return False
    
    def _generate_fault_mask(self, shape: torch.Size) -> torch.Tensor:
        """Generate boolean mask for fault injection."""
        ...
```

#### 2.2 `ErrInjLayer` (Compatibility Layer)

Maintains API compatibility with existing `fat/src/` implementation:

```python
class ErrInjLayer(QuantFaultInjectionLayer):
    """Compatibility wrapper matching fat/src/err_inj_layer API."""
    
    def __init__(
        self,
        p: float = 0.0,
        layer_id: int = 0,
        injection_layer: int = 0,
        counter: int = 0,
        num_layers: int = 0,
        epoch: int = 0,
        num_iterations: int = 0,
        condition_injector: torch.Tensor = None,
        **kwargs,
    ):
        ...
```

---

### 3. Model Injector (`utils/fault_injection/injector.py`)

The core component that transforms models at runtime.

```python
class FaultInjector:
    """Transforms models to add fault injection layers at runtime.
    
    This class analyzes a model's structure and inserts fault injection
    layers after target layers (QuantIdentity, QuantReLU, etc.) without
    modifying the original model definition.
    
    Example:
        ```python
        injector = FaultInjector()
        config = FaultInjectionConfig(enabled=True, probability=5.0)
        model = injector.inject(model, config)
        
        # Later, update parameters
        injector.update_epoch(model, epoch=10)
        
        # Remove injection layers
        model = injector.remove(model)
        ```
    """
    
    # Layer types to inject after
    QUANT_TARGET_LAYERS = (
        "QuantIdentity",
        "QuantReLU", 
        "QuantHardTanh",
    )
    
    def inject(
        self, 
        model: nn.Module, 
        config: FaultInjectionConfig,
    ) -> nn.Module:
        """Add fault injection layers to a model.
        
        Walks the model graph and inserts QuantFaultInjectionLayer
        instances after each target layer.
        
        Args:
            model: The model to transform.
            config: Fault injection configuration.
            
        Returns:
            Transformed model with injection layers.
        """
        ...
    
    def remove(self, model: nn.Module) -> nn.Module:
        """Remove all fault injection layers from a model.
        
        Args:
            model: Model with injection layers.
            
        Returns:
            Original model without injection layers.
        """
        ...
    
    def update_epoch(self, model: nn.Module, epoch: int) -> None:
        """Update epoch for all injection layers.
        
        Args:
            model: Model with injection layers.
            epoch: Current epoch number.
        """
        ...
    
    def update_probability(
        self, 
        model: nn.Module, 
        probability: float,
        layer_id: Optional[int] = None,
    ) -> None:
        """Update injection probability.
        
        Args:
            model: Model with injection layers.
            probability: New probability (0-100).
            layer_id: Specific layer to update (None = all layers).
        """
        ...
    
    def reset_counters(self, model: nn.Module) -> None:
        """Reset iteration counters for all injection layers."""
        ...
    
    def set_mode(self, model: nn.Module, mode: str) -> None:
        """Set injection mode (train/eval)."""
        ...
    
    def get_num_layers(self, model: nn.Module) -> int:
        """Count the number of injection layers in a model."""
        ...
    
    def get_injection_layers(
        self, 
        model: nn.Module,
    ) -> List[QuantFaultInjectionLayer]:
        """Get all injection layer instances."""
        ...
```

**Injection Process:**

The injector uses `torch.fx` or direct module replacement to insert layers:

```python
def _inject_sequential(self, module: nn.Sequential, config: FaultInjectionConfig) -> nn.Sequential:
    """Inject into a Sequential module."""
    new_modules = []
    layer_id = 0
    
    for name, child in module.named_children():
        new_modules.append((name, child))
        
        # Check if we should inject after this layer
        if child.__class__.__name__ in self.QUANT_TARGET_LAYERS:
            inj_layer = QuantFaultInjectionLayer(
                layer_id=layer_id,
                probability=config.probability,
                num_layers=self._total_layers,
                strategy=self._create_strategy(config),
                config=config,
            )
            new_modules.append((f"fault_inj_{layer_id}", inj_layer))
            layer_id += 1
    
    return nn.Sequential(OrderedDict(new_modules))
```

---

### 4. Injection Strategies (`utils/fault_injection/strategies.py`)

Pluggable strategies for different fault types:

```python
class InjectionStrategy(ABC):
    """Abstract base class for injection strategies."""
    
    @abstractmethod
    def inject(
        self,
        tensor: QuantTensor,
        mask: torch.Tensor,
        bit_width: torch.Tensor,
        signed: bool,
    ) -> torch.Tensor:
        """Apply fault injection to tensor values.
        
        Args:
            tensor: Original QuantTensor.
            mask: Boolean mask indicating which values to inject.
            bit_width: Quantization bit width.
            signed: Whether quantization is signed.
            
        Returns:
            Tensor with injected faults (scaled float values).
        """
        pass


class RandomStrategy(InjectionStrategy):
    """Replace values with random integers in valid range.
    
    For unsigned N-bit: [1, 2^N]
    For signed N-bit: [-(2^(N-1)) + 1, 2^(N-1) - 1]
    """
    
    def inject(self, tensor, mask, bit_width, signed):
        scale = tensor.scale
        int_tensor = torch.round(tensor.value / scale).to(torch.int32)
        
        if signed:
            min_val = -(2 ** (bit_width - 1)).int() + 1
            max_val = (2 ** (bit_width - 1)).int() - 1
        else:
            min_val = 1
            max_val = (2 ** bit_width).int()
        
        # Generate random values
        rand_tensor = torch.randint(min_val, max_val + 1, tensor.shape, device=tensor.device)
        
        # Apply modular addition (matches existing implementation)
        range_size = max_val - min_val + 1
        injected = ((int_tensor + rand_tensor - min_val) % range_size) + min_val
        
        # Apply mask and scale back to float
        result = torch.where(mask, injected * scale, tensor.value)
        return result


class LSBFlipStrategy(InjectionStrategy):
    """Flip the least significant bit."""
    
    def inject(self, tensor, mask, bit_width, signed):
        scale = tensor.scale
        int_tensor = torch.round(tensor.value / scale).to(torch.int32)
        
        # XOR with 1 to flip LSB
        flipped = int_tensor ^ 1
        
        result = torch.where(mask, flipped * scale, tensor.value)
        return result


class MSBFlipStrategy(InjectionStrategy):
    """Flip the most significant bit."""
    
    def inject(self, tensor, mask, bit_width, signed):
        scale = tensor.scale
        int_tensor = torch.round(tensor.value / scale).to(torch.int32)
        
        # XOR with MSB position
        msb_mask = 1 << (bit_width.int().item() - 1)
        flipped = int_tensor ^ msb_mask
        
        result = torch.where(mask, flipped * scale, tensor.value)
        return result


class FullFlipStrategy(InjectionStrategy):
    """Flip all bits (bitwise NOT within valid range)."""
    
    def inject(self, tensor, mask, bit_width, signed):
        scale = tensor.scale
        int_tensor = torch.round(tensor.value / scale).to(torch.int32)
        
        # XOR with all 1s in bit_width range
        all_ones = (1 << bit_width.int().item()) - 1
        flipped = int_tensor ^ all_ones
        
        result = torch.where(mask, flipped * scale, tensor.value)
        return result


def get_strategy(name: str) -> InjectionStrategy:
    """Factory function to get strategy by name."""
    strategies = {
        "random": RandomStrategy,
        "lsb_flip": LSBFlipStrategy,
        "msb_flip": MSBFlipStrategy,
        "full_flip": FullFlipStrategy,
    }
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
    return strategies[name]()
```

---

### 5. Statistics Tracking (`utils/fault_injection/statistics.py`)

Track and report fault injection statistics:

```python
@dataclass
class LayerStatistics:
    """Statistics for a single layer."""
    layer_id: int
    total_activations: int = 0
    injected_count: int = 0
    rmse_sum: float = 0.0
    cosine_similarity_sum: float = 0.0
    sample_count: int = 0
    
    @property
    def injection_rate(self) -> float:
        """Actual injection rate as percentage."""
        if self.total_activations == 0:
            return 0.0
        return 100.0 * self.injected_count / self.total_activations
    
    @property
    def avg_rmse(self) -> float:
        """Average RMSE across samples."""
        if self.sample_count == 0:
            return 0.0
        return self.rmse_sum / self.sample_count
    
    @property
    def avg_cosine_similarity(self) -> float:
        """Average cosine similarity across samples."""
        if self.sample_count == 0:
            return 0.0
        return self.cosine_similarity_sum / self.sample_count


class FaultStatistics:
    """Tracks fault injection statistics across layers.
    
    Computes:
    - Number of injected faults per layer
    - RMSE between clean and faulty outputs
    - Cosine similarity between clean and faulty outputs
    - Percentage of different values
    
    Example:
        ```python
        stats = FaultStatistics(num_layers=10)
        
        # During forward pass
        stats.record(clean_tensor, faulty_tensor, layer_id=0)
        
        # After evaluation
        stats.print_report()
        stats.save_to_file("fault_stats.json")
        ```
    """
    
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.layer_stats: Dict[int, LayerStatistics] = {
            i: LayerStatistics(layer_id=i) for i in range(num_layers)
        }
        self.enabled = True
    
    def record(
        self,
        clean: QuantTensor,
        faulty: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Record statistics for a forward pass.
        
        Args:
            clean: Original (clean) QuantTensor.
            faulty: Faulty tensor values.
            layer_id: Layer identifier.
        """
        if not self.enabled:
            return
        
        stats = self.layer_stats[layer_id]
        
        # Count differences
        clean_int = clean.int()
        faulty_int = torch.round(faulty / clean.scale).to(torch.int32)
        different_mask = clean_int != faulty_int
        
        stats.total_activations += clean_int.numel()
        stats.injected_count += different_mask.sum().item()
        
        # Compute RMSE (only on different values)
        if different_mask.any():
            diff = (clean_int[different_mask].float() - faulty_int[different_mask].float())
            rmse = torch.sqrt(torch.mean(diff ** 2)).item()
            stats.rmse_sum += rmse
            
            # Compute cosine similarity
            clean_flat = clean_int.flatten().float()
            faulty_flat = faulty_int.flatten().float()
            cos_sim = torch.nn.functional.cosine_similarity(
                clean_flat.unsqueeze(0), 
                faulty_flat.unsqueeze(0)
            ).item()
            stats.cosine_similarity_sum += cos_sim
            
            stats.sample_count += 1
    
    def reset(self) -> None:
        """Reset all statistics."""
        for stats in self.layer_stats.values():
            stats.total_activations = 0
            stats.injected_count = 0
            stats.rmse_sum = 0.0
            stats.cosine_similarity_sum = 0.0
            stats.sample_count = 0
    
    def print_report(self) -> None:
        """Print statistics report to console."""
        print("\n" + "=" * 80)
        print("FAULT INJECTION STATISTICS")
        print("=" * 80)
        print(f"{'Layer':<8} {'Injected':<12} {'Total':<12} {'Rate (%)':<10} {'RMSE':<10} {'Cos Sim':<10}")
        print("-" * 80)
        
        total_injected = 0
        total_activations = 0
        
        for layer_id in sorted(self.layer_stats.keys()):
            stats = self.layer_stats[layer_id]
            total_injected += stats.injected_count
            total_activations += stats.total_activations
            
            print(
                f"{layer_id:<8} "
                f"{stats.injected_count:<12} "
                f"{stats.total_activations:<12} "
                f"{stats.injection_rate:<10.2f} "
                f"{stats.avg_rmse:<10.2f} "
                f"{stats.avg_cosine_similarity * 100:<10.2f}"
            )
        
        print("-" * 80)
        overall_rate = 100.0 * total_injected / total_activations if total_activations > 0 else 0
        print(f"{'TOTAL':<8} {total_injected:<12} {total_activations:<12} {overall_rate:<10.2f}")
        print("=" * 80 + "\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export statistics as dictionary."""
        ...
    
    def save_to_file(self, path: str) -> None:
        """Save statistics to JSON file."""
        ...
```

---

### 6. Trainer Integration

Modify `utils/trainer.py` to support fault injection:

```python
class Trainer:
    def __init__(self, ...):
        ...
        
        # Fault injection setup
        self.fault_injector: Optional[FaultInjector] = None
        self.fault_config: Optional[FaultInjectionConfig] = None
        self.fault_stats: Optional[FaultStatistics] = None
        
        fi_config = config.get("fault_injection", {})
        if fi_config.get("enabled", False):
            self.fault_config = FaultInjectionConfig.from_dict(fi_config)
            self.fault_injector = FaultInjector()
            self.model = self.fault_injector.inject(self.model, self.fault_config)
            
            if self.fault_config.track_statistics:
                num_layers = self.fault_injector.get_num_layers(self.model)
                self.fault_stats = FaultStatistics(num_layers)
                self.fault_injector.set_statistics(self.model, self.fault_stats)
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        # Update fault injection parameters for this epoch
        if self.fault_injector is not None:
            self.fault_injector.update_epoch(self.model, epoch)
            self.fault_injector.reset_counters(self.model)
            
            # Set mode based on config
            if self.fault_config.apply_during in ("train", "both"):
                self.fault_injector.set_enabled(self.model, True)
            else:
                self.fault_injector.set_enabled(self.model, False)
        
        ... # existing training code
    
    def evaluate(self, loader, desc="Evaluating") -> Tuple[float, float]:
        # Enable/disable injection for evaluation
        if self.fault_injector is not None:
            if self.fault_config.apply_during in ("eval", "both"):
                self.fault_injector.set_enabled(self.model, True)
            else:
                self.fault_injector.set_enabled(self.model, False)
        
        ... # existing evaluation code
        
        # Print statistics after evaluation
        if self.fault_stats is not None and self.fault_config.apply_during in ("eval", "both"):
            self.fault_stats.print_report()
```

---

### 7. Evaluation Script (`evaluate.py`)

Create a dedicated evaluation script for fault injection experiments:

```python
"""Evaluation script for fault injection experiments.

Runs fault injection evaluation on trained models with configurable
fault parameters. Supports multiple injection runs with different
seeds for statistical analysis.

Usage:
    # Single evaluation
    python evaluate.py --config configs/eval_fault.yaml --checkpoint path/to/model.pth
    
    # Multiple runs with different seeds
    python evaluate.py --config configs/eval_fault.yaml --checkpoint path/to/model.pth --runs 10
    
    # Sweep over probabilities
    python evaluate.py --config configs/eval_fault.yaml --checkpoint path/to/model.pth --sweep-prob 0,1,2,5,10
"""

import argparse
from typing import List

from datasets import get_dataset
from models import get_model
from utils import get_device, load_config, set_seed
from utils.fault_injection import FaultInjector, FaultInjectionConfig, FaultStatistics


def evaluate_with_faults(
    model: nn.Module,
    test_loader: DataLoader,
    config: FaultInjectionConfig,
    device: torch.device,
    seed: int = None,
) -> Tuple[float, FaultStatistics]:
    """Run evaluation with fault injection.
    
    Args:
        model: Trained model.
        test_loader: Test data loader.
        config: Fault injection configuration.
        device: Compute device.
        seed: Random seed for reproducible faults.
        
    Returns:
        Tuple of (accuracy, statistics).
    """
    ...


def main():
    parser = argparse.ArgumentParser(description="Fault Injection Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--runs", type=int, default=1, help="Number of evaluation runs")
    parser.add_argument("--sweep-prob", type=str, default=None, 
                        help="Comma-separated probabilities to sweep")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = get_device()
    
    # Load dataset
    dataset = get_dataset(config)
    _, _, test_loader = dataset.get_loaders()
    
    # Load model
    model = get_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Parse probability sweep
    if args.sweep_prob:
        probabilities = [float(p) for p in args.sweep_prob.split(",")]
    else:
        probabilities = [config["fault_injection"]["probability"]]
    
    results = []
    
    for prob in probabilities:
        config["fault_injection"]["probability"] = prob
        
        for run in range(args.runs):
            seed = run if args.runs > 1 else config["fault_injection"].get("seed")
            
            acc, stats = evaluate_with_faults(
                model, test_loader, 
                FaultInjectionConfig.from_dict(config["fault_injection"]),
                device, seed
            )
            
            results.append({
                "probability": prob,
                "run": run,
                "seed": seed,
                "accuracy": acc,
                "injection_rate": stats.overall_injection_rate,
            })
            
            print(f"Prob: {prob}%, Run: {run}, Acc: {acc:.2f}%")
    
    # Save results
    if args.output:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `utils/fault_injection/__init__.py` | Create | Public API exports |
| `utils/fault_injection/config.py` | Create | FaultInjectionConfig dataclass |
| `utils/fault_injection/layers.py` | Create | QuantFaultInjectionLayer, ErrInjLayer |
| `utils/fault_injection/injector.py` | Create | FaultInjector class |
| `utils/fault_injection/strategies.py` | Create | Injection strategy classes |
| `utils/fault_injection/statistics.py` | Create | FaultStatistics class |
| `utils/__init__.py` | Modify | Export fault injection components |
| `utils/trainer.py` | Modify | Integrate fault injection |
| `evaluate.py` | Create | Dedicated evaluation script |
| `configs/fault_injection_example.yaml` | Create | Example configuration |
| `docs/configuration/fault_injection.md` | Create | Documentation |

---

## Example Configurations

### Fault-Aware Training (FAT)

```yaml
# configs/fat_resnet20_cifar10.yaml
seed:
  enabled: true
  value: 42
  deterministic: true

dataset:
  name: "cifar10"
  root: "./data"
  download: true
  num_workers: 8

model:
  name: "quant_resnet20"

quantization:
  weight_bit_width: 4
  act_bit_width: 4

training:
  batch_size: 128
  epochs: 200

optimizer:
  name: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0001

scheduler:
  name: "cosine"
  T_max: 200

fault_injection:
  enabled: true
  probability: 5.0
  mode: "full_model"
  injection_type: "random"
  apply_during: "train"
  epoch_interval: 1
  step_interval: 0.5
  track_statistics: false
  verbose: false

checkpoint:
  enabled: true
  dir: "./experiments"
  save_best: true
```

### Fault Evaluation

```yaml
# configs/eval_fault.yaml
seed:
  enabled: true
  value: 42

dataset:
  name: "cifar10"
  root: "./data"
  download: true
  num_workers: 8

model:
  name: "quant_resnet20"

quantization:
  weight_bit_width: 4
  act_bit_width: 4

training:
  batch_size: 128
  epochs: 1  # Not used for evaluation

fault_injection:
  enabled: true
  probability: 5.0
  mode: "full_model"
  injection_type: "random"
  apply_during: "eval"
  seed: 42
  track_statistics: true
  verbose: true
```

---

## Implementation Order

1. **Phase 1: Core Infrastructure** ✅
   - [x] `utils/fault_injection/config.py` - Configuration dataclass
   - [x] `utils/fault_injection/strategies.py` - Injection strategies
   - [x] `utils/fault_injection/layers.py` - Fault injection layers

2. **Phase 2: Model Transformation** ✅
   - [x] `utils/fault_injection/injector.py` - Model injector
   - [x] `utils/fault_injection/statistics.py` - Statistics tracking
   - [x] `utils/fault_injection/__init__.py` - Public exports

3. **Phase 3: Integration** ✅
   - [x] Modify `utils/__init__.py` - Export new components
   - [x] Modify `utils/trainer.py` - Integrate fault injection
   - [x] Create `evaluate.py` - Evaluation script

4. **Phase 4: Documentation & Examples** ✅
   - [x] Create example configurations (`configs/fault_injection_example.yaml`)
   - [x] Create `docs/configuration/fault_injection.md`
   - [x] Update `mkdocs.yml` navigation

---

## Testing Plan

1. **Unit Tests**
   - Test each injection strategy produces valid outputs
   - Test FaultInjectionConfig validation
   - Test statistics calculations (RMSE, cosine similarity)

2. **Integration Tests**
   - Test injection into QuantCNV model
   - Test injection into QuantResNet model
   - Verify model output shape unchanged after injection

3. **End-to-End Tests**
   - Train with FAT enabled, verify convergence
   - Evaluate with different probabilities, verify accuracy degradation
   - Compare results with existing `fat/src/` implementation

---

## Open Questions (Resolved)

These features were addressed:

1. **Hardware Mask Support**: The existing implementation has `hw_mask` and `frequency_value` options. *IMPLEMENTED - Added `hw_mask` and `frequency_value` parameters with `HardwareMaskGenerator` class for hardware-aware periodic fault patterns.*

2. **Gradient Handling**: The existing code has commented-out gradient hooks (`zero_gradients_hook`). *Deferred - STE (Straight-Through Estimator) is used instead for gradient flow.*

3. **Clean Tensor Saving**: The existing code can save clean tensors for later comparison. *Deferred - statistics tracking provides RMSE and cosine similarity instead.*

---

## Approval Checklist (Completed)

All items were reviewed and implemented:

- [x] Directory structure is acceptable
- [x] YAML configuration schema meets requirements
- [x] Layer injection approach (runtime transformation) is acceptable
- [x] Statistics tracking metrics are sufficient
- [x] Evaluation script features are adequate
- [x] Implementation order makes sense

~~Once approved, I will begin implementation starting with Phase 1.~~

**Implementation completed December 2025.**
