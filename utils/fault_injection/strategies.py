"""Injection strategies for fault injection.

Provides different strategies for injecting faults into quantized tensors,
including random value replacement and bit-flip operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

import torch
from torch import Tensor


class InjectionStrategy(ABC):
    """Abstract base class for injection strategies.

    Subclasses implement specific fault injection mechanisms such as
    random value replacement or bit flipping.

    Example:
        ```python
        strategy = RandomStrategy()
        faulty_int = strategy.inject(
            int_tensor=quantized_values,
            mask=fault_mask,
            bit_width=8,
            signed=True,
            device=torch.device("cuda"),
        )
        ```
    """

    @abstractmethod
    def inject(
        self,
        int_tensor: Tensor,
        mask: Tensor,
        bit_width: int,
        signed: bool,
        device: torch.device,
    ) -> Tensor:
        """Apply fault injection to integer tensor values.

        Args:
            int_tensor: Quantized integer tensor values.
            mask: Boolean mask indicating which values to inject.
            bit_width: Quantization bit width.
            signed: Whether quantization is signed.
            device: Device to create tensors on.

        Returns:
            Integer tensor with injected faults.
        """
        pass

    def _get_value_range(
        self, bit_width: int, signed: bool
    ) -> tuple[int, int, int]:
        """Get the valid value range for quantization.

        Args:
            bit_width: Number of bits.
            signed: Whether values are signed.

        Returns:
            Tuple of (min_value, max_value, range_size).
        """
        if signed:
            min_val = -(2 ** (bit_width - 1)) + 1
            max_val = (2 ** (bit_width - 1)) - 1
        else:
            min_val = 1
            max_val = 2 ** bit_width

        range_size = max_val - min_val + 1
        return min_val, max_val, range_size


class RandomStrategy(InjectionStrategy):
    """Replace values with random integers using modular addition.

    This strategy adds a random value to each selected element and wraps
    the result using modular arithmetic to stay within the valid range.

    For unsigned N-bit: values in [1, 2^N]
    For signed N-bit: values in [-(2^(N-1)) + 1, 2^(N-1) - 1]
    """

    def inject(
        self,
        int_tensor: Tensor,
        mask: Tensor,
        bit_width: int,
        signed: bool,
        device: torch.device,
    ) -> Tensor:
        """Apply random fault injection using modular addition.

        Args:
            int_tensor: Quantized integer tensor values.
            mask: Boolean mask indicating which values to inject.
            bit_width: Quantization bit width.
            signed: Whether quantization is signed.
            device: Device to create tensors on.

        Returns:
            Integer tensor with random faults injected.
        """
        min_val, max_val, range_size = self._get_value_range(bit_width, signed)

        # Generate random values to add
        rand_tensor = torch.randint(
            low=min_val,
            high=max_val + 1,
            size=int_tensor.shape,
            device=device,
            dtype=torch.int32,
        )

        # Apply modular addition to wrap values within range
        modular_result = ((int_tensor + rand_tensor - min_val) % range_size) + min_val

        # Apply mask: only inject where mask is True
        result = torch.where(mask, modular_result, int_tensor)
        return result


class LSBFlipStrategy(InjectionStrategy):
    """Flip the least significant bit.

    This strategy flips bit 0 of each selected value, effectively
    toggling between even and odd values.
    """

    def inject(
        self,
        int_tensor: Tensor,
        mask: Tensor,
        bit_width: int,
        signed: bool,
        device: torch.device,
    ) -> Tensor:
        """Apply LSB flip fault injection.

        Args:
            int_tensor: Quantized integer tensor values.
            mask: Boolean mask indicating which values to inject.
            bit_width: Quantization bit width.
            signed: Whether quantization is signed.
            device: Device to create tensors on.

        Returns:
            Integer tensor with LSB flipped where mask is True.
        """
        # XOR with 1 to flip LSB
        flipped = int_tensor ^ 1

        result = torch.where(mask, flipped, int_tensor)
        return result


class MSBFlipStrategy(InjectionStrategy):
    """Flip the most significant bit.

    This strategy flips the highest bit of each selected value,
    causing large magnitude changes in the output.
    """

    def inject(
        self,
        int_tensor: Tensor,
        mask: Tensor,
        bit_width: int,
        signed: bool,
        device: torch.device,
    ) -> Tensor:
        """Apply MSB flip fault injection.

        Args:
            int_tensor: Quantized integer tensor values.
            mask: Boolean mask indicating which values to inject.
            bit_width: Quantization bit width.
            signed: Whether quantization is signed.
            device: Device to create tensors on.

        Returns:
            Integer tensor with MSB flipped where mask is True.
        """
        # XOR with MSB position
        msb_mask = 1 << (bit_width - 1)
        flipped = int_tensor ^ msb_mask

        result = torch.where(mask, flipped, int_tensor)
        return result


class FullFlipStrategy(InjectionStrategy):
    """Flip all bits (bitwise NOT within valid range).

    This strategy inverts all bits of each selected value,
    effectively computing the bitwise complement.
    """

    def inject(
        self,
        int_tensor: Tensor,
        mask: Tensor,
        bit_width: int,
        signed: bool,
        device: torch.device,
    ) -> Tensor:
        """Apply full bit flip fault injection.

        Args:
            int_tensor: Quantized integer tensor values.
            mask: Boolean mask indicating which values to inject.
            bit_width: Quantization bit width.
            signed: Whether quantization is signed.
            device: Device to create tensors on.

        Returns:
            Integer tensor with all bits flipped where mask is True.
        """
        # XOR with all 1s in bit_width range
        all_ones = (1 << bit_width) - 1
        flipped = int_tensor ^ all_ones

        result = torch.where(mask, flipped, int_tensor)
        return result


# Registry of available strategies
_STRATEGIES: Dict[str, Type[InjectionStrategy]] = {
    "random": RandomStrategy,
    "lsb_flip": LSBFlipStrategy,
    "msb_flip": MSBFlipStrategy,
    "full_flip": FullFlipStrategy,
}


def get_strategy(name: str) -> InjectionStrategy:
    """Factory function to get strategy by name.

    Args:
        name: Strategy name ("random", "lsb_flip", "msb_flip", "full_flip").

    Returns:
        Instance of the requested strategy.

    Raises:
        ValueError: If strategy name is not recognized.

    Example:
        ```python
        strategy = get_strategy("random")
        ```
    """
    if name not in _STRATEGIES:
        available = list(_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: '{name}'. Available: {available}")

    return _STRATEGIES[name]()


def list_strategies() -> list[str]:
    """List all available strategy names.

    Returns:
        List of strategy names.
    """
    return list(_STRATEGIES.keys())
