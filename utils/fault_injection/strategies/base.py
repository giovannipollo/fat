"""Base injection strategy class.

Provides the abstract base class for all injection strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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
            min_val = -(2 ** (bit_width - 1))
            max_val = (2 ** (bit_width - 1)) - 1
        else:
            min_val = 0
            max_val = (2 ** bit_width) - 1

        range_size = max_val - min_val + 1
        return min_val, max_val, range_size