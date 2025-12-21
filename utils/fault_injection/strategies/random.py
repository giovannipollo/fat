"""Random fault injection strategy.

Replaces values with random integers using modular addition.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import InjectionStrategy


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