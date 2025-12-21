"""Full flip fault injection strategy.

Flips all bits (bitwise NOT).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import InjectionStrategy


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