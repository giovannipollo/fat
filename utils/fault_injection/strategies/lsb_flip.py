"""LSB flip fault injection strategy.

Flips the least significant bit.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import InjectionStrategy


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