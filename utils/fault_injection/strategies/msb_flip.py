"""MSB flip fault injection strategy.

Flips the most significant bit.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import InjectionStrategy


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