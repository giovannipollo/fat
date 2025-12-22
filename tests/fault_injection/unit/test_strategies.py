"""Unit tests for fault injection strategies."""

from __future__ import annotations

import pytest
import torch

from utils.fault_injection.strategies import (
    FullFlipStrategy,
    LSBFlipStrategy,
    MSBFlipStrategy,
    RandomStrategy,
    get_strategy,
)


class TestStrategies:
    """Test suite for fault injection strategies."""

    @pytest.mark.parametrize("strategy_class", [
        LSBFlipStrategy,
        MSBFlipStrategy,
        FullFlipStrategy,
        RandomStrategy,
    ])
    
    def test_strategy_changes_tensor_where_mask_true(
        self, strategy_class, sample_int_tensor, sample_mask, device
    ):
        """Test that each strategy modifies the tensor where mask is True."""
        strategy = strategy_class()
        original = sample_int_tensor.clone()
        bit_width = 4
        signed = True

        result = strategy.inject(
            int_tensor=original,
            mask=sample_mask,
            bit_width=bit_width,
            signed=signed,
            device=device,
        )

        # Check that result differs from original where mask is True
        masked_original = original[sample_mask]
        masked_result = result[sample_mask]
        assert not torch.equal(masked_original, masked_result), (
            f"{strategy_class.__name__} did not change tensor where mask is True"
        )

        # Check that result is unchanged where mask is False
        unmasked_original = original[~sample_mask]
        unmasked_result = result[~sample_mask]
        assert torch.equal(unmasked_original, unmasked_result), (
            f"{strategy_class.__name__} changed tensor where mask is False"
        )

    def test_lsb_flip_strategy_specific_behavior(
        self, sample_int_tensor, sample_mask, device
    ):
        """Test LSB flip strategy flips the least significant bit."""
        strategy = LSBFlipStrategy()
        original = sample_int_tensor.clone()
        bit_width = 4
        signed = True

        result = strategy.inject(
            int_tensor=original,
            mask=sample_mask,
            bit_width=bit_width,
            signed=signed,
            device=device,
        )

        # LSB flip: XOR with 1
        expected_flipped = original ^ 1
        expected = torch.where(sample_mask, expected_flipped, original)

        assert torch.equal(result, expected), "LSB flip did not match expected XOR with 1"

    def test_msb_flip_strategy_specific_behavior(
        self, sample_int_tensor, sample_mask, device
    ):
        """Test MSB flip strategy flips the most significant bit."""
        strategy = MSBFlipStrategy()
        original = sample_int_tensor.clone()
        bit_width = 4
        signed = True

        result = strategy.inject(
            int_tensor=original,
            mask=sample_mask,
            bit_width=bit_width,
            signed=signed,
            device=device,
        )

        # MSB for 4-bit signed: bit 3 (8 in decimal)
        msb_mask = 1 << (bit_width - 1)  # 8 for bit_width=4
        expected_flipped = original ^ msb_mask
        expected = torch.where(sample_mask, expected_flipped, original)

        assert torch.equal(result, expected), f"MSB flip did not match expected XOR with {msb_mask}"

    def test_full_flip_strategy_specific_behavior(
        self, sample_int_tensor, sample_mask, device
    ):
        """Test full flip strategy flips all bits."""
        strategy = FullFlipStrategy()
        original = sample_int_tensor.clone()
        bit_width = 4
        signed = True

        result = strategy.inject(
            int_tensor=original,
            mask=sample_mask,
            bit_width=bit_width,
            signed=signed,
            device=device,
        )

        # Full flip: XOR with all-ones mask
        full_mask = (1 << bit_width) - 1  # 15 for 4 bits
        expected_flipped = original ^ full_mask
        expected = torch.where(sample_mask, expected_flipped, original)

        assert torch.equal(result, expected), f"Full flip did not match expected XOR with {full_mask}"

    def test_random_strategy_changes_within_range(
        self, sample_int_tensor, sample_mask, device
    ):
        """Test random strategy changes values within valid range."""
        strategy = RandomStrategy()
        original = sample_int_tensor.clone()
        bit_width = 4
        signed = True

        result = strategy.inject(
            int_tensor=original,
            mask=sample_mask,
            bit_width=bit_width,
            signed=signed,
            device=device,
        )

        # Check values are within range
        min_val, max_val, _ = strategy._get_value_range(bit_width, signed)
        assert torch.all(result >= min_val), f"Random strategy produced value below {min_val}"
        assert torch.all(result <= max_val), f"Random strategy produced value above {max_val}"

        # Check changes occurred where mask is True
        masked_original = original[sample_mask]
        masked_result = result[sample_mask]
        assert not torch.equal(masked_original, masked_result), (
            "Random strategy did not change tensor where mask is True"
        )

    def test_get_strategy_factory(self):
        """Test strategy factory function."""
        assert isinstance(get_strategy("random"), RandomStrategy)
        assert isinstance(get_strategy("lsb_flip"), LSBFlipStrategy)
        assert isinstance(get_strategy("msb_flip"), MSBFlipStrategy)
        assert isinstance(get_strategy("full_flip"), FullFlipStrategy)

        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("invalid")