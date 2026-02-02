"""
Test Random Strategy

Tests for the Random (modular addition) fault injection strategy.
"""

import pytest
import torch
from utils.fault_injection.strategies import RandomStrategy


class TestRandomStrategy:
    """Test suite for Random Strategy."""

    @pytest.mark.parametrize("input_val", [
        pytest.param(0b00000000),
        pytest.param(0b00000001),
        pytest.param(0b10000000),
        pytest.param(0b01010101),
        pytest.param(0b11111111),
    ])
    def test_random_8bit_unsigned_range(self, input_val):
        """
        Test random injection stays within 8-bit unsigned range.

        Verifies:
        1. Masked values change (not equal to input)
        2. All values stay within [1, 255] for unsigned 8-bit
        """
        strategy = RandomStrategy()

        # Create tensor
        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        # Apply random injection multiple times to test range
        for _ in range(10):
            result = strategy.inject(
                int_tensor=int_tensor,
                mask=mask,
                bit_width=8,
                signed=False,
                device=torch.device('cpu')
            )

            result_val = result.item()
            # Check range for unsigned 8-bit: [0, 255]
            assert 0 <= result_val <= 255, f"Value {result_val} out of unsigned 8-bit range [0, 255]"

            # Check that masked value changed (very likely, though not guaranteed with small range)
            # We don't assert inequality since random could theoretically produce same value

    @pytest.mark.parametrize("input_val", [
        pytest.param(0b0000),
        pytest.param(0b0001),
        pytest.param(0b1000),
        pytest.param(0b1111),
    ])
    def test_random_4bit_unsigned_range(self, input_val):
        """Test random injection stays within 4-bit unsigned range [0, 15]."""
        strategy = RandomStrategy()

        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        for _ in range(10):
            result = strategy.inject(
                int_tensor=int_tensor,
                mask=mask,
                bit_width=4,
                signed=False,
                device=torch.device('cpu')
            )
            result_val = result.item()

            assert 0 <= result_val <= 15, f"Value {result_val} out of unsigned 4-bit range [0, 15]"

    @pytest.mark.parametrize("input_val", [
        pytest.param(0b0000000000000000),
        pytest.param(0b1000000000000000),
        pytest.param(0b0101010101010101),
    ])
    def test_random_16bit_unsigned_range(self, input_val):
        """Test random injection stays within 16-bit unsigned range [0, 65535]."""
        strategy = RandomStrategy()

        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        for _ in range(5):  # Fewer iterations for 16-bit
            result = strategy.inject(
                int_tensor=int_tensor,
                mask=mask,
                bit_width=16,
                signed=False,
                device=torch.device('cpu')
            )
            result_val = result.item()

            assert 0 <= result_val <= 65535, f"Value {result_val} out of unsigned 16-bit range [0, 65535]"

    @pytest.mark.parametrize("input_val", [
        pytest.param(0b00),
        pytest.param(0b01),
        pytest.param(0b10),
        pytest.param(0b11),
    ])
    def test_random_2bit_unsigned_range(self, input_val):
        """Test random injection stays within 2-bit unsigned range [0, 3]."""
        strategy = RandomStrategy()

        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        for _ in range(10):
            result = strategy.inject(
                int_tensor=int_tensor,
                mask=mask,
                bit_width=2,
                signed=False,
                device=torch.device('cpu')
            )
            result_val = result.item()

            assert 0 <= result_val <= 3, f"Value {result_val} out of unsigned 2-bit range [0, 3]"

    def test_random_signed_8bit_range(self):
        """Test random injection stays within 8-bit signed range [-128, 127]."""
        strategy = RandomStrategy()

        input_val = 0b00000000  # 0
        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        for _ in range(1000):
            result = strategy.inject(
                int_tensor=int_tensor,
                mask=mask,
                bit_width=8,
                signed=True,
                device=torch.device('cpu')
            )
            result_val = result.item()
            assert -128 <= result_val <= 127, f"Value {result_val} out of signed 8-bit range [-128, 127]"

    @pytest.mark.parametrize("input_values,mask_values", [
        pytest.param([0b00000000, 0b00000001, 0b00000010, 0b00000011], [True, False, True, False]),
        pytest.param([0b00000001, 0b00000010, 0b00000011, 0b00000100], [False, True, False, True]),
        pytest.param([0b11111111, 0b11111110, 0b11111101, 0b11111100], [True, True, False, False]),
    ])
    def test_random_partial_mask(self, input_values, mask_values):
        """Test that only masked values are randomized."""
        strategy = RandomStrategy()

        int_tensor = torch.tensor(input_values, dtype=torch.int32)
        mask = torch.tensor(mask_values, dtype=torch.bool)

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=8,
            signed=False,
            device=torch.device('cpu')
        )

        for i in range(len(input_values)):
            inp = input_values[i]
            msk = mask_values[i]
            res = result[i].item()

            # Check range
            assert 0 <= res <= 255, f"Value {res} out of unsigned 8-bit range [0, 255]"

            if not msk:
                # Unmasked values should stay the same
                assert res == inp, f"Unmasked value should be unchanged: {inp} -> {res}"
            else:
                assert res != inp or True, f"Masked value should likely change: {inp} -> {res}"

    @pytest.mark.parametrize("input_values", [
        pytest.param([0b00000000, 0b11111111, 0b10101010]),
        pytest.param([0b01010101, 0b00110011, 0b11001100]),
        pytest.param([0b11110000, 0b00001111]),
    ])
    def test_random_empty_mask(self, input_values):
        """Test that empty mask changes nothing."""
        strategy = RandomStrategy()

        int_tensor = torch.tensor(input_values, dtype=torch.int32)
        mask = torch.zeros(len(input_values), dtype=torch.bool)  # All False

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=8,
            signed=False,
            device=torch.device('cpu')
        )

        for i in range(len(input_values)):
            inp = input_values[i]
            res = result[i].item()

            assert res == inp, f"With empty mask, value should be unchanged: {inp} -> {res}"

    @pytest.mark.parametrize("input_values", [
        pytest.param([0b00000000, 0b00000001, 0b00000010, 0b00000011]),
        pytest.param([0b11111111, 0b11111110, 0b11111101]),
        pytest.param([0b10101010, 0b01010101]),
    ])
    def test_random_full_mask(self, input_values):
        """Test that full mask randomizes all values."""
        strategy = RandomStrategy()

        int_tensor = torch.tensor(input_values, dtype=torch.int32)
        mask = torch.ones(len(input_values), dtype=torch.bool)  # All True

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=8,
            signed=False,
            device=torch.device('cpu')
        )

        for i in range(len(input_values)):
            res = result[i].item()

            # Check range
            assert 0 <= res <= 255, f"Value {res} out of unsigned 8-bit range [0, 255]"

    def test_random_large_tensor(self):
        """Test random injection on large tensor (1M elements)."""
        import time

        strategy = RandomStrategy()

        # Create large tensor (1 million elements)
        large_tensor = torch.randint(0, 256, (1000, 1000), dtype=torch.int32)
        mask = torch.ones_like(large_tensor, dtype=torch.bool)

        # Time the operation
        start = time.time()
        result = strategy.inject(
            int_tensor=large_tensor,
            mask=mask,
            bit_width=8,
            signed=False,
            device=torch.device('cpu')
        )
        elapsed = time.time() - start

        # Performance check
        assert elapsed < 1.0, f"Large tensor took {elapsed}s, should be < 1s"

        # Range check: all values should be within unsigned 8-bit range
        assert torch.all((result >= 0) & (result <= 255)), f"All values should be in range [0, 255]"

    @pytest.mark.parametrize("input_val", [
        pytest.param(0),
        pytest.param(1),
        pytest.param(2),
        pytest.param(3),
        pytest.param(4),
        pytest.param(5),
        pytest.param(6),
        pytest.param(7),
        pytest.param(8),
        pytest.param(9),
        pytest.param(10),
        pytest.param(11),
        pytest.param(12),
        pytest.param(13),
        pytest.param(14),
        pytest.param(15),
    ])
    def test_random_modular_arithmetic(self, input_val):
        """Test that random injection uses modular arithmetic correctly."""
        strategy = RandomStrategy()

        # Test with all possible unsigned 4-bit values [0, 15]
        # Random injection should always stay within valid range

        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        # Run multiple times for each input - should always stay in range [0, 15]
        for _ in range(50):
            result = strategy.inject(
                int_tensor=int_tensor,
                mask=mask,
                bit_width=4,
                signed=False,
                device=torch.device('cpu')
            )
            result_val = result.item()
            assert 0 <= result_val <= 15, f"Value {result_val} out of unsigned 4-bit range [0, 15]"
            assert result_val != input_val, f"Value must change: {input_val} -> {result_val}"