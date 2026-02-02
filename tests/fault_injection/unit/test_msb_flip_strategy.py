"""
Test MSB Flip Strategy

Tests for the MSB (Most Significant Bit) flip fault injection strategy.
"""

import pytest
import torch
from utils.fault_injection.strategies import MSBFlipStrategy


class TestMSBFlipStrategy:
    """Test suite for MSB Flip Strategy."""

    @pytest.mark.parametrize("input_val,expected_val", [
        pytest.param(0b00000000, 0b10000000),
        pytest.param(0b10000000, 0b00000000),
        pytest.param(0b00000001, 0b10000001),
        pytest.param(0b10000001, 0b00000001),
        pytest.param(0b01010101, 0b11010101),
        pytest.param(0b11010101, 0b01010101),
        pytest.param(0b01111111, 0b11111111),
        pytest.param(0b11111111, 0b01111111),
        pytest.param(0b10101010, 0b00101010),
        pytest.param(0b00101010, 0b10101010),
    ])
    def test_msb_flip_8bit_bitlevel(self, input_val, expected_val):
        """
        Test MSB flip at bit level.

        Verifies:
        1. Output = Input XOR 0b10000000
        2. Only bit 7 changes
        3. Bits 0-6 remain unchanged
        """
        strategy = MSBFlipStrategy()

        # Create tensor
        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        # Apply MSB flip
        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=8,
            signed=None,
            device=torch.device('cpu')
        )

        result_val = result.item()

        # Check result matches expected
        assert result_val == expected_val, f"Expected {expected_val}, got {result_val}"

        # Verify result = input XOR 128 (0b10000000)
        assert result_val == (input_val ^ 0b10000000), f"Result should equal input XOR 128"

        # Verify only bit 7 changed
        for bit_pos in range(8):
            input_bit = (input_val >> bit_pos) & 1
            result_bit = (result_val >> bit_pos) & 1

            if bit_pos == 7:
                # Bit 7 MUST flip
                assert input_bit != result_bit, f"Bit 7 must flip"
            else:
                # Bits 0-6 MUST NOT change
                assert input_bit == result_bit, f"Bit {bit_pos} must not change"

    @pytest.mark.parametrize("input_val,expected_val", [
        pytest.param(0b0000, 0b1000),
        pytest.param(0b1000, 0b0000),
        pytest.param(0b0001, 0b1001),
        pytest.param(0b0111, 0b1111),
        pytest.param(0b1111, 0b0111),
        pytest.param(0b1010, 0b0010),
        pytest.param(0b0101, 0b1101),
        pytest.param(0b1100, 0b0100),
    ])
    def test_msb_flip_4bit_bitlevel(self, input_val, expected_val):
        """Test 4-bit MSB flip - verify only bit 3 changes."""
        strategy = MSBFlipStrategy()

        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=4,
            signed=None,
            device=torch.device('cpu')
        )
        result_val = result.item()

        assert result_val == expected_val, f"Expected {expected_val}, got {result_val}"

        # Verify only bit 3 changed
        for bit_pos in range(4):
            input_bit = (input_val >> bit_pos) & 1
            result_bit = (result_val >> bit_pos) & 1

            if bit_pos == 3:
                assert input_bit != result_bit, f"Bit 3 must flip"
            else:
                assert input_bit == result_bit, f"Bit {bit_pos} must not change"

    @pytest.mark.parametrize("input_val,expected_val", [
        pytest.param(0b0000000000000000, 0b1000000000000000),
        pytest.param(0b1000000000000000, 0b0000000000000000),
        pytest.param(0b0000000000000001, 0b1000000000000001),
        pytest.param(0b0111111111111111, 0b1111111111111111),
        pytest.param(0b1111111111111111, 0b0111111111111111),
    ])
    def test_msb_flip_16bit_bitlevel(self, input_val, expected_val):
        """Test 16-bit MSB flip - verify only bit 15 changes."""
        strategy = MSBFlipStrategy()

        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=16,
            signed=None,
            device=torch.device('cpu')
        )
        result_val = result.item()

        assert result_val == expected_val, f"Expected {expected_val}, got {result_val}"

        # Verify only bit 15 changed
        for bit_pos in range(16):
            input_bit = (input_val >> bit_pos) & 1
            result_bit = (result_val >> bit_pos) & 1

            if bit_pos == 15:
                assert input_bit != result_bit, f"Bit 15 must flip"
            else:
                assert input_bit == result_bit, f"Bit {bit_pos} must not change"

    @pytest.mark.parametrize("input_val,expected_val", [
        pytest.param(0b00, 0b10),
        pytest.param(0b10, 0b00),
        pytest.param(0b01, 0b11),
        pytest.param(0b11, 0b01),
    ])
    def test_msb_flip_2bit_bitlevel(self, input_val, expected_val):
        """Test 2-bit MSB flip - verify only bit 1 changes."""
        strategy = MSBFlipStrategy()

        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=2,
            signed=None,
            device=torch.device('cpu')
        )
        result_val = result.item()

        assert result_val == expected_val, f"Expected {expected_val}, got {result_val}"

        # Verify only bit 1 changed
        for bit_pos in range(2):
            input_bit = (input_val >> bit_pos) & 1
            result_bit = (result_val >> bit_pos) & 1

            if bit_pos == 1:
                assert input_bit != result_bit, f"Bit 1 must flip"
            else:
                assert input_bit == result_bit, f"Bit {bit_pos} must not change"

    @pytest.mark.parametrize("input_values,mask_values,expected_values", [
        pytest.param([0b00000000, 0b00000001, 0b00000010, 0b00000011], [True, False, True, False], [0b10000000, 0b00000001, 0b10000010, 0b00000011]),
        pytest.param([0b00000001, 0b00000010, 0b00000011, 0b00000100], [False, True, False, True], [0b00000001, 0b10000010, 0b00000011, 0b10000100]),
        pytest.param([0b11111111, 0b11111110, 0b11111101, 0b11111100], [True, True, False, False], [0b01111111, 0b01111110, 0b11111101, 0b11111100]),
    ])
    def test_msb_flip_partial_mask(self, input_values, mask_values, expected_values):
        """Test that only masked values have MSB flipped."""
        strategy = MSBFlipStrategy()

        int_tensor = torch.tensor(input_values, dtype=torch.int32)
        mask = torch.tensor(mask_values, dtype=torch.bool)

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=8,
            signed=None,
            device=torch.device('cpu')
        )

        for i in range(len(input_values)):
            inp = input_values[i]
            msk = mask_values[i]
            exp = expected_values[i]
            res = result[i].item()

            assert res == exp, f"Position {i}: expected {exp}, got {res}"

            if msk:
                # Should be flipped
                assert res == (inp ^ 0b10000000), f"Masked value should have MSB flipped"
            else:
                # Should be unchanged
                assert res == inp, f"Unmasked value should be unchanged"

    @pytest.mark.parametrize("input_values", [
        pytest.param([0b00000000, 0b11111111, 0b10101010]),
        pytest.param([0b01010101, 0b00110011, 0b11001100]),
        pytest.param([0b11110000, 0b00001111]),
    ])
    def test_msb_flip_empty_mask(self, input_values):
        """Test that empty mask changes nothing."""
        strategy = MSBFlipStrategy()

        int_tensor = torch.tensor(input_values, dtype=torch.int32)
        mask = torch.zeros(len(input_values), dtype=torch.bool)  # All False

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=8,
            signed=None,
            device=torch.device('cpu')
        )

        for i in range(len(input_values)):
            inp = input_values[i]
            res = result[i].item()

            assert res == inp, f"With empty mask, value should be unchanged: {inp} -> {res}"

    @pytest.mark.parametrize("input_values,expected_values", [
        pytest.param([0b00000000, 0b00000001, 0b00000010, 0b00000011], [0b10000000, 0b10000001, 0b10000010, 0b10000011]),
        pytest.param([0b11111111, 0b11111110, 0b11111101], [0b01111111, 0b01111110, 0b01111101]),
        pytest.param([0b10101010, 0b01010101], [0b00101010, 0b11010101]),
    ])
    def test_msb_flip_full_mask(self, input_values, expected_values):
        """Test that full mask flips MSB for all values."""
        strategy = MSBFlipStrategy()

        int_tensor = torch.tensor(input_values, dtype=torch.int32)
        mask = torch.ones(len(input_values), dtype=torch.bool)  # All True

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=8,
            signed=None,
            device=torch.device('cpu')
        )

        for i in range(len(input_values)):
            inp = input_values[i]
            exp = expected_values[i]
            res = result[i].item()

            assert res == exp, f"Position {i}: expected {exp}, got {res}"
            assert res == (inp ^ 0b10000000), f"Result should be input XOR 128"

    def test_msb_flip_large_tensor(self):
        """Test MSB flip on large tensor (1M elements)."""
        import time

        strategy = MSBFlipStrategy()

        # Create large tensor (1 million elements)
        large_tensor = torch.randint(0, 256, (1000, 1000), dtype=torch.int32)
        mask = torch.ones_like(large_tensor, dtype=torch.bool)

        # Time the operation
        start = time.time()
        result = strategy.inject(
            int_tensor=large_tensor,
            mask=mask,
            bit_width=8,
            signed=None,
            device=torch.device('cpu')
        )
        elapsed = time.time() - start

        # Performance check
        assert elapsed < 1.0, f"Large tensor took {elapsed}s, should be < 1s"

        # Correctness check: all values should be input XOR 128
        expected = large_tensor ^ 0b10000000
        assert torch.equal(result, expected), f"Result should equal input XOR 128 for all elements"