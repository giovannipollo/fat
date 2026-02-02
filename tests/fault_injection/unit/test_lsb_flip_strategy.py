"""
Test LSB Flip Strategy

Tests for the LSB (Least Significant Bit) flip fault injection strategy.
"""

import pytest
import torch
from utils.fault_injection.strategies import LSBFlipStrategy


class TestLSBFlipStrategy:
    """Test suite for LSB Flip Strategy."""

    @pytest.mark.parametrize(
        "input_val,expected_val",
        [
            pytest.param(0b00000001, 0b00000000),
            pytest.param(0b00000000, 0b00000001),
            pytest.param(0b00000010, 0b00000011),
            pytest.param(0b00000011, 0b00000010),
            pytest.param(0b01010101, 0b01010100),
            pytest.param(0b10101010, 0b10101011),
            pytest.param(0b11111110, 0b11111111),
            pytest.param(0b11111111, 0b11111110),
            pytest.param(0b10000000, 0b10000001),
            pytest.param(0b01111111, 0b01111110),
        ],
    )
    def test_lsb_flip_8bit_bitlevel(self, input_val, expected_val):
        """
        Test LSB flip at bit level.

        Verifies:
        1. Output = Input XOR 1
        2. Only bit 0 changes
        3. Bits 1-7 remain unchanged
        """
        strategy = LSBFlipStrategy()

        # Create tensor
        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        # Apply LSB flip
        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=8,
            signed=None,
            device=torch.device("cpu"),
        )

        result_val = result.item()

        # Check result matches expected
        assert result_val == expected_val, f"Expected {expected_val}, got {result_val}"

        # Verify result = input XOR 1
        assert result_val == (input_val ^ 1), f"Result should equal input XOR 1"

        # Verify only bit 0 changed
        for bit_pos in range(8):
            input_bit = (input_val >> bit_pos) & 1
            result_bit = (result_val >> bit_pos) & 1

            if bit_pos == 0:
                # Bit 0 MUST flip
                assert input_bit != result_bit, f"Bit 0 must flip"
            else:
                # Bits 1-7 MUST NOT change
                assert input_bit == result_bit, f"Bit {bit_pos} must not change"

    @pytest.mark.parametrize(
        "input_val,expected_val",
        [
            pytest.param(0b0001, 0b0000),
            pytest.param(0b0000, 0b0001),
            pytest.param(0b0010, 0b0011),
            pytest.param(0b0111, 0b0110),
            pytest.param(0b1000, 0b1001),
            pytest.param(0b1111, 0b1110),
            pytest.param(0b1010, 0b1011),
            pytest.param(0b0101, 0b0100),
        ],
    )
    def test_lsb_flip_4bit_bitlevel(self, input_val, expected_val):
        """Test 4-bit LSB flip - verify only bit 0 changes."""
        strategy = LSBFlipStrategy()

        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=4,
            signed=None,
            device=torch.device("cpu"),
        )
        result_val = result.item()

        # Check result
        assert result_val == expected_val, f"Expected {expected_val}, got {result_val}"

        # Verify only bit 0 changed
        for bit_pos in range(4):
            input_bit = (input_val >> bit_pos) & 1
            result_bit = (result_val >> bit_pos) & 1

            if bit_pos == 0:
                assert input_bit != result_bit, f"Bit 0 must flip"
            else:
                assert input_bit == result_bit, f"Bit {bit_pos} must not change"

    @pytest.mark.parametrize(
        "input_val,expected_val",
        [
            pytest.param(0b0000000000000001, 0b0000000000000000),
            pytest.param(0b0000000000000000, 0b0000000000000001),
            pytest.param(0b0000001111111110, 0b0000001111111111),
            pytest.param(0b1111111111111111, 0b1111111111111110),
            pytest.param(0b1000000000000000, 0b1000000000000001),
        ],
    )
    def test_lsb_flip_16bit_bitlevel(self, input_val, expected_val):
        """Test 16-bit LSB flip - verify only bit 0 changes."""
        strategy = LSBFlipStrategy()

        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=16,
            signed=None,
            device=torch.device("cpu"),
        )
        result_val = result.item()

        assert result_val == expected_val, f"Expected {expected_val}, got {result_val}"

        # Verify only bit 0 changed
        for bit_pos in range(16):
            input_bit = (input_val >> bit_pos) & 1
            result_bit = (result_val >> bit_pos) & 1

            if bit_pos == 0:
                assert input_bit != result_bit, f"Bit 0 must flip"
            else:
                assert input_bit == result_bit, f"Bit {bit_pos} must not change"

    @pytest.mark.parametrize(
        "input_val,expected_val",
        [
            pytest.param(0b01, 0b00),
            pytest.param(0b00, 0b01),
            pytest.param(0b10, 0b11),
            pytest.param(0b11, 0b10),
        ],
    )
    def test_lsb_flip_2bit_bitlevel(self, input_val, expected_val):
        """Test 2-bit LSB flip - verify only bit 0 changes."""
        strategy = LSBFlipStrategy()

        int_tensor = torch.tensor([input_val], dtype=torch.int32)
        mask = torch.tensor([True], dtype=torch.bool)

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=2,
            signed=None,
            device=torch.device("cpu"),
        )
        result_val = result.item()

        assert result_val == expected_val, f"Expected {expected_val}, got {result_val}"

        # Verify only bit 0 changed
        for bit_pos in range(2):
            input_bit = (input_val >> bit_pos) & 1
            result_bit = (result_val >> bit_pos) & 1

            if bit_pos == 0:
                assert input_bit != result_bit, f"Bit 0 must flip"
            else:
                assert input_bit == result_bit, f"Bit {bit_pos} must not change"

    @pytest.mark.parametrize(
        "input_values,mask_values,expected_values",
        [
            pytest.param(
                [0b00000000, 0b00000001, 0b00000010, 0b00000011],
                [True, False, True, False],
                [0b00000001, 0b00000001, 0b00000011, 0b00000011],
            ),
            pytest.param(
                [0b00000001, 0b00000010, 0b00000011, 0b00000100],
                [False, True, False, True],
                [0b00000001, 0b00000011, 0b00000011, 0b00000101],
            ),
            pytest.param(
                [0b11111111, 0b11111110, 0b11111101, 0b11111100],
                [True, True, False, False],
                [0b11111110, 0b11111111, 0b11111101, 0b11111100],
            ),
        ],
    )
    def test_lsb_flip_partial_mask(self, input_values, mask_values, expected_values):
        """Test that only masked values have bit 0 flipped."""
        strategy = LSBFlipStrategy()

        int_tensor = torch.tensor(input_values, dtype=torch.int32)
        mask = torch.tensor(mask_values, dtype=torch.bool)

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=8,
            signed=None,
            device=torch.device("cpu"),
        )

        for i in range(len(input_values)):
            inp = input_values[i]
            msk = mask_values[i]
            exp = expected_values[i]
            res = result[i].item()

            assert res == exp, f"Position {i}: expected {exp}, got {res}"

            if msk:
                # Should be flipped
                assert res == (inp ^ 1), f"Masked value should have bit 0 flipped"
            else:
                # Should be unchanged
                assert res == inp, f"Unmasked value should be unchanged"

    @pytest.mark.parametrize(
        "input_values",
        [
            pytest.param([0b00000000, 0b11111111, 0b10101010]),
            pytest.param([0b01010101, 0b00110011, 0b11001100]),
            pytest.param([0b11110000, 0b00001111]),
        ],
    )
    def test_lsb_flip_empty_mask(self, input_values):
        """Test that empty mask changes nothing."""
        strategy = LSBFlipStrategy()

        int_tensor = torch.tensor(input_values, dtype=torch.int32)
        mask = torch.zeros(len(input_values), dtype=torch.bool)  # All False

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=8,
            signed=None,
            device=torch.device("cpu"),
        )

        for i in range(len(input_values)):
            inp = input_values[i]
            res = result[i].item()

            assert (
                res == inp
            ), f"With empty mask, value should be unchanged: {inp} -> {res}"

    @pytest.mark.parametrize(
        "input_values,expected_values",
        [
            pytest.param(
                [0b00000000, 0b00000001, 0b00000010, 0b00000011],
                [0b00000001, 0b00000000, 0b00000011, 0b00000010],
            ),
            pytest.param(
                [0b11111111, 0b11111110, 0b11111101],
                [0b11111110, 0b11111111, 0b11111100],
            ),
            pytest.param([0b10101010, 0b01010101], [0b10101011, 0b01010100]),
        ],
    )
    def test_lsb_flip_full_mask(self, input_values, expected_values):
        """Test that full mask flips bit 0 for all values."""
        strategy = LSBFlipStrategy()

        int_tensor = torch.tensor(input_values, dtype=torch.int32)
        mask = torch.ones(len(input_values), dtype=torch.bool)  # All True

        result = strategy.inject(
            int_tensor=int_tensor,
            mask=mask,
            bit_width=8,
            signed=None,
            device=torch.device("cpu"),
        )

        for i in range(len(input_values)):
            inp = input_values[i]
            exp = expected_values[i]
            res = result[i].item()

            assert res == exp, f"Position {i}: expected {exp}, got {res}"
            assert res == (inp ^ 1), f"Result should be input XOR 1"

    def test_lsb_flip_large_tensor(self):
        """Test LSB flip on large tensor (1M elements)."""
        import time

        strategy = LSBFlipStrategy()

        # Create large tensor (1 million elements)
        large_tensor = torch.randint(0, 256, (10000, 10000), dtype=torch.int32)
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
        assert elapsed < 0.1, f"Large tensor took {elapsed}s, should be < 1s"

        # Correctness check: all values should be input XOR 1
        expected = large_tensor ^ 1
        assert torch.equal(result, expected), f"Result should equal input XOR 1 for all elements"
