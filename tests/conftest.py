"""Pytest fixtures for fault injection tests."""

from __future__ import annotations

import pytest
import torch
from brevitas.quant_tensor import QuantTensor


@pytest.fixture
def sample_int_tensor() -> torch.Tensor:
    """Sample quantized integer tensor for testing."""
    # Create a simple 2x2 tensor with values in range for 4-bit signed
    return torch.tensor([[1, -2], [3, 0]], dtype=torch.int32)


@pytest.fixture
def sample_mask() -> torch.Tensor:
    """Sample boolean mask for fault injection."""
    return torch.tensor([[True, False], [False, True]], dtype=torch.bool)


@pytest.fixture
def device() -> torch.device:
    """Device for tensor operations."""
    return torch.device("cpu")  # Use CPU for simplicity in tests


@pytest.fixture
def quant_tensor(sample_int_tensor: torch.Tensor) -> QuantTensor:
    """Sample QuantTensor for testing."""
    # Create a basic QuantTensor with scale and zero_point
    scale = torch.tensor(1.0)
    zero_point = torch.tensor(0.0)
    bit_width = 4
    signed = True
    return QuantTensor(
        value=sample_int_tensor.float(),  # QuantTensor expects float value
        scale=scale,
        zero_point=zero_point,
        bit_width=bit_width,
        signed=signed,
    )