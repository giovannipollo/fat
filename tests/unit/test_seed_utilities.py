"""Unit tests for seed utility functions."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from utils.seed import RngState, restore_rng_state, save_rng_state, set_eval_seed, set_seed


class TestSaveRestoreRngState:
    """Tests for save_rng_state() and restore_rng_state()."""

    def test_snapshot_captures_python_random(self) -> None:
        set_seed(42)
        _ = [random.random() for _ in range(10)]

        snapshot = save_rng_state()
        value_before = random.random()

        restore_rng_state(snapshot)
        value_after = random.random()

        assert value_before == value_after

    def test_snapshot_captures_numpy(self) -> None:
        set_seed(42)
        _ = np.random.rand(10)

        snapshot = save_rng_state()
        values_before = np.random.rand(5)

        restore_rng_state(snapshot)
        values_after = np.random.rand(5)

        np.testing.assert_array_equal(values_before, values_after)

    def test_snapshot_captures_torch_cpu(self) -> None:
        set_seed(42)
        _ = torch.rand(10)

        snapshot = save_rng_state()
        values_before = torch.rand(5)

        restore_rng_state(snapshot)
        values_after = torch.rand(5)

        assert torch.allclose(values_before, values_after)

    def test_snapshot_is_reusable(self) -> None:
        set_seed(100)
        snapshot = save_rng_state()

        for _ in range(1000):
            random.random()
            np.random.rand(1)
            torch.rand(1)

        restore_rng_state(snapshot)
        first = torch.rand(3)

        restore_rng_state(snapshot)
        second = torch.rand(3)

        assert torch.allclose(first, second)

    def test_rng_state_has_expected_types(self) -> None:
        state = save_rng_state()

        assert isinstance(state, RngState)
        assert isinstance(state.python_state, tuple)
        assert isinstance(state.numpy_state, tuple)
        assert isinstance(state.torch_state, torch.Tensor)
        assert isinstance(state.cuda_states, list)

    @pytest.mark.parametrize("seed_value", [0, 1, 42, 2**31 - 1])
    def test_snapshot_round_trip_various_seeds(self, seed_value: int) -> None:
        set_seed(seed_value)
        snapshot = save_rng_state()
        values_before = torch.rand(4)

        _ = torch.rand(100)

        restore_rng_state(snapshot)
        values_after = torch.rand(4)

        assert torch.allclose(values_before, values_after)


class TestSetEvalSeed:
    """Tests for set_eval_seed()."""

    def test_eval_seed_resets_torch(self) -> None:
        set_eval_seed(99)
        first = torch.rand(10).tolist()

        set_eval_seed(99)
        second = torch.rand(10).tolist()

        assert first == second

    def test_eval_seed_resets_python_random(self) -> None:
        set_eval_seed(77)
        first = [random.random() for _ in range(10)]

        set_eval_seed(77)
        second = [random.random() for _ in range(10)]

        assert first == second

    def test_eval_seed_resets_numpy(self) -> None:
        set_eval_seed(55)
        first = np.random.rand(10)

        set_eval_seed(55)
        second = np.random.rand(10)

        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("val_seed,test_seed", [(43, 44), (100, 200), (0, 1)])
    def test_different_seeds_produce_different_sequences(
        self, val_seed: int, test_seed: int
    ) -> None:
        set_eval_seed(val_seed)
        val_seq = torch.rand(20).tolist()

        set_eval_seed(test_seed)
        test_seq = torch.rand(20).tolist()

        assert val_seq != test_seq

    def test_eval_seed_does_not_modify_cudnn_flags(self) -> None:
        original_deterministic = torch.backends.cudnn.deterministic
        original_benchmark = torch.backends.cudnn.benchmark

        set_eval_seed(42)

        assert torch.backends.cudnn.deterministic == original_deterministic
        assert torch.backends.cudnn.benchmark == original_benchmark


class TestSaveSetRestore:
    """Tests for save -> set_eval_seed -> restore flow."""

    def test_training_rng_continues_as_if_eval_never_happened(self) -> None:
        set_seed(42)
        _ = torch.rand(50)

        baseline_snapshot = save_rng_state()
        baseline_sequence = torch.rand(20).tolist()

        restore_rng_state(baseline_snapshot)
        eval_snapshot = save_rng_state()
        set_eval_seed(999)
        _ = torch.rand(200)
        restore_rng_state(eval_snapshot)
        sequence_after_eval = torch.rand(20).tolist()

        assert baseline_sequence == sequence_after_eval

    def test_exception_in_eval_still_restores_rng(self) -> None:
        set_seed(42)
        _ = torch.rand(50)

        baseline_snapshot = save_rng_state()
        expected = torch.rand(10).tolist()
        restore_rng_state(baseline_snapshot)

        eval_snapshot = save_rng_state()
        set_eval_seed(999)
        try:
            _ = torch.rand(100)
            raise RuntimeError("simulated eval error")
        except RuntimeError:
            restore_rng_state(eval_snapshot)

        actual = torch.rand(10).tolist()
        assert expected == actual
