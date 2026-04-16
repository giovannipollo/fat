"""Integration tests for Trainer eval seeding."""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.seed import restore_rng_state, save_rng_state, set_eval_seed, set_seed
from utils.trainer import Trainer


def _make_tiny_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def _make_loader(
    n_samples: int = 64,
    input_dim: int = 4,
    num_classes: int = 2,
    seed: int = 0,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, input_dim, generator=generator)
    y = torch.randint(0, num_classes, (n_samples,), generator=generator)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16, shuffle=False)


def _make_config(
    val_seed: int = 43,
    test_seed: int = 44,
    train_seed: int = 42,
    enabled: bool = True,
) -> Dict[str, Any]:
    return {
        "seed": {
            "enabled": enabled,
            "value": train_seed,
            "deterministic": False,
            "val_seed": val_seed,
            "test_seed": test_seed,
        },
        "training": {"epochs": 3, "batch_size": 16},
        "optimizer": {
            "name": "sgd",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "nesterov": False,
        },
        "scheduler": {"name": "none"},
        "loss": {"name": "cross_entropy"},
        "checkpoint": {"enabled": False},
        "progress": {"enabled": False},
        "logging": {"file_enabled": False},
        "amp": {"enabled": False},
        "model": {"name": "test_model"},
        "dataset": {"name": "test_dataset"},
    }


class TestTrainerValidateSeed:
    def test_validate_same_result_across_calls(self) -> None:
        set_seed(42)
        trainer = Trainer(
            model=_make_tiny_model(),
            train_loader=_make_loader(n_samples=64, seed=1),
            test_loader=_make_loader(n_samples=32, seed=3),
            config=_make_config(val_seed=43, test_seed=44),
            device=torch.device("cpu"),
            val_loader=_make_loader(n_samples=32, seed=2),
        )

        loss1, acc1 = trainer.validate()
        _ = torch.rand(1000)
        loss2, acc2 = trainer.validate()

        assert loss1 == pytest.approx(loss2, rel=1e-8)
        assert acc1 == pytest.approx(acc2, rel=1e-8)

    def test_validate_with_seeding_disabled_runs(self) -> None:
        set_seed(42)
        trainer = Trainer(
            model=_make_tiny_model(),
            train_loader=_make_loader(n_samples=64, seed=1),
            test_loader=_make_loader(n_samples=32, seed=3),
            config=_make_config(enabled=False),
            device=torch.device("cpu"),
            val_loader=_make_loader(n_samples=32, seed=2),
        )

        loss1, acc1 = trainer.validate()
        loss2, acc2 = trainer.validate()

        assert isinstance(loss1, float)
        assert isinstance(loss2, float)
        assert isinstance(acc1, float)
        assert isinstance(acc2, float)


class TestTrainerTestSeed:
    def test_test_same_result_across_calls(self) -> None:
        set_seed(42)
        trainer = Trainer(
            model=_make_tiny_model(),
            train_loader=_make_loader(n_samples=64, seed=1),
            test_loader=_make_loader(n_samples=32, seed=3),
            config=_make_config(test_seed=44),
            device=torch.device("cpu"),
        )

        loss1, acc1 = trainer.test()
        _ = torch.rand(2000)
        loss2, acc2 = trainer.test()

        assert loss1 == pytest.approx(loss2, rel=1e-8)
        assert acc1 == pytest.approx(acc2, rel=1e-8)


class TestTrainerRngRestoration:
    def test_validate_does_not_perturb_training_rng(self) -> None:
        set_seed(42)
        trainer = Trainer(
            model=_make_tiny_model(),
            train_loader=_make_loader(n_samples=64, seed=1),
            test_loader=_make_loader(n_samples=32, seed=3),
            config=_make_config(val_seed=43, test_seed=44),
            device=torch.device("cpu"),
            val_loader=_make_loader(n_samples=32, seed=2),
        )

        _ = torch.rand(100)
        baseline_snapshot = save_rng_state()
        baseline_next = torch.rand(50).tolist()

        restore_rng_state(baseline_snapshot)
        trainer.validate()
        with_eval_next = torch.rand(50).tolist()

        assert baseline_next == with_eval_next


class TestValSeedVsTestSeedAreDistinct:
    def test_val_seed_and_test_seed_produce_distinct_random_samples(self) -> None:
        set_seed(42)
        snapshot = save_rng_state()

        set_eval_seed(43)
        val_rng_sample = torch.rand(10).tolist()
        restore_rng_state(snapshot)

        set_eval_seed(44)
        test_rng_sample = torch.rand(10).tolist()

        assert val_rng_sample != test_rng_sample
