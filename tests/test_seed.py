from __future__ import annotations

import builtins
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from citegnn.utils.seed import seed_everything


class _FakeCuda:
    def __init__(self, *, available: bool) -> None:
        self.available = available
        self.manual_seed_all_calls: list[int] = []

    def is_available(self) -> bool:
        return self.available

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seed_all_calls.append(seed)


class _FakeTorch:
    def __init__(self) -> None:
        self.manual_seed_calls: list[int] = []
        self.cuda = _FakeCuda(available=True)
        self.use_calls: list[tuple[bool, bool]] = []

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)

    def use_deterministic_algorithms(self, enabled: bool, warn_only: bool = False) -> None:
        self.use_calls.append((enabled, warn_only))
        if warn_only:
            raise TypeError("warn_only is unsupported")


class SeedEverythingTests(unittest.TestCase):
    def test_seed_everything_resets_python_numpy_and_torch_rngs(self) -> None:
        seed_everything(123)
        python_first = random.random()
        numpy_first = np.random.rand(3)
        torch_first = torch.rand(3)

        seed_everything(123)
        python_second = random.random()
        numpy_second = np.random.rand(3)
        torch_second = torch.rand(3)

        self.assertEqual(python_first, python_second)
        np.testing.assert_allclose(numpy_first, numpy_second)
        self.assertTrue(torch.equal(torch_first, torch_second))

    def test_seed_everything_still_seeds_python_and_numpy_without_torch(self) -> None:
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                raise ModuleNotFoundError("torch is intentionally unavailable")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            seed_everything(77)
            python_first = random.random()
            numpy_first = np.random.rand(2)

            seed_everything(77)
            python_second = random.random()
            numpy_second = np.random.rand(2)

        self.assertEqual(python_first, python_second)
        np.testing.assert_allclose(numpy_first, numpy_second)

    def test_seed_everything_uses_cuda_and_typeerror_fallback_for_determinism(self) -> None:
        original_import = builtins.__import__
        fake_torch = _FakeTorch()

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                return fake_torch
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            seed_everything(19)

        self.assertEqual(fake_torch.manual_seed_calls, [19])
        self.assertEqual(fake_torch.cuda.manual_seed_all_calls, [19])
        self.assertEqual(fake_torch.use_calls, [(True, True), (True, False)])


if __name__ == "__main__":
    unittest.main()
