from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from citegnn.utils.seed import seed_everything


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


if __name__ == "__main__":
    unittest.main()
