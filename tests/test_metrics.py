from __future__ import annotations

import builtins
import unittest
from unittest.mock import patch

import numpy as np
import torch

from citegnn.utils.metrics import (
    ClassificationMetrics,
    _to_numpy,
    choose_best_threshold,
    compute_classification_metrics,
)


class ToNumpyTests(unittest.TestCase):
    def test_converts_tensor_to_flat_numpy_array(self) -> None:
        array = _to_numpy(torch.tensor([[1, 2], [3, 4]]))
        self.assertIsInstance(array, np.ndarray)
        np.testing.assert_array_equal(array, np.asarray([1, 2, 3, 4]))

    def test_converts_python_lists_when_torch_is_unavailable(self) -> None:
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                raise ModuleNotFoundError("torch is intentionally unavailable")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            array = _to_numpy([1, 2, 3])

        np.testing.assert_array_equal(array, np.asarray([1, 2, 3]))

    def test_classification_metrics_to_dict_returns_plain_mapping(self) -> None:
        metrics = ClassificationMetrics(
            roc_auc=0.9,
            average_precision=0.8,
            precision=0.7,
            recall=0.6,
            f1=0.5,
            threshold=0.4,
            positive_rate=0.3,
        )

        self.assertEqual(
            metrics.to_dict(),
            {
                "roc_auc": 0.9,
                "average_precision": 0.8,
                "precision": 0.7,
                "recall": 0.6,
                "f1": 0.5,
                "threshold": 0.4,
                "positive_rate": 0.3,
            },
        )


class ThresholdSelectionTests(unittest.TestCase):
    def test_choose_best_threshold_maximizes_f1(self) -> None:
        threshold = choose_best_threshold(
            y_true=[0, 1, 1, 0],
            y_score=[0.1, 0.4, 0.9, 0.8],
        )
        self.assertAlmostEqual(threshold, 0.4)

    def test_choose_best_threshold_returns_single_candidate(self) -> None:
        threshold = choose_best_threshold(y_true=[0, 1], y_score=[0.5, 0.5])
        self.assertEqual(threshold, 0.5)

    def test_choose_best_threshold_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one example"):
            choose_best_threshold([], [])

    def test_choose_best_threshold_rejects_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            choose_best_threshold([0, 1], [0.2])

    def test_choose_best_threshold_supports_torch_tensors(self) -> None:
        threshold = choose_best_threshold(
            y_true=torch.tensor([0, 1, 1, 0], dtype=torch.int64),
            y_score=torch.tensor([0.1, 0.4, 0.9, 0.8], dtype=torch.float32),
        )
        self.assertAlmostEqual(threshold, 0.4)


class ClassificationMetricTests(unittest.TestCase):
    def test_compute_classification_metrics_returns_expected_fields(self) -> None:
        metrics = compute_classification_metrics(
            y_true=[0, 1, 1, 0],
            y_score=[0.1, 0.9, 0.8, 0.2],
            threshold=0.5,
        )
        self.assertAlmostEqual(metrics.roc_auc, 1.0)
        self.assertAlmostEqual(metrics.average_precision, 1.0)
        self.assertAlmostEqual(metrics.precision, 1.0)
        self.assertAlmostEqual(metrics.recall, 1.0)
        self.assertAlmostEqual(metrics.f1, 1.0)
        self.assertEqual(metrics.threshold, 0.5)
        self.assertAlmostEqual(metrics.positive_rate, 0.5)

    def test_compute_classification_metrics_rejects_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            compute_classification_metrics([0, 1], [0.5], threshold=0.5)

    def test_compute_classification_metrics_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one example"):
            compute_classification_metrics([], [], threshold=0.5)

    def test_compute_classification_metrics_requires_both_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "both positive and negative"):
            compute_classification_metrics([1, 1], [0.8, 0.9], threshold=0.5)

    def test_compute_classification_metrics_supports_torch_tensors(self) -> None:
        metrics = compute_classification_metrics(
            y_true=torch.tensor([0, 1, 1, 0], dtype=torch.int64),
            y_score=torch.tensor([0.1, 0.9, 0.8, 0.2], dtype=torch.float32),
            threshold=0.5,
        )
        self.assertAlmostEqual(metrics.roc_auc, 1.0)
        self.assertAlmostEqual(metrics.average_precision, 1.0)
        self.assertAlmostEqual(metrics.precision, 1.0)
        self.assertAlmostEqual(metrics.recall, 1.0)
        self.assertAlmostEqual(metrics.f1, 1.0)
        self.assertAlmostEqual(metrics.positive_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
