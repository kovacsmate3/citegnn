from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from citegnn.baselines.feature_only import (
    FeatureOnlyBaselineConfig,
    _edge_features,
    build_pairwise_feature_dataset,
    run_feature_only_baseline,
)
from tests.support import make_graph_data, make_link_prediction_split


class FeatureOnlyDatasetTests(unittest.TestCase):
    def test_config_validation_rejects_invalid_feature_mode(self) -> None:
        config = FeatureOnlyBaselineConfig(feature_mode="unknown")
        with self.assertRaisesRegex(ValueError, "feature_mode must be one of"):
            config.validate()

    def test_edge_features_support_symmetric_pairwise_modes(self) -> None:
        graph = make_graph_data()
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        hadamard = _edge_features(graph.x, edge_index, feature_mode="hadamard")
        abs_diff = _edge_features(graph.x, edge_index, feature_mode="abs_diff")
        combined = _edge_features(graph.x, edge_index, feature_mode="hadamard_abs_diff")

        self.assertTrue(torch.equal(hadamard[0], hadamard[1]))
        self.assertTrue(torch.equal(abs_diff[0], abs_diff[1]))
        self.assertEqual(combined.size(1), graph.num_features * 2)

    def test_build_pairwise_feature_dataset_uses_split_shapes(self) -> None:
        split = make_link_prediction_split()
        graph = make_graph_data()

        dataset = build_pairwise_feature_dataset(split, graph, feature_mode="hadamard_abs_diff")

        self.assertEqual(dataset.dataset_name, "Cora")
        self.assertEqual(dataset.train_features.size(0), split.train_supervision_edge_count)
        self.assertEqual(dataset.val_features.size(0), 2)
        self.assertEqual(dataset.test_features.size(0), 2)
        self.assertEqual(dataset.feature_dim, graph.num_features * 2)
        np.testing.assert_array_equal(dataset.val_labels.numpy(), np.asarray([1.0, 0.0]))


class FeatureOnlyBaselineTests(unittest.TestCase):
    @patch("citegnn.baselines.feature_only.load_citation_graph")
    def test_run_feature_only_baseline_returns_metrics(self, load_graph_mock) -> None:
        load_graph_mock.return_value = make_graph_data()
        split = make_link_prediction_split()
        config = FeatureOnlyBaselineConfig(
            hidden_dim=8,
            epochs=3,
            patience=2,
            batch_size=4,
            dropout=0.0,
        )

        result = run_feature_only_baseline(split, config=config, seed=5, data_root="ignored")

        self.assertEqual(result.dataset, "Cora")
        self.assertEqual(result.method, "feature_only_mlp")
        self.assertEqual(result.seed, 5)
        self.assertEqual(result.params["feature_mode"], "hadamard_abs_diff")
        self.assertGreaterEqual(result.params["pairwise_feature_dim"], 1)
        self.assertIn("roc_auc", result.validation)
        self.assertIn("threshold", result.test)


if __name__ == "__main__":
    unittest.main()
