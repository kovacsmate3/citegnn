from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from citegnn.baselines.gnn import (
    GNNBaselineConfig,
    GNNMethod,
    _build_model,
    _combined_edge_index,
    _edge_labels,
    _predict_scores,
    run_gnn_baseline,
)
from tests.support import make_graph_data, make_link_prediction_split


class GnnConfigTests(unittest.TestCase):
    def test_config_validation_rejects_invalid_dropout(self) -> None:
        config = GNNBaselineConfig(dropout=1.0)
        with self.assertRaisesRegex(ValueError, "dropout must be in"):
            config.validate()

    def test_config_validation_rejects_non_positive_hidden_dim(self) -> None:
        config = GNNBaselineConfig(hidden_dim=0)
        with self.assertRaisesRegex(ValueError, "hidden_dim must be positive"):
            config.validate()

    def test_config_to_dict_serializes_all_fields(self) -> None:
        config = GNNBaselineConfig(hidden_dim=8, dropout=0.1, epochs=3)

        self.assertEqual(config.to_dict()["hidden_dim"], 8)
        self.assertEqual(config.to_dict()["dropout"], 0.1)
        self.assertEqual(config.to_dict()["epochs"], 3)


class GnnHelperTests(unittest.TestCase):
    def test_edge_helpers_build_expected_labels_and_edge_index(self) -> None:
        pos_edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        neg_edge_index = torch.tensor([[1], [4]], dtype=torch.long)

        labels = _edge_labels(pos_edge_index, neg_edge_index)
        edge_index = _combined_edge_index(pos_edge_index, neg_edge_index)

        np.testing.assert_array_equal(labels.numpy(), np.asarray([1.0, 1.0, 0.0]))
        self.assertTrue(
            torch.equal(
                edge_index,
                torch.tensor([[0, 2, 1], [1, 3, 4]], dtype=torch.long),
            )
        )

    def test_build_model_rejects_unsupported_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported GNN method"):
            _build_model(method="invalid", input_dim=3, config=GNNBaselineConfig())  # type: ignore[arg-type]

    def test_predict_scores_returns_numpy_arrays(self) -> None:
        graph = make_graph_data()
        split = make_link_prediction_split()
        model = _build_model(
            method=GNNMethod.GCN,
            input_dim=graph.num_features,
            config=GNNBaselineConfig(hidden_dim=4, num_layers=2, dropout=0.0, decoder_hidden_dim=4),
        )

        labels, scores = _predict_scores(
            model,
            x=graph.x,
            message_passing_edge_index=split.train_graph_edge_index,
            pos_edge_index=split.val_pos_edge_index,
            neg_edge_index=split.val_neg_edge_index,
        )

        np.testing.assert_array_equal(labels.cpu().numpy(), np.asarray([1, 0]))
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(scores.dtype, torch.float64)
        self.assertTrue(torch.all((scores >= 0.0) & (scores <= 1.0)).item())


class GnnBaselineTests(unittest.TestCase):
    @patch("citegnn.baselines.gnn.load_citation_graph")
    def test_run_gnn_baseline_smoke_test_for_both_methods(self, load_citation_graph_mock) -> None:
        load_citation_graph_mock.return_value = make_graph_data()
        split = make_link_prediction_split()
        config = GNNBaselineConfig(
            hidden_dim=4,
            num_layers=2,
            dropout=0.0,
            decoder_hidden_dim=4,
            learning_rate=0.01,
            weight_decay=0.0,
            epochs=2,
            patience=1,
        )

        for method in (GNNMethod.GCN, GNNMethod.GRAPHSAGE):
            with self.subTest(method=method.value):
                result = run_gnn_baseline(
                    split,
                    method=method,
                    config=config,
                    seed=5,
                    data_root="ignored",
                )
                self.assertEqual(result.dataset, "Cora")
                self.assertEqual(result.method, method.value)
                self.assertEqual(result.seed, 5)
                self.assertIn("device", result.params)
                self.assertGreaterEqual(result.params["best_epoch"], 1)
                self.assertIn("roc_auc", result.validation)
                self.assertIn("threshold", result.test)


if __name__ == "__main__":
    unittest.main()
