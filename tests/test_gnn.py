from __future__ import annotations

import unittest
from unittest.mock import patch

from citegnn.baselines.gnn import GNNBaselineConfig, GNNMethod, run_gnn_baseline
from tests.support import make_graph_data, make_link_prediction_split


class GnnConfigTests(unittest.TestCase):
    def test_config_validation_rejects_invalid_dropout(self) -> None:
        config = GNNBaselineConfig(dropout=1.0)
        with self.assertRaisesRegex(ValueError, "dropout must be in"):
            config.validate()


class GnnBaselineTests(unittest.TestCase):
    @patch("citegnn.baselines.gnn.load_citation_graph")
    def test_run_gnn_baseline_smoke_test_for_both_methods(self, load_citation_graph_mock) -> None:
        load_citation_graph_mock.return_value = make_graph_data(is_directed=False)
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
                self.assertGreaterEqual(result.params["best_epoch"], 1)
                self.assertIn("roc_auc", result.validation)
                self.assertIn("threshold", result.test)


if __name__ == "__main__":
    unittest.main()
