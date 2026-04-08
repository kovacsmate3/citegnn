from __future__ import annotations

import unittest

import numpy as np

from citegnn.baselines.node2vec import (
    Node2VecConfig,
    _generate_random_walks,
    _node2vec_transition_weights,
    _positive_pairs_from_walks,
    run_node2vec_baseline,
)
from tests.support import make_link_prediction_split


class Node2VecHelperTests(unittest.TestCase):
    def test_config_validation_rejects_non_positive_values(self) -> None:
        config = Node2VecConfig(embedding_dim=0)
        with self.assertRaisesRegex(ValueError, "embedding_dim must be positive"):
            config.validate()

    def test_transition_weights_normalize_and_apply_bias(self) -> None:
        weights = _node2vec_transition_weights(
            previous=0,
            current=1,
            neighbors=[0, 2, 4],
            edge_pairs={(0, 1), (1, 2), (0, 2)},
            p=0.5,
            q=2.0,
        )

        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_generate_random_walks_keeps_isolated_nodes_as_singleton_walks(self) -> None:
        adjacency = {0: [1], 1: [0], 2: []}
        walks = _generate_random_walks(
            adjacency,
            walk_length=4,
            walks_per_node=1,
            p=1.0,
            q=1.0,
            rng=np.random.default_rng(0),
        )

        self.assertEqual(len(walks), 3)
        self.assertIn([2], walks)

    def test_positive_pairs_from_walks_uses_context_window(self) -> None:
        pairs = _positive_pairs_from_walks([[0, 1, 2]], context_window=1)
        self.assertEqual(pairs, [(0, 1), (1, 0), (1, 2), (2, 1)])


class Node2VecBaselineTests(unittest.TestCase):
    def test_run_node2vec_baseline_returns_metrics_on_synthetic_split(self) -> None:
        split = make_link_prediction_split()
        config = Node2VecConfig(
            embedding_dim=8,
            walk_length=4,
            walks_per_node=2,
            context_window=1,
            negative_samples=2,
            epochs=1,
            batch_size=8,
            learning_rate=0.01,
            p=1.0,
            q=1.0,
            classifier_c=1.0,
        )

        result = run_node2vec_baseline(split, config=config, seed=7)

        self.assertEqual(result.dataset, "Cora")
        self.assertEqual(result.method, "node2vec")
        self.assertEqual(result.seed, 7)
        self.assertEqual(result.params["embedding_dim"], 8)
        self.assertEqual(result.params["train_supervision_edges"], 8)
        self.assertIn("roc_auc", result.validation)
        self.assertIn("positive_rate", result.test)


if __name__ == "__main__":
    unittest.main()
