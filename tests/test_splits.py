"""Tests for the HeaRT-aligned split factory and on-disk artifact format."""

from __future__ import annotations

import unittest

import torch

from citegnn.data.heart_split import (
    DEFAULT_K,
    create_heart_split,
    heart_split_artifact_name,
    load_heart_split,
    save_heart_split,
)
from citegnn.data.splits import (
    LinkPredictionSplit,
    create_link_prediction_split,
    load_split_artifact,
    save_split_artifact,
)
from tests.support import make_graph_data, make_heart_split, workspace_temp_dir


class HeartSplitConstructionTests(unittest.TestCase):
    def test_split_partitions_canonical_edges_into_85_5_10(self) -> None:
        graph = make_graph_data()

        split = create_heart_split(
            graph,
            seed=7,
            K=4,
            split_ratio=(0.6, 0.2, 0.2),
            ppr_iterations=10,
        )

        total_positives = (
            split.num_train_positives + split.num_val_positives + split.num_test_positives
        )
        # 10 unique undirected edges in the toy graph.
        self.assertEqual(total_positives, 10)
        self.assertGreaterEqual(split.num_val_positives, 1)
        self.assertGreaterEqual(split.num_test_positives, 1)
        self.assertEqual(split.num_nodes, graph.num_nodes)
        self.assertEqual(split.K, 4)

    def test_train_message_passing_graph_is_symmetric(self) -> None:
        graph = make_graph_data()

        split = create_heart_split(
            graph,
            seed=11,
            K=4,
            split_ratio=(0.6, 0.2, 0.2),
            ppr_iterations=10,
        )
        edges = split.train_edge_index
        self.assertEqual(edges.size(0), 2)
        # Every (u, v) must have a matching (v, u).
        forward = {(int(u), int(v)) for u, v in edges.t().tolist()}
        backward = {(v, u) for (u, v) in forward}
        self.assertEqual(forward, backward)

    def test_held_out_positives_are_excluded_from_train_graph(self) -> None:
        graph = make_graph_data()

        split = create_heart_split(
            graph,
            seed=11,
            K=4,
            split_ratio=(0.6, 0.2, 0.2),
            ppr_iterations=10,
        )
        train_pairs = {
            (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
            for u, v in split.train_edge_index.t().tolist()
        }
        for partition in (split.val_pos_edge_index, split.test_pos_edge_index):
            for u, v in partition.t().tolist():
                key = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
                self.assertNotIn(key, train_pairs)

    def test_negative_nodes_have_expected_shape(self) -> None:
        graph = make_graph_data()

        split = create_heart_split(
            graph,
            seed=11,
            K=4,
            split_ratio=(0.6, 0.2, 0.2),
            ppr_iterations=10,
        )
        self.assertEqual(split.val_neg_nodes.size(1), split.K)
        self.assertEqual(split.test_neg_nodes.size(1), split.K)
        self.assertEqual(split.val_neg_nodes.size(0), split.num_val_positives)
        self.assertEqual(split.test_neg_nodes.size(0), split.num_test_positives)

    def test_K_must_be_positive_and_even(self) -> None:
        graph = make_graph_data()
        with self.assertRaisesRegex(ValueError, "K must be a positive even number"):
            create_heart_split(graph, seed=0, K=3)
        with self.assertRaisesRegex(ValueError, "K must be a positive even number"):
            create_heart_split(graph, seed=0, K=0)


class HeartSplitArtifactTests(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        split = make_heart_split()
        with workspace_temp_dir("heart") as tmp_dir:
            path = save_heart_split(split, output_dir=tmp_dir)
            loaded = load_heart_split(path)

        expected_name = heart_split_artifact_name(split.dataset_name, seed=split.seed, K=split.K)
        self.assertEqual(path.name, expected_name)
        self.assertEqual(loaded.dataset_name, split.dataset_name)
        self.assertTrue(torch.equal(loaded.train_edge_index, split.train_edge_index))
        self.assertTrue(torch.equal(loaded.val_neg_nodes, split.val_neg_nodes))
        self.assertEqual(loaded.split_ratio, split.split_ratio)
        self.assertEqual(loaded.heuristics_used, split.heuristics_used)


class BackwardsCompatibilityShimTests(unittest.TestCase):
    def test_legacy_aliases_point_at_heart_implementation(self) -> None:
        self.assertIs(LinkPredictionSplit.__name__, "HeartSplit")
        # The aliased factories must reach the new implementation.
        self.assertIs(create_link_prediction_split, create_heart_split)
        self.assertIs(load_split_artifact, load_heart_split)
        self.assertIs(save_split_artifact, save_heart_split)
        self.assertEqual(DEFAULT_K, 500)


if __name__ == "__main__":
    unittest.main()
