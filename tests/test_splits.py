from __future__ import annotations

import unittest

import torch

from citegnn.data.splits import (
    _edge_pairs,
    _split_labeled_edges,
    _validate_ratios,
    create_link_prediction_split,
    load_split_artifact,
    save_split_artifact,
    validate_link_prediction_split,
)
from citegnn.utils.seed import seed_everything
from tests.support import make_graph_data, make_link_prediction_split, workspace_temp_dir


class SplitHelperTests(unittest.TestCase):
    def test_split_properties_report_edge_counts(self) -> None:
        split = make_link_prediction_split()

        self.assertEqual(split.train_graph_edge_count, 10)
        self.assertEqual(split.train_supervision_edge_count, 8)

    def test_edge_pairs_canonicalizes_undirected_duplicates(self) -> None:
        edge_index = torch.tensor([[2, 1], [1, 2]], dtype=torch.long)
        pairs = _edge_pairs(edge_index)
        self.assertEqual(pairs, {(1, 2)})

    def test_edge_pairs_rejects_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape \\[2, num_edges\\]"):
            _edge_pairs(torch.tensor([0, 1], dtype=torch.long))

    def test_validate_ratios_rejects_invalid_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "val_ratio must be in"):
            _validate_ratios(0.0, 0.2)
        with self.assertRaisesRegex(ValueError, "test_ratio must be in"):
            _validate_ratios(0.2, 1.0)
        with self.assertRaisesRegex(ValueError, "must be < 1"):
            _validate_ratios(0.6, 0.4)

    def test_split_labeled_edges_separates_positive_and_negative_columns(self) -> None:
        edge_label_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_label = torch.tensor([1, 0, 1], dtype=torch.float32)
        pos_edges, neg_edges = _split_labeled_edges(edge_label_index, edge_label)

        self.assertTrue(torch.equal(pos_edges, torch.tensor([[0, 2], [1, 3]], dtype=torch.long)))
        self.assertTrue(torch.equal(neg_edges, torch.tensor([[1], [2]], dtype=torch.long)))

    def test_split_labeled_edges_rejects_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape \\[2, num_edges\\]"):
            _split_labeled_edges(torch.tensor([0, 1], dtype=torch.long), torch.tensor([1, 0]))


class SplitArtifactTests(unittest.TestCase):
    def test_create_link_prediction_split_rejects_non_positive_negative_ratio(self) -> None:
        graph = make_graph_data()

        with self.assertRaisesRegex(ValueError, "negative_ratio must be positive"):
            create_link_prediction_split(graph, negative_ratio=0.0)

    def test_create_link_prediction_split_is_reproducible_with_global_seed(self) -> None:
        graph = make_graph_data()

        seed_everything(123)
        split_a = create_link_prediction_split(
            graph,
            val_ratio=0.2,
            test_ratio=0.2,
            negative_ratio=1.0,
            seed=7,
        )
        seed_everything(123)
        split_b = create_link_prediction_split(
            graph,
            val_ratio=0.2,
            test_ratio=0.2,
            negative_ratio=1.0,
            seed=7,
        )

        self.assertTrue(torch.equal(split_a.train_graph_edge_index, split_b.train_graph_edge_index))
        self.assertTrue(torch.equal(split_a.val_pos_edge_index, split_b.val_pos_edge_index))
        self.assertEqual(split_a.seed, 7)
        self.assertEqual(split_a.dataset_name, "Cora")
        self.assertGreater(split_a.train_pos_edge_index.size(1), 0)
        self.assertGreater(split_a.train_neg_edge_index.size(1), 0)

    def test_save_and_load_split_artifact_round_trip(self) -> None:
        split = make_link_prediction_split()

        with workspace_temp_dir("splits") as tmp_dir:
            artifact_path = save_split_artifact(split, output_dir=tmp_dir)
            loaded = load_split_artifact(artifact_path)

        self.assertEqual(artifact_path.name, "cora_seed42_lp.pt")
        self.assertEqual(loaded.dataset_name, split.dataset_name)
        self.assertTrue(torch.equal(loaded.train_graph_edge_index, split.train_graph_edge_index))
        self.assertTrue(torch.equal(loaded.test_neg_edge_index, split.test_neg_edge_index))

    def test_validate_link_prediction_split_rejects_leakage(self) -> None:
        split = make_link_prediction_split()
        split.val_pos_edge_index = torch.tensor([[0], [1]], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "leak into the training graph"):
            validate_link_prediction_split(split)

    def test_validate_link_prediction_split_rejects_validation_test_overlap(self) -> None:
        split = make_link_prediction_split()
        split.test_pos_edge_index = split.val_pos_edge_index.clone()

        with self.assertRaisesRegex(ValueError, "validation and test positive edges overlap"):
            validate_link_prediction_split(split)


if __name__ == "__main__":
    unittest.main()
