from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from citegnn.baselines import heuristics
from citegnn.baselines.heuristics import HeuristicMethod
from tests.support import make_metrics
from tests.support import make_link_prediction_split


class HeuristicHelperTests(unittest.TestCase):
    def test_edge_helpers_validate_and_combine_candidate_edges(self) -> None:
        pos_edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        neg_edge_index = torch.tensor([[1], [4]], dtype=torch.long)

        combined_edges, labels = heuristics._combine_labeled_edges(pos_edge_index, neg_edge_index)

        self.assertEqual(combined_edges, [(0, 1), (2, 3), (1, 4)])
        np.testing.assert_array_equal(labels, np.asarray([1, 1, 0]))

    def test_edge_columns_reject_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape \\[2, num_edges\\]"):
            heuristics._edge_columns(torch.tensor([0, 1], dtype=torch.long))

    def test_build_scoring_graph_preserves_directionality_and_node_count(self) -> None:
        undirected_graph = heuristics._build_scoring_graph(make_link_prediction_split(is_undirected=True))
        directed_graph = heuristics._build_scoring_graph(make_link_prediction_split(is_undirected=False))

        self.assertEqual(undirected_graph.number_of_nodes(), 6)
        self.assertFalse(undirected_graph.is_directed())
        self.assertTrue(directed_graph.is_directed())

    def test_known_common_neighbor_and_adamic_adar_scores_rank_positive_edge_higher(self) -> None:
        graph = heuristics._build_scoring_graph(make_link_prediction_split())
        edges = [(0, 2), (1, 5)]

        common_scores = heuristics.score_edges(
            graph,
            edges,
            method=HeuristicMethod.COMMON_NEIGHBORS,
            ppr_alpha=0.85,
        )
        adamic_scores = heuristics.score_edges(
            graph,
            edges,
            method=HeuristicMethod.ADAMIC_ADAR,
            ppr_alpha=0.85,
        )
        ppr_scores = heuristics.score_edges(
            graph,
            edges,
            method=HeuristicMethod.PERSONALIZED_PAGERANK,
            ppr_alpha=0.85,
        )

        np.testing.assert_array_equal(common_scores, np.asarray([2.0, 0.0]))
        self.assertGreater(adamic_scores[0], adamic_scores[1])
        self.assertGreater(ppr_scores[0], ppr_scores[1])

    def test_score_split_with_heuristic_returns_aligned_outputs(self) -> None:
        split = make_link_prediction_split()
        y_val_true, y_val_score, y_test_true, y_test_score = heuristics.score_split_with_heuristic(
            split,
            method=HeuristicMethod.COMMON_NEIGHBORS,
        )

        self.assertEqual(y_val_true.shape, y_val_score.shape)
        self.assertEqual(y_test_true.shape, y_test_score.shape)
        np.testing.assert_array_equal(y_val_true, np.asarray([1, 0]))
        np.testing.assert_array_equal(y_test_true, np.asarray([1, 0]))

    def test_score_edges_rejects_unsupported_method(self) -> None:
        graph = heuristics._build_scoring_graph(make_link_prediction_split())

        with self.assertRaisesRegex(ValueError, "unsupported heuristic method"):
            heuristics.score_edges(graph, [(0, 1)], method="invalid", ppr_alpha=0.85)  # type: ignore[arg-type]


class HeuristicBaselineTests(unittest.TestCase):
    def test_run_heuristic_baseline_returns_standard_result(self) -> None:
        split = make_link_prediction_split()
        result = heuristics.run_heuristic_baseline(
            split,
            method=HeuristicMethod.ADAMIC_ADAR,
        )

        self.assertEqual(result.dataset, "Cora")
        self.assertEqual(result.method, "adamic_adar")
        self.assertEqual(result.params["split_seed"], 42)
        self.assertIn("roc_auc", result.validation)
        self.assertIn("threshold", result.test)

    def test_run_heuristic_baseline_ppr_records_alpha_and_custom_seed(self) -> None:
        split = make_link_prediction_split()

        with patch.object(
            heuristics,
            "score_split_with_heuristic",
            return_value=(
                np.asarray([1, 0]),
                np.asarray([0.9, 0.1]),
                np.asarray([1, 0]),
                np.asarray([0.8, 0.2]),
            ),
        ), patch.object(
            heuristics,
            "evaluate_with_validation_threshold",
            return_value=(make_metrics(threshold=0.5), make_metrics(threshold=0.5)),
        ):
            result = heuristics.run_heuristic_baseline(
                split,
                method=HeuristicMethod.PERSONALIZED_PAGERANK,
                seed=9,
                ppr_alpha=0.9,
            )

        self.assertEqual(result.seed, 9)
        self.assertEqual(result.params["alpha"], 0.9)
        self.assertEqual(result.params["train_graph_edges"], split.train_graph_edge_count)


if __name__ == "__main__":
    unittest.main()
