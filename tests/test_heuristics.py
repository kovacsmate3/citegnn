from __future__ import annotations

import unittest

import numpy as np

from citegnn.baselines import heuristics
from citegnn.baselines.heuristics import HeuristicMethod
from tests.support import make_link_prediction_split


class HeuristicHelperTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
