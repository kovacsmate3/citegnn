"""Tests for the HeaRT-aware evaluation pipeline and JSON IO."""

from __future__ import annotations

import json
import unittest

import torch

from citegnn.evaluation import (
    ExperimentResult,
    build_experiment_result,
    describe_experiment_result,
    evaluate_with_heart_negatives,
    save_and_describe_experiment_result,
    save_experiment_result,
    score_pos_and_heart_negatives,
)
from tests.support import (
    make_experiment_result,
    make_heart_split,
    workspace_temp_dir,
)


class HeartScoringTests(unittest.TestCase):
    def test_score_pos_and_heart_negatives_layout(self) -> None:
        split = make_heart_split(K=4)

        def score_fn(edges):
            # Score each edge by `u + v` so we can predict the layout.
            return (edges[0] + edges[1]).to(dtype=torch.float64)

        pos_scores, neg_scores = score_pos_and_heart_negatives(
            pos_edge_index=split.val_pos_edge_index,
            neg_nodes=split.val_neg_nodes,
            score_fn=score_fn,
            batch_size=8,
        )
        self.assertEqual(pos_scores.shape, (split.num_val_positives,))
        self.assertEqual(neg_scores.shape, (split.num_val_positives, split.K))
        # Positive (0, 2) -> score 2.
        self.assertAlmostEqual(pos_scores.item(), 2.0)
        # First half negatives are paired with the source (0); second half with target (2).
        # neg_nodes[0] = [5, 4, 3, 5] -> first half [5, 4], second half [3, 5].
        expected_neg = torch.tensor([0 + 5, 0 + 4, 3 + 2, 5 + 2], dtype=torch.float64)
        self.assertTrue(torch.allclose(neg_scores[0], expected_neg))

    def test_evaluate_with_heart_negatives_reports_ranking_and_classification(self) -> None:
        split = make_heart_split(K=4)

        # Score the true positive higher than every negative so we get a perfect ranking.
        def score_fn(edges):
            scores = torch.zeros(edges.size(1), dtype=torch.float64)
            mask = (edges[0] == 0) & (edges[1] == 2)
            scores[mask] = 10.0
            mask2 = (edges[0] == 1) & (edges[1] == 3)
            scores[mask2] = 10.0
            return scores

        validation, test = evaluate_with_heart_negatives(
            score_fn=score_fn,
            val_pos_edge_index=split.val_pos_edge_index,
            val_neg_nodes=split.val_neg_nodes,
            test_pos_edge_index=split.test_pos_edge_index,
            test_neg_nodes=split.test_neg_nodes,
        )
        self.assertAlmostEqual(validation["mrr"], 1.0)
        self.assertAlmostEqual(test["mrr"], 1.0)
        self.assertEqual(int(validation["num_positives"]), 1)
        self.assertEqual(int(test["num_negatives_per_positive"]), 4)
        self.assertIn("roc_auc", validation)
        self.assertIn("hits_at_10", test)


class ExperimentResultTests(unittest.TestCase):
    def test_describe_uses_mrr_by_default(self) -> None:
        result = make_experiment_result(method="common_neighbors")
        line = describe_experiment_result(result)
        self.assertIn("test_mrr=", line)
        self.assertIn("test_hits@10=", line)

    def test_save_and_load_round_trip(self) -> None:
        result = build_experiment_result(
            dataset="Cora",
            method="adamic_adar",
            seed=7,
            params={"alpha": 0.85},
            validation={"mrr": 0.21, "hits_at_10": 0.42, "roc_auc": 0.7},
            test={"mrr": 0.22, "hits_at_10": 0.45, "roc_auc": 0.72},
        )
        with workspace_temp_dir("results") as tmp_dir:
            path = save_experiment_result(result, output_dir=tmp_dir)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["method"], "adamic_adar")
        self.assertEqual(payload["seed"], 7)
        self.assertAlmostEqual(payload["test"]["mrr"], 0.22)

    def test_save_and_describe_returns_summary(self) -> None:
        result = make_experiment_result(method="gcn")
        with workspace_temp_dir("results") as tmp_dir:
            _, summary = save_and_describe_experiment_result(result, output_dir=tmp_dir)
        self.assertIsInstance(summary, str)
        self.assertIn("Cora gcn:", summary)


if __name__ == "__main__":
    unittest.main()
