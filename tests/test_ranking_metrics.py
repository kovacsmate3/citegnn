"""Tests for the HeaRT-style MRR / Hits@K metrics."""

from __future__ import annotations

import unittest

import torch

from citegnn.utils.ranking_metrics import (
    DEFAULT_HITS_KS,
    compute_ranking_metrics,
)


class RankingMetricsTests(unittest.TestCase):
    def test_positive_strictly_above_all_negatives_gives_perfect_metrics(self) -> None:
        pos = torch.tensor([1.0, 1.0])
        neg = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        metrics = compute_ranking_metrics(pos, neg, hits_ks=(1, 3, 10))
        self.assertAlmostEqual(metrics.mrr, 1.0)
        self.assertEqual(metrics.hits_at[1], 1.0)
        self.assertEqual(metrics.hits_at[3], 1.0)

    def test_positive_below_all_negatives_gives_worst_rank(self) -> None:
        pos = torch.tensor([0.0])
        neg = torch.tensor([[1.0, 2.0, 3.0]])
        metrics = compute_ranking_metrics(pos, neg, hits_ks=(1, 3))
        # Rank should be 4 (all 3 negatives strictly greater, plus 1).
        self.assertAlmostEqual(metrics.mrr, 1.0 / 4.0)
        self.assertEqual(metrics.hits_at[1], 0.0)
        self.assertEqual(metrics.hits_at[3], 0.0)

    def test_ties_use_average_rank(self) -> None:
        pos = torch.tensor([1.0])
        neg = torch.tensor([[1.0, 1.0, 1.0]])
        metrics = compute_ranking_metrics(pos, neg, hits_ks=(1, 3))
        # Average ranking: 0.5 * (#strict + #geq) + 1 = 0.5 * (0 + 3) + 1 = 2.5
        self.assertAlmostEqual(metrics.mrr, 1.0 / 2.5)
        self.assertEqual(metrics.hits_at[1], 0.0)
        self.assertEqual(metrics.hits_at[3], 1.0)

    def test_mean_rank_across_multiple_positives(self) -> None:
        pos = torch.tensor([0.5, 1.5])
        neg = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        # First positive: 3 negatives strictly greater -> rank 4 -> 1/4
        # Second positive: 0 negatives strictly greater -> rank 1 -> 1
        metrics = compute_ranking_metrics(pos, neg, hits_ks=(1, 3))
        self.assertAlmostEqual(metrics.mrr, (0.25 + 1.0) / 2.0)
        self.assertAlmostEqual(metrics.hits_at[1], 0.5)
        self.assertAlmostEqual(metrics.hits_at[3], 0.5)

    def test_to_dict_payload_includes_default_hits_keys(self) -> None:
        pos = torch.tensor([1.0])
        neg = torch.tensor([[0.0, 0.0]])
        metrics = compute_ranking_metrics(pos, neg)
        payload = metrics.to_dict()
        for k in DEFAULT_HITS_KS:
            self.assertIn(f"hits_at_{k}", payload)
        self.assertIn("mrr", payload)
        self.assertEqual(payload["num_negatives_per_positive"], 2)
        self.assertEqual(payload["num_positives"], 1)

    def test_shape_mismatches_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "pos_scores"):
            compute_ranking_metrics(torch.tensor([[1.0]]), torch.tensor([[1.0]]))
        with self.assertRaisesRegex(ValueError, "neg_scores"):
            compute_ranking_metrics(torch.tensor([1.0]), torch.tensor([1.0]))
        with self.assertRaisesRegex(ValueError, "same number of positives"):
            compute_ranking_metrics(torch.tensor([1.0, 2.0]), torch.tensor([[0.0]]))


if __name__ == "__main__":
    unittest.main()
