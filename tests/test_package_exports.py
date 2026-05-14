"""Verify that the public package surface matches what runners import."""

from __future__ import annotations

import unittest

import citegnn.baselines as baselines
import citegnn.data as data
import citegnn.utils as utils
from citegnn.baselines import (
    FeatureOnlyBaselineConfig,
    GNNBaselineConfig,
    GNNMethod,
    HeuristicConfig,
    HeuristicMethod,
    Node2VecConfig,
    compute_structural_features,
    run_feature_only_baseline,
    run_gnn_baseline,
    run_heuristic_baseline,
    run_node2vec_baseline,
)
from citegnn.data import (
    DEFAULT_K,
    HeartSplit,
    LinkPredictionSplit,
    create_heart_split,
    load_heart_split,
)
from citegnn.utils import (
    ClassificationMetrics,
    RankingMetrics,
    compute_classification_metrics,
    compute_ranking_metrics,
    seed_everything,
    write_json,
)


class PackageExportsTests(unittest.TestCase):
    def test_baselines_package_reexports_public_api(self) -> None:
        for name in (
            "FeatureOnlyBaselineConfig",
            "GNNBaselineConfig",
            "GNNMethod",
            "HeuristicConfig",
            "HeuristicMethod",
            "Node2VecConfig",
            "compute_structural_features",
            "run_feature_only_baseline",
            "run_gnn_baseline",
            "run_heuristic_baseline",
            "run_node2vec_baseline",
        ):
            self.assertIn(name, baselines.__all__)
        self.assertIs(baselines.FeatureOnlyBaselineConfig, FeatureOnlyBaselineConfig)
        self.assertIs(baselines.GNNBaselineConfig, GNNBaselineConfig)
        self.assertIs(baselines.HeuristicConfig, HeuristicConfig)
        self.assertIs(baselines.HeuristicMethod, HeuristicMethod)
        self.assertIs(baselines.Node2VecConfig, Node2VecConfig)
        self.assertIs(baselines.run_feature_only_baseline, run_feature_only_baseline)
        self.assertIs(baselines.run_gnn_baseline, run_gnn_baseline)
        self.assertIs(baselines.run_heuristic_baseline, run_heuristic_baseline)
        self.assertIs(baselines.run_node2vec_baseline, run_node2vec_baseline)

    def test_data_package_reexports_public_api(self) -> None:
        for name in ("HeartSplit", "LinkPredictionSplit", "create_heart_split", "load_heart_split", "DEFAULT_K"):
            self.assertIn(name, data.__all__)
        self.assertIs(data.HeartSplit, HeartSplit)
        # LinkPredictionSplit is a backwards-compat alias.
        self.assertIs(data.LinkPredictionSplit, HeartSplit)
        self.assertIs(data.create_heart_split, create_heart_split)
        self.assertIs(data.load_heart_split, load_heart_split)
        self.assertEqual(DEFAULT_K, 500)

    def test_utils_package_reexports_public_api(self) -> None:
        for name in (
            "ClassificationMetrics",
            "RankingMetrics",
            "compute_classification_metrics",
            "compute_ranking_metrics",
            "seed_everything",
            "write_json",
        ):
            self.assertIn(name, utils.__all__)
        self.assertIs(utils.ClassificationMetrics, ClassificationMetrics)
        self.assertIs(utils.RankingMetrics, RankingMetrics)
        self.assertIs(utils.compute_classification_metrics, compute_classification_metrics)
        self.assertIs(utils.compute_ranking_metrics, compute_ranking_metrics)
        self.assertIs(utils.seed_everything, seed_everything)
        self.assertIs(utils.write_json, write_json)


if __name__ == "__main__":
    unittest.main()
