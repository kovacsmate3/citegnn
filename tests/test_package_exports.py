from __future__ import annotations

import unittest

import citegnn
from citegnn import baselines, data, utils
from citegnn.baselines.feature_only import (
    FeatureOnlyBaselineConfig,
    run_feature_only_baseline,
)
from citegnn.baselines.gnn import GNNBaselineConfig, run_gnn_baseline
from citegnn.baselines.heuristics import HeuristicMethod
from citegnn.baselines.node2vec import Node2VecConfig, run_node2vec_baseline
from citegnn.data.loaders import CitationDataset, GraphData, load_citation_graph
from citegnn.data.splits import LinkPredictionSplit, create_link_prediction_split, load_split_artifact, save_split_artifact
from citegnn.utils.io import write_json
from citegnn.utils.metrics import ClassificationMetrics, choose_best_threshold, compute_classification_metrics
from citegnn.utils.seed import seed_everything


class PackageExportTests(unittest.TestCase):
    def test_root_package_has_expected_docstring(self) -> None:
        self.assertEqual(citegnn.__doc__, "CiteGNN package.")

    def test_data_package_reexports_public_api(self) -> None:
        self.assertIn("CitationDataset", data.__all__)
        self.assertIs(data.CitationDataset, CitationDataset)
        self.assertIs(data.GraphData, GraphData)
        self.assertIs(data.LinkPredictionSplit, LinkPredictionSplit)
        self.assertIs(data.create_link_prediction_split, create_link_prediction_split)
        self.assertIs(data.load_citation_graph, load_citation_graph)
        self.assertIs(data.load_split_artifact, load_split_artifact)
        self.assertIs(data.save_split_artifact, save_split_artifact)

    def test_utils_package_reexports_public_api(self) -> None:
        self.assertIn("ClassificationMetrics", utils.__all__)
        self.assertIs(utils.ClassificationMetrics, ClassificationMetrics)
        self.assertIs(utils.choose_best_threshold, choose_best_threshold)
        self.assertIs(utils.compute_classification_metrics, compute_classification_metrics)
        self.assertIs(utils.seed_everything, seed_everything)
        self.assertIs(utils.write_json, write_json)

    def test_baselines_package_reexports_public_api(self) -> None:
        self.assertIn("FeatureOnlyBaselineConfig", baselines.__all__)
        self.assertIn("GNNBaselineConfig", baselines.__all__)
        self.assertIs(baselines.FeatureOnlyBaselineConfig, FeatureOnlyBaselineConfig)
        self.assertIs(baselines.GNNBaselineConfig, GNNBaselineConfig)
        self.assertIs(baselines.HeuristicMethod, HeuristicMethod)
        self.assertIs(baselines.Node2VecConfig, Node2VecConfig)
        self.assertIs(baselines.run_feature_only_baseline, run_feature_only_baseline)
        self.assertIs(baselines.run_gnn_baseline, run_gnn_baseline)
        self.assertIs(baselines.run_node2vec_baseline, run_node2vec_baseline)


if __name__ == "__main__":
    unittest.main()
