"""End-to-end smoke tests for every baseline on a small synthetic HeaRT split.

The baselines normally load Cora/PubMed from disk; here
``load_citation_graph`` is patched to return a 30-node synthetic graph, so
the tests run offline in a few seconds. They check that each training and
evaluation path runs and produces well-formed HeaRT metrics, not that the
models learn anything useful.
"""

from __future__ import annotations

import io
import math
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import torch

from citegnn.baselines.augmentation import augment_features, compute_structural_features
from citegnn.baselines.feature_only import FeatureOnlyBaselineConfig, run_feature_only_baseline
from citegnn.baselines.gnn import GNNBaselineConfig, GNNMethod, run_gnn_baseline
from citegnn.baselines.heuristics import HeuristicConfig, HeuristicMethod, run_heuristic_baseline
from citegnn.baselines.node2vec import Node2VecConfig, run_node2vec_baseline
from citegnn.data import smoke
from citegnn.data.heart_split import HeartSplit, create_heart_split, load_heart_split
from citegnn.data.loaders import GraphData
from citegnn.evaluation import ExperimentResult
from tests.support import edge_index_from_pairs, workspace_temp_dir

NUM_NODES = 30
NUM_FEATURES = 8
K = 6


def make_synthetic_graph() -> GraphData:
    """A ring with chords and random features: connected, non-trivial spectrum."""

    generator = torch.Generator().manual_seed(0)
    x = torch.rand((NUM_NODES, NUM_FEATURES), generator=generator)
    pairs = [(i, (i + 1) % NUM_NODES) for i in range(NUM_NODES)]
    pairs += [(i, (i + 5) % NUM_NODES) for i in range(0, NUM_NODES, 2)]
    pairs += [(i, (i + 11) % NUM_NODES) for i in range(0, NUM_NODES, 3)]
    edge_index = edge_index_from_pairs(pairs, bidirectional=True)
    mask = torch.zeros(NUM_NODES, dtype=torch.bool)
    return GraphData(
        name="Cora",
        num_nodes=NUM_NODES,
        num_features=NUM_FEATURES,
        x=x,
        edge_index=edge_index,
        original_edge_index=edge_index.clone(),
        train_mask=mask,
        val_mask=mask.clone(),
        test_mask=mask.clone(),
    )


GRAPH = make_synthetic_graph()


def make_synthetic_split() -> HeartSplit:
    with redirect_stdout(io.StringIO()):
        return create_heart_split(
            GRAPH, seed=3, K=K, split_ratio=(0.7, 0.15, 0.15), ppr_iterations=10
        )


SPLIT = make_synthetic_split()


def _patched_graph(*args: object, **kwargs: object) -> GraphData:
    return GRAPH


class BaselineResultAssertions(unittest.TestCase):
    def assertWellFormed(self, result: ExperimentResult, method: str) -> None:
        self.assertEqual(result.dataset, "Cora")
        self.assertEqual(result.method, method)
        self.assertEqual(result.seed, SPLIT.seed)
        for metrics in (result.validation, result.test):
            for key in ("mrr", "hits_at_1", "hits_at_10", "roc_auc"):
                self.assertIn(key, metrics)
                self.assertTrue(math.isfinite(metrics[key]), key)
            self.assertGreater(metrics["mrr"], 0.0)
            self.assertLessEqual(metrics["mrr"], 1.0)
            self.assertGreaterEqual(metrics["hits_at_10"], metrics["hits_at_1"])


class HeuristicBaselineTests(BaselineResultAssertions):
    def test_every_heuristic_runs(self) -> None:
        for method in HeuristicMethod:
            with self.subTest(method=method.value):
                result = run_heuristic_baseline(
                    SPLIT, method=method, config=HeuristicConfig(ppr_iterations=10)
                )
                self.assertWellFormed(result, method.value)
                self.assertEqual(result.params["K"], K)

    def test_ppr_records_its_configuration(self) -> None:
        result = run_heuristic_baseline(
            SPLIT,
            method=HeuristicMethod.PERSONALIZED_PAGERANK,
            config=HeuristicConfig(ppr_alpha=0.5, ppr_iterations=5),
        )
        self.assertEqual(result.params["ppr_alpha"], 0.5)
        self.assertEqual(result.params["ppr_iterations"], 5)


class Node2VecBaselineTests(BaselineResultAssertions):
    def test_node2vec_runs_with_biased_walks(self) -> None:
        config = Node2VecConfig(
            embedding_dim=8, walk_length=6, walks_per_node=2, epochs=1, batch_size=64, p=0.5, q=2.0
        )
        result = run_node2vec_baseline(SPLIT, config=config)
        self.assertWellFormed(result, "node2vec")
        self.assertEqual(result.params["embedding_dim"], 8)

    def test_invalid_configuration_is_rejected(self) -> None:
        for field in ("embedding_dim", "walk_length", "epochs", "learning_rate", "q"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                Node2VecConfig(**{field: 0}).validate()


@mock.patch("citegnn.baselines.feature_only.load_citation_graph", _patched_graph)
class FeatureOnlyBaselineTests(BaselineResultAssertions):
    def test_every_feature_mode_runs(self) -> None:
        for feature_mode in ("concat", "hadamard", "abs_diff", "hadamard_abs_diff"):
            with self.subTest(feature_mode=feature_mode):
                config = FeatureOnlyBaselineConfig(
                    feature_mode=feature_mode,
                    hidden_dim=8,
                    epochs=3,
                    patience=1,
                    batch_size=16,
                    train_negatives_per_positive=2,
                    show_progress=False,
                )
                result = run_feature_only_baseline(SPLIT, config=config)
                self.assertWellFormed(result, "feature_only_mlp")
                self.assertEqual(result.params["feature_mode"], feature_mode)

    def test_progress_output(self) -> None:
        config = FeatureOnlyBaselineConfig(hidden_dim=8, epochs=1, show_progress=True)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = run_feature_only_baseline(SPLIT, config=config)
        self.assertWellFormed(result, "feature_only_mlp")

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid = [
            {"feature_mode": "sum"},
            {"hidden_dim": 0},
            {"dropout": 1.0},
            {"learning_rate": 0.0},
            {"weight_decay": -1.0},
            {"epochs": 0},
            {"patience": 0},
            {"eval_every": 0},
            {"train_negatives_per_positive": 0},
            {"batch_size": 0},
        ]
        for kwargs in invalid:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                FeatureOnlyBaselineConfig(**kwargs).validate()


@mock.patch("citegnn.baselines.gnn.load_citation_graph", _patched_graph)
class GNNBaselineTests(BaselineResultAssertions):
    @staticmethod
    def _config(**overrides: object) -> GNNBaselineConfig:
        params: dict[str, object] = {
            "hidden_dim": 8,
            "decoder_hidden_dim": 8,
            "num_layers": 2,
            "epochs": 3,
            "patience": 1,
            "eval_every": 1,
            "train_batch_size": 8,
            "gat_heads": 2,
        }
        params.update(overrides)
        return GNNBaselineConfig(**params)  # type: ignore[arg-type]

    def test_every_encoder_decoder_and_training_mode_runs(self) -> None:
        for method in GNNMethod:
            for decoder in ("hadamard_mlp", "dot"):
                for train_mode in ("fullbatch", "minibatch_dropedge"):
                    with self.subTest(method=method.value, decoder=decoder, train_mode=train_mode):
                        config = self._config(decoder=decoder, train_mode=train_mode)
                        result = run_gnn_baseline(SPLIT, method=method, config=config)
                        self.assertWellFormed(result, method.value)
                        self.assertEqual(result.params["input_feature_dim"], NUM_FEATURES)
                        self.assertEqual(result.params["train_mode"], train_mode)

    def test_single_layer_encoders_run(self) -> None:
        for method in GNNMethod:
            with self.subTest(method=method.value):
                result = run_gnn_baseline(SPLIT, method=method, config=self._config(num_layers=1))
                self.assertWellFormed(result, method.value)

    def test_structural_features_widen_the_input(self) -> None:
        config = self._config(use_structural_features=True, num_laplacian_eigvecs=3)
        result = run_gnn_baseline(
            SPLIT, method=GNNMethod.GCN, config=config, method_name_override="gcn_pe"
        )
        self.assertWellFormed(result, "gcn_pe")
        self.assertEqual(result.params["input_feature_dim"], NUM_FEATURES + 1 + 3)

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid = [
            {"hidden_dim": 0},
            {"num_layers": 0},
            {"dropout": 1.0},
            {"decoder_hidden_dim": 0},
            {"learning_rate": 0.0},
            {"weight_decay": -1.0},
            {"epochs": 0},
            {"patience": 0},
            {"eval_every": 0},
            {"decoder": "bilinear"},
            {"train_negatives_per_positive": 0},
            {"train_mode": "sampled"},
            {"train_batch_size": 0},
            {"num_laplacian_eigvecs": -1},
            {"gat_heads": 0},
            {"gat_attention_dropout": 1.0},
        ]
        for kwargs in invalid:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                GNNBaselineConfig(**kwargs).validate()  # type: ignore[arg-type]


class StructuralFeatureTests(unittest.TestCase):
    def test_log_degree_and_laplacian_columns(self) -> None:
        features = compute_structural_features(
            num_nodes=NUM_NODES, train_edge_index=SPLIT.train_edge_index, num_eigvecs=4
        )
        self.assertEqual(tuple(features.shape), (NUM_NODES, 5))
        self.assertTrue(torch.isfinite(features).all())

    def test_zero_eigenvectors_gives_log_degree_only(self) -> None:
        features = compute_structural_features(
            num_nodes=NUM_NODES, train_edge_index=SPLIT.train_edge_index, num_eigvecs=0
        )
        self.assertEqual(tuple(features.shape), (NUM_NODES, 1))

    def test_more_eigenvectors_than_nodes_are_zero_padded(self) -> None:
        edges = edge_index_from_pairs([(0, 1), (1, 2), (2, 3)], bidirectional=True)
        features = compute_structural_features(num_nodes=4, train_edge_index=edges, num_eigvecs=6)
        self.assertEqual(tuple(features.shape), (4, 7))

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_structural_features(num_nodes=0, train_edge_index=SPLIT.train_edge_index)
        with self.assertRaises(ValueError):
            compute_structural_features(
                num_nodes=NUM_NODES, train_edge_index=SPLIT.train_edge_index, num_eigvecs=-1
            )

    def test_augment_features_concatenates_columns(self) -> None:
        structural = torch.ones((NUM_NODES, 2))
        augmented = augment_features(GRAPH.x, structural=structural)
        self.assertEqual(tuple(augmented.shape), (NUM_NODES, NUM_FEATURES + 2))
        with self.assertRaises(ValueError):
            augment_features(GRAPH.x[0], structural=structural)
        with self.assertRaises(ValueError):
            augment_features(GRAPH.x, structural=structural[0])
        with self.assertRaises(ValueError):
            augment_features(GRAPH.x, structural=structural[:-1])


@mock.patch("citegnn.data.smoke.load_citation_graph", _patched_graph)
class SplitCliTests(unittest.TestCase):
    def test_cli_builds_and_saves_a_split(self) -> None:
        with workspace_temp_dir("smoke-cli") as tmp_dir:
            argv = [
                "smoke",
                "--dataset",
                "Cora",
                "--seed",
                "5",
                "--K",
                "4",
                "--train-ratio",
                "0.7",
                "--val-ratio",
                "0.15",
                "--test-ratio",
                "0.15",
                "--ppr-iterations",
                "10",
                "--output-dir",
                str(tmp_dir),
            ]
            stdout = io.StringIO()
            with mock.patch.object(sys, "argv", argv), redirect_stdout(stdout):
                smoke.main()
            artifacts = list(tmp_dir.glob("*.pt"))
            self.assertEqual(len(artifacts), 1)
            split = load_heart_split(artifacts[0])
            self.assertEqual(split.K, 4)
            self.assertEqual(split.seed, 5)
            self.assertIn('"num_test_positives"', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
