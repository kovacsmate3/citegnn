from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from citegnn.baselines.heuristics import HeuristicMethod
from citegnn.baselines.gnn import GNNMethod
from citegnn.data import smoke
from tests.support import (
    load_script_module,
    make_experiment_result,
    make_graph_data,
    make_link_prediction_split,
)


class SmokeCliTests(unittest.TestCase):
    def test_smoke_main_prints_json_summary_from_pipeline_outputs(self) -> None:
        graph = make_graph_data()
        split = make_link_prediction_split()
        fake_artifact = Path("artifacts/splits/cora_seed42_lp.pt")

        stdout = io.StringIO()
        with patch("citegnn.data.smoke.seed_everything") as seed_mock, patch(
            "citegnn.data.smoke.load_citation_graph",
            return_value=graph,
        ) as load_graph_mock, patch(
            "citegnn.data.smoke.create_link_prediction_split",
            return_value=split,
        ) as split_mock, patch(
            "citegnn.data.smoke.save_split_artifact",
            return_value=fake_artifact,
        ), patch(
            "sys.argv",
            [
                "smoke.py",
                "--dataset",
                "Cora",
                "--seed",
                "7",
                "--output-dir",
                "tmp-artifacts",
            ],
        ), redirect_stdout(stdout):
            smoke.main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["dataset"], "Cora")
        self.assertEqual(payload["artifact_path"], str(fake_artifact))
        seed_mock.assert_called_once_with(7)
        load_graph_mock.assert_called_once()
        split_mock.assert_called_once()


class BaselineScriptIntegrationTests(unittest.TestCase):
    def test_run_heuristic_script_runs_all_methods(self) -> None:
        module = load_script_module(
            "tests.run_heuristic_baselines_script",
            "scripts/run_heuristic_baselines.py",
        )
        split = make_link_prediction_split()
        args = SimpleNamespace(
            split_path=Path("dummy.pt"),
            method="all",
            output_dir=Path("results"),
            ppr_alpha=0.9,
        )

        def build_result(_split, *, method, ppr_alpha):
            self.assertEqual(ppr_alpha, 0.9)
            return make_experiment_result(method.value)

        stdout = io.StringIO()
        with patch.object(module, "parse_args", return_value=args), patch.object(
            module,
            "load_split_artifact",
            return_value=split,
        ), patch.object(
            module,
            "run_heuristic_baseline",
            side_effect=build_result,
        ) as run_mock, patch.object(
            module,
            "save_and_describe_experiment_result",
            side_effect=lambda result, output_dir: (
                output_dir / f"{result.method}.json",
                f"{result.method} -> {output_dir / f'{result.method}.json'}",
            ),
        ), redirect_stdout(stdout):
            module.main()

        self.assertEqual(run_mock.call_count, len(HeuristicMethod))
        printed = stdout.getvalue()
        for method in HeuristicMethod:
            self.assertIn(method.value, printed)

    def test_run_node2vec_script_builds_config_and_saves_result(self) -> None:
        module = load_script_module(
            "tests.run_node2vec_baseline_script",
            "scripts/run_node2vec_baseline.py",
        )
        args = SimpleNamespace(
            split_path=Path("dummy.pt"),
            output_dir=Path("results"),
            embedding_dim=8,
            walk_length=4,
            walks_per_node=2,
            context_window=1,
            negative_samples=2,
            epochs=1,
            batch_size=16,
            learning_rate=0.01,
            p=1.0,
            q=2.0,
            classifier_c=0.5,
        )

        stdout = io.StringIO()
        with patch.object(module, "parse_args", return_value=args), patch.object(
            module,
            "load_split_artifact",
            return_value=make_link_prediction_split(),
        ), patch.object(
            module,
            "run_node2vec_baseline",
            return_value=make_experiment_result("node2vec"),
        ) as run_mock, patch.object(
            module,
            "save_and_describe_experiment_result",
            return_value=(
                Path("results/cora/node2vec_seed42.json"),
                "node2vec -> results/cora/node2vec_seed42.json",
            ),
        ), redirect_stdout(stdout):
            module.main()

        config = run_mock.call_args.kwargs["config"]
        self.assertEqual(config.embedding_dim, 8)
        self.assertEqual(config.q, 2.0)
        self.assertEqual(config.classifier_c, 0.5)
        self.assertIn("node2vec", stdout.getvalue())

    def test_run_feature_only_script_builds_config_and_saves_result(self) -> None:
        module = load_script_module(
            "tests.run_feature_only_baseline_script",
            "scripts/run_feature_only_baseline.py",
        )
        args = SimpleNamespace(
            split_path=Path("dummy.pt"),
            output_dir=Path("results"),
            data_root="data",
            feature_mode="hadamard_abs_diff",
            hidden_dim=32,
            dropout=0.1,
            learning_rate=0.001,
            weight_decay=0.0001,
            epochs=4,
            patience=2,
            batch_size=16,
        )

        stdout = io.StringIO()
        with patch.object(module, "parse_args", return_value=args), patch.object(
            module,
            "load_split_artifact",
            return_value=make_link_prediction_split(),
        ), patch.object(
            module,
            "run_feature_only_baseline",
            return_value=make_experiment_result("feature_only_mlp"),
        ) as run_mock, patch.object(
            module,
            "save_and_describe_experiment_result",
            return_value=(
                Path("results/cora/feature_only_mlp_seed42.json"),
                "feature_only_mlp -> results/cora/feature_only_mlp_seed42.json",
            ),
        ), redirect_stdout(stdout):
            module.main()

        config = run_mock.call_args.kwargs["config"]
        self.assertEqual(config.feature_mode, "hadamard_abs_diff")
        self.assertEqual(config.hidden_dim, 32)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(run_mock.call_args.kwargs["data_root"], "data")
        self.assertIn("feature_only_mlp", stdout.getvalue())

    def test_run_gnn_script_runs_all_requested_methods(self) -> None:
        module = load_script_module(
            "tests.run_gnn_baselines_script",
            "scripts/run_gnn_baselines.py",
        )
        args = SimpleNamespace(
            split_path=Path("dummy.pt"),
            method="all",
            output_dir=Path("results"),
            data_root="data",
            hidden_dim=4,
            num_layers=2,
            dropout=0.0,
            decoder_hidden_dim=4,
            learning_rate=0.01,
            weight_decay=0.0,
            epochs=2,
            patience=1,
        )

        def build_result(_split, *, method, config, data_root):
            self.assertEqual(config.hidden_dim, 4)
            self.assertEqual(data_root, "data")
            return make_experiment_result(method.value)

        stdout = io.StringIO()
        with patch.object(module, "parse_args", return_value=args), patch.object(
            module,
            "load_split_artifact",
            return_value=make_link_prediction_split(),
        ), patch.object(
            module,
            "run_gnn_baseline",
            side_effect=build_result,
        ) as run_mock, patch.object(
            module,
            "save_and_describe_experiment_result",
            side_effect=lambda result, output_dir: (
                output_dir / f"{result.method}.json",
                f"{result.method} -> {output_dir / f'{result.method}.json'}",
            ),
        ), redirect_stdout(stdout):
            module.main()

        self.assertEqual(run_mock.call_count, len(GNNMethod))
        printed = stdout.getvalue()
        for method in GNNMethod:
            self.assertIn(method.value, printed)

    def test_run_baseline_ablation_script_includes_feature_only_baseline(self) -> None:
        module = load_script_module(
            "tests.run_baseline_ablation_script",
            "scripts/run_baseline_ablation.py",
        )
        args = SimpleNamespace(
            datasets=["Cora"],
            seed=42,
            data_root=Path("data"),
            split_output_dir=Path("artifacts/splits"),
            result_output_dir=Path("results"),
            gnn_hidden_dim=4,
            gnn_decoder_hidden_dim=4,
            gnn_epochs=2,
            gnn_patience=1,
            gnn_num_layers_deep=4,
            node2vec_epochs=1,
            feature_only_hidden_dim=16,
            feature_only_epochs=3,
            feature_only_patience=2,
        )

        stdout = io.StringIO()
        with patch.object(module, "parse_args", return_value=args), patch.object(
            module,
            "_build_split",
            return_value=make_link_prediction_split(),
        ), patch.object(
            module,
            "run_heuristic_baseline",
            side_effect=lambda split, method: make_experiment_result(method.value),
        ), patch.object(
            module,
            "run_node2vec_baseline",
            return_value=make_experiment_result("node2vec"),
        ), patch.object(
            module,
            "run_feature_only_baseline",
            return_value=make_experiment_result("feature_only_mlp"),
        ) as feature_only_mock, patch.object(
            module,
            "run_gnn_baseline",
            side_effect=lambda split, method, config, data_root, method_name_override=None: make_experiment_result(method.value),
        ), patch.object(
            module,
            "_print_saved_result",
            side_effect=lambda output_dir, result: print(result.method),
        ), patch.object(module, "seed_everything"), redirect_stdout(stdout):
            module.main()

        config = feature_only_mock.call_args.kwargs["config"]
        self.assertEqual(config.hidden_dim, 16)
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.patience, 2)
        self.assertIn("feature_only_mlp", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
