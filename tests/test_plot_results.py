from __future__ import annotations

import json
from pathlib import Path
import unittest

from tests.support import load_script_module, workspace_temp_dir


plot_results = load_script_module(
    "tests.plot_results_script",
    "scripts/plot_results.py",
)


def _write_result(
    path: Path,
    *,
    dataset: str,
    method: str,
    score: float,
    seed: int = 42,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "params": {},
        "validation": {"roc_auc": score},
        "test": {
            "roc_auc": score,
            "average_precision": score - 0.1,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class PlotResultLoadingTests(unittest.TestCase):
    def test_load_result_records_exposes_metrics_and_paths(self) -> None:
        with workspace_temp_dir("plot-result-records") as tmp_dir:
            _write_result(
                tmp_dir / "cora" / "gcn_seed42.json",
                dataset="Cora",
                method="gcn",
                score=0.71,
            )

            records = plot_results.load_result_records(tmp_dir)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["dataset"], "Cora")
        self.assertEqual(records[0]["method"], "GCN")
        self.assertEqual(records[0]["test_roc_auc"], 0.71)
        self.assertEqual(records[0]["test_average_precision"], 0.61)
        self.assertEqual(Path(records[0]["relative_path"]), Path("cora") / "gcn_seed42.json")

    def test_load_results_reads_nested_dataset_directories(self) -> None:
        with workspace_temp_dir("plot-results") as tmp_dir:
            _write_result(
                tmp_dir / "cora" / "gcn_seed42.json",
                dataset="Cora",
                method="gcn",
                score=0.71,
            )
            _write_result(
                tmp_dir / "pubmed" / "node2vec_seed42.json",
                dataset="PubMed",
                method="node2vec",
                score=0.66,
            )

            results = plot_results.load_results(tmp_dir)

        self.assertEqual(results["Cora"]["GCN"], 0.71)
        self.assertEqual(results["PubMed"]["Node2Vec"], 0.66)

    def test_load_results_prefers_default_filename_when_duplicates_exist(self) -> None:
        with workspace_temp_dir("plot-results-duplicates") as tmp_dir:
            _write_result(
                tmp_dir / "cora" / "gcn_seed42.json",
                dataset="Cora",
                method="gcn",
                score=0.71,
            )
            _write_result(
                tmp_dir / "cora" / "gcn_undirected_seed42.json",
                dataset="Cora",
                method="gcn",
                score=0.63,
            )

            results = plot_results.load_results(tmp_dir)

        self.assertEqual(results["Cora"]["GCN"], 0.71)

    def test_load_results_averages_scores_across_matching_seeds(self) -> None:
        with workspace_temp_dir("plot-results-seeds") as tmp_dir:
            _write_result(
                tmp_dir / "cora" / "gcn_seed42.json",
                dataset="Cora",
                method="gcn",
                score=0.70,
                seed=42,
            )
            _write_result(
                tmp_dir / "cora" / "gcn_seed7.json",
                dataset="Cora",
                method="gcn",
                score=0.80,
                seed=7,
            )

            results = plot_results.load_results(tmp_dir)

        self.assertEqual(results["Cora"]["GCN"], 0.75)


class PlotRenderingTests(unittest.TestCase):
    def test_plot_grouped_bar_chart_writes_output_file(self) -> None:
        with workspace_temp_dir("plot-render") as tmp_dir:
            output_path = tmp_dir / "comparison.png"
            plot_results.plot_grouped_bar_chart(
                {
                    "Cora": {"GCN": 0.71, "GraphSAGE": 0.69},
                    "PubMed": {"GCN": 0.75, "GraphSAGE": 0.73},
                },
                output_path,
                metric="roc_auc",
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
