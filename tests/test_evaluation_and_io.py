from __future__ import annotations

import json
from pathlib import Path
import unittest

from citegnn.evaluation import (
    build_experiment_result,
    evaluate_with_validation_threshold,
    save_experiment_result,
)
from citegnn.utils.io import write_json
from tests.support import make_metrics, workspace_temp_dir


class EvaluationTests(unittest.TestCase):
    def test_evaluate_with_validation_threshold_reuses_the_selected_threshold(self) -> None:
        validation_metrics, test_metrics = evaluate_with_validation_threshold(
            y_val_true=[0, 1, 1, 0],
            y_val_score=[0.1, 0.4, 0.9, 0.8],
            y_test_true=[0, 1],
            y_test_score=[0.2, 0.7],
        )
        self.assertAlmostEqual(validation_metrics.threshold, 0.4)
        self.assertAlmostEqual(test_metrics.threshold, 0.4)

    def test_build_and_save_experiment_result_round_trip(self) -> None:
        result = build_experiment_result(
            dataset="Cora",
            method="common_neighbors",
            seed=7,
            params={"alpha": 0.85},
            validation_metrics=make_metrics(threshold=0.4),
            test_metrics=make_metrics(threshold=0.4, roc_auc=0.88),
        )

        with workspace_temp_dir("evaluation") as tmp_dir:
            output_path = save_experiment_result(result, output_dir=tmp_dir)
            self.assertEqual(
                output_path,
                tmp_dir / "cora" / "common_neighbors_seed7.json",
            )
            payload = json.loads(output_path.read_text())

        self.assertEqual(payload["dataset"], "Cora")
        self.assertEqual(payload["method"], "common_neighbors")
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["params"]["alpha"], 0.85)
        self.assertEqual(payload["validation"]["threshold"], 0.4)
        self.assertEqual(payload["test"]["roc_auc"], 0.88)


class JsonIoTests(unittest.TestCase):
    def test_write_json_creates_parents_and_appends_newline(self) -> None:
        with workspace_temp_dir("io") as tmp_dir:
            output_path = write_json(
                {"b": 2, "a": 1},
                tmp_dir / "nested" / "artifact.json",
            )
            content = output_path.read_text()

        self.assertTrue(content.endswith("\n"))
        self.assertEqual(json.loads(content), {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
