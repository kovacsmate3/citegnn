"""The full-sweep script must train with the configurations behind the reported numbers."""

from __future__ import annotations

import dataclasses
import json
import unittest

from citegnn.baselines.feature_only import FeatureOnlyBaselineConfig
from citegnn.baselines.gnn import GNNBaselineConfig
from citegnn.baselines.node2vec import Node2VecConfig
from tests.support import REPO_ROOT, load_script_module

run_all_heart = load_script_module("run_all_heart", "scripts/run_all_heart.py")


def _build_config(method_name: str, params: dict[str, object]) -> dict[str, object]:
    if method_name == "node2vec":
        config: object = Node2VecConfig(**run_all_heart._config_fields(Node2VecConfig, params))
    elif method_name == "feature_only_mlp":
        config = FeatureOnlyBaselineConfig(
            **run_all_heart._config_fields(FeatureOnlyBaselineConfig, params), show_progress=False
        )
    else:
        config = GNNBaselineConfig(
            **run_all_heart._config_fields(GNNBaselineConfig, params),
            use_structural_features=method_name.endswith("_pe"),
        )
    return dataclasses.asdict(config)  # type: ignore[call-overload]


class RecordedConfigTests(unittest.TestCase):
    def test_configs_match_the_saved_result_params(self) -> None:
        checked = 0
        for dataset, configs in run_all_heart.RECORDED_CONFIGS.items():
            for method_name in configs:
                path = REPO_ROOT / "results" / dataset.lower() / f"{method_name}_seed1.json"
                if not path.exists():
                    continue  # e.g. Node2Vec was never run on PubMed
                with self.subTest(dataset=dataset, method=method_name):
                    recorded = json.loads(path.read_text(encoding="utf-8"))["params"]
                    built = _build_config(method_name, run_all_heart._recorded(dataset, method_name))
                    differences = {
                        key: (value, recorded[key])
                        for key, value in built.items()
                        if key in recorded and value != recorded[key]
                    }
                    self.assertEqual(differences, {})
                    checked += 1
        self.assertEqual(checked, 15)

    def test_overrides_replace_only_given_fields(self) -> None:
        params = run_all_heart._recorded("Cora", "gat", epochs=7, patience=None)
        self.assertEqual(params["epochs"], 7)
        self.assertEqual(params["patience"], 10)
        self.assertEqual(run_all_heart.RECORDED_CONFIGS["Cora"]["gat"]["epochs"], 400)

    def test_unknown_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_all_heart._config_fields(Node2VecConfig, {"hidden_dim": 8})


if __name__ == "__main__":
    unittest.main()
