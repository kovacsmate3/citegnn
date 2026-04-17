"""Plot saved experiment results from the repo's JSON artifact layout."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np


MetricName = Literal["roc_auc", "average_precision"]
ResultTable = dict[str, dict[str, float]]
ResultRecord = dict[str, Any]

_METHOD_LABELS = {
    "adamic_adar": "Adamic-Adar",
    "common_neighbors": "Common Neighbors",
    "feature_only_mlp": "Feature-Only MLP",
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "node2vec": "Node2Vec",
    "personalized_pagerank": "Personalized PageRank",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the result plotting utility."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing dataset subfolders with JSON result files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/model_comparison.png"),
        help="Path where the generated plot is written.",
    )
    parser.add_argument(
        "--metric",
        choices=("roc_auc", "average_precision"),
        default="roc_auc",
        help="Test metric to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed filter. When omitted, scores are averaged across available seeds.",
    )
    return parser.parse_args()


def _result_method_label(method_name: str) -> str:
    """Convert one stored method identifier into a plot-friendly label."""

    return _METHOD_LABELS.get(method_name, method_name.replace("_", " ").title())


def _path_preference(path: Path) -> tuple[int, str]:
    """Rank duplicate artifact paths so the current naming scheme wins deterministically."""

    name = path.name
    return (1 if "_undirected_" in name else 0, name)


def load_result_records(
    results_dir: Path,
    *,
    seed: int | None = None,
) -> list[ResultRecord]:
    """Load matching result JSON files into notebook-friendly records."""

    if not results_dir.exists():
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    selected_results: dict[tuple[str, str, int], tuple[tuple[int, str], ResultRecord]] = {}

    for json_file in sorted(results_dir.rglob("*.json")):
        payload: dict[str, Any] = json.loads(json_file.read_text(encoding="utf-8"))
        params = payload.get("params", {})
        dataset_name = str(payload.get("dataset", "Unknown"))
        method_name = _result_method_label(str(payload.get("method", "Unknown")))
        result_seed = int(payload.get("seed", 0))
        if seed is not None and result_seed != seed:
            continue

        test_metrics = payload.get("test", {})
        validation_metrics = payload.get("validation", {})
        record: ResultRecord = {
            "dataset": dataset_name,
            "method": method_name,
            "seed": result_seed,
            "test_roc_auc": float(test_metrics.get("roc_auc", float("nan"))),
            "test_average_precision": float(
                test_metrics.get("average_precision", float("nan"))
            ),
            "val_roc_auc": float(validation_metrics.get("roc_auc", float("nan"))),
            "val_average_precision": float(
                validation_metrics.get("average_precision", float("nan"))
            ),
            "params": params,
            "relative_path": str(json_file.relative_to(results_dir)),
        }

        key = (dataset_name, method_name, result_seed)
        candidate = (_path_preference(json_file), record)
        current = selected_results.get(key)
        if current is None or candidate[0] < current[0]:
            selected_results[key] = candidate

    records = [selected_results[key][1] for key in sorted(selected_results)]
    if not records:
        raise ValueError(
            "No result files matched the requested filters. "
            "Check the results directory, graph variant, and seed."
        )
    return records


def load_results(
    results_dir: Path,
    *,
    metric: MetricName = "roc_auc",
    seed: int | None = None,
) -> ResultTable:
    """Load matching result JSON files and aggregate scores by dataset and method."""

    aggregated_scores: defaultdict[str, defaultdict[str, list[float]]]
    aggregated_scores = defaultdict(lambda: defaultdict(list))
    metric_key = f"test_{metric}"
    for record in load_result_records(
        results_dir,
        seed=seed,
    ):
        aggregated_scores[record["dataset"]][record["method"]].append(float(record[metric_key]))

    result_table: ResultTable = {}
    for dataset_name in sorted(aggregated_scores):
        result_table[dataset_name] = {}
        for method_name in sorted(aggregated_scores[dataset_name]):
            scores = aggregated_scores[dataset_name][method_name]
            result_table[dataset_name][method_name] = float(np.mean(scores))
    return result_table


def plot_grouped_bar_chart(
    data: ResultTable,
    output_path: Path,
    *,
    metric: MetricName,
) -> None:
    """Render one grouped comparison chart and save it to disk."""

    datasets = list(data.keys())
    methods = sorted({method for method_scores in data.values() for method in method_scores})
    x_positions = np.arange(len(datasets), dtype=float)
    width = 0.8 / max(len(methods), 1)
    offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    color_map = plt.get_cmap("tab10")

    for index, method_name in enumerate(methods):
        scores = [data[dataset_name].get(method_name, 0.0) for dataset_name in datasets]
        rects = ax.bar(
            x_positions + offsets[index],
            scores,
            width * 0.95,
            label=method_name,
            color=color_map(index % 10),
        )
        ax.bar_label(rects, padding=3, fmt="%.3f", fontsize=9)

    metric_label = "ROC-AUC" if metric == "roc_auc" else "Average Precision"
    ax.set_ylabel(f"Test {metric_label}", fontsize=12)
    ax.set_title("Link Prediction Performance Comparison", fontsize=14, pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(loc="upper right", title="Models", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot successfully saved to: {output_path}")


def main() -> None:
    """Load saved results and generate one comparison chart."""

    args = parse_args()
    print(f"Scanning for JSON results in: {args.results_dir}")
    aggregated_data = load_results(
        args.results_dir,
        metric=args.metric,
        seed=args.seed,
    )
    plot_grouped_bar_chart(
        aggregated_data,
        args.output_path,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
