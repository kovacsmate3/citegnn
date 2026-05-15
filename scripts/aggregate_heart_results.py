"""Aggregate HeaRT result JSONs into Markdown tables.

Reads every ``results/<dataset>/<method>_seed<S>.json`` produced by the
HeaRT baseline runners, groups them by (dataset, method), computes mean
+- std across seeds, and prints a comparison against the numbers
reported by Li et al. (NeurIPS 2023) Table 4. Output is written both to
stdout and to ``results/HEART_RESULTS.md`` (or the path passed via
``--output``).

The published numbers in this script are transcribed from the HeaRT
paper -- specifically Cora / PubMed columns of Table 4 (MRR + Hits@10,
percent units). Mismatches are expected because we evaluate a smaller
model family (heuristics + node2vec + feature-only + GCN/SAGE) at
default hyperparameters, while the paper tunes more aggressively.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

# Published HeaRT numbers (percent) for the methods we share with the paper.
# Source: Table 4 of Li et al. (NeurIPS 2023). MRR and Hits@10 columns.
PUBLISHED_HEART: dict[tuple[str, str], dict[str, float]] = {
    ("Cora", "common_neighbors"): {"mrr": 9.78, "hits_at_10": 20.11},
    ("Cora", "adamic_adar"): {"mrr": 11.91, "hits_at_10": 24.10},
    ("Cora", "personalized_pagerank"): {"mrr": 13.34, "hits_at_10": 28.65},
    ("Cora", "node2vec"): {"mrr": 14.47, "hits_at_10": 32.77},
    ("Cora", "gcn"): {"mrr": 16.61, "hits_at_10": 36.26},
    ("Cora", "graphsage"): {"mrr": 14.74, "hits_at_10": 34.65},
    ("PubMed", "common_neighbors"): {"mrr": 2.28, "hits_at_10": 4.78},
    ("PubMed", "adamic_adar"): {"mrr": 2.63, "hits_at_10": 5.71},
    ("PubMed", "personalized_pagerank"): {"mrr": 4.26, "hits_at_10": 9.81},
    ("PubMed", "node2vec"): {"mrr": 3.94, "hits_at_10": 8.51},
    ("PubMed", "gcn"): {"mrr": 7.13, "hits_at_10": 15.22},
    ("PubMed", "graphsage"): {"mrr": 9.39, "hits_at_10": 20.54},
}

METHOD_DISPLAY: dict[str, str] = {
    "common_neighbors": "Common Neighbors",
    "adamic_adar": "Adamic-Adar",
    "personalized_pagerank": "Personalized PageRank",
    "node2vec": "Node2Vec",
    "feature_only_mlp": "Feature-Only MLP",
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gat": "GAT",
    "gcn_pe": "GCN + structural PE",
    "graphsage_pe": "GraphSAGE + structural PE",
    "gat_pe": "GAT + structural PE",
}

METHOD_ORDER: list[str] = [
    "common_neighbors",
    "adamic_adar",
    "personalized_pagerank",
    "node2vec",
    "feature_only_mlp",
    "gcn",
    "graphsage",
    "gat",
    "gcn_pe",
    "graphsage_pe",
    "gat_pe",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/HEART_RESULTS.md")
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Cora", "PubMed"],
    )
    return parser.parse_args()


def _load_records(results_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for jsonfile in sorted(results_dir.glob("*/*.json")):
        # Skip the legacy folder.
        if "legacy" in jsonfile.parent.name.lower():
            continue
        if jsonfile.parent.name == "legacy_random_neg":
            continue
        payload = json.loads(jsonfile.read_text(encoding="utf-8"))
        if "mrr" not in payload.get("test", {}):
            continue  # ignore legacy random-neg JSONs
        dataset = str(payload.get("dataset", "?"))
        method = str(payload.get("method", "?"))
        grouped[(dataset, method)].append(payload)
    return grouped


def _summary_for(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean/std for the metrics we report (in percent units)."""

    keys = ["mrr", "hits_at_1", "hits_at_3", "hits_at_10", "hits_at_100", "roc_auc"]
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values: list[float] = []
        for record in records:
            value = record.get("test", {}).get(key)
            if value is not None:
                values.append(float(value) * 100.0)
        if values:
            summary[key] = {
                "mean": float(mean(values)),
                "std": float(pstdev(values)) if len(values) > 1 else 0.0,
                "n": float(len(values)),
            }
    return summary


def _format_value(stat: dict[str, float] | None) -> str:
    if stat is None:
        return "--"
    mean_v = stat["mean"]
    std_v = stat["std"]
    n = int(stat["n"])
    if n > 1:
        return f"{mean_v:5.2f} ± {std_v:4.2f}"
    return f"{mean_v:5.2f}"


def _format_published(dataset: str, method: str, metric: str) -> str:
    published = PUBLISHED_HEART.get((dataset, method))
    if published is None or metric not in published:
        return "--"
    return f"{published[metric]:5.2f}"


def _format_delta(stat: dict[str, float] | None, dataset: str, method: str, metric: str) -> str:
    published = PUBLISHED_HEART.get((dataset, method))
    if stat is None or published is None or metric not in published:
        return "--"
    delta = stat["mean"] - published[metric]
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):4.2f}"


def _build_dataset_table(
    dataset: str,
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
) -> str:
    methods = [
        method
        for method in METHOD_ORDER
        if (dataset, method) in grouped
        or (dataset, method) in PUBLISHED_HEART
    ]
    if not methods:
        return f"### {dataset}\n\n_No HeaRT results recorded yet._\n"

    lines: list[str] = []
    lines.append(f"### {dataset}\n")
    lines.append("All metrics are reported in percent (mean ± std across seeds).")
    lines.append("The 'Paper' column is the value reported by Li et al. (NeurIPS 2023) Table 4.")
    lines.append("'Δ MRR' is ours minus paper.\n")
    lines.append("| Method | MRR | Hits@1 | Hits@3 | Hits@10 | Hits@100 | Paper MRR | Paper Hits@10 | Δ MRR |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for method in methods:
        records = grouped.get((dataset, method), [])
        summary = _summary_for(records) if records else {}
        row = [
            METHOD_DISPLAY.get(method, method),
            _format_value(summary.get("mrr")),
            _format_value(summary.get("hits_at_1")),
            _format_value(summary.get("hits_at_3")),
            _format_value(summary.get("hits_at_10")),
            _format_value(summary.get("hits_at_100")),
            _format_published(dataset, method, "mrr"),
            _format_published(dataset, method, "hits_at_10"),
            _format_delta(summary.get("mrr"), dataset, method, "mrr"),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    grouped = _load_records(args.results_dir)

    sections: list[str] = ["# HeaRT-Aligned Link Prediction Results", ""]
    sections.append(
        "Each row aggregates the JSON outputs in `results/<dataset>/`. "
        "Values are means (± sample standard deviation) across the "
        "available seeds; rerun with additional `--seed` values to "
        "tighten the std estimate. The reference 'Paper MRR' / "
        "'Paper Hits@10' come from **Table 4 of Li et al. (NeurIPS 2023)** "
        "\"Evaluating Graph Neural Networks for Link Prediction\"."
    )
    sections.append("")
    for dataset in args.datasets:
        sections.append(_build_dataset_table(dataset, grouped))
    text = "\n".join(sections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote summary to {args.output}")


if __name__ == "__main__":
    main()
