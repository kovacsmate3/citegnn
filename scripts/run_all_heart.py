"""End-to-end HeaRT experiment runner.

Builds (or reuses) a HeaRT split for one dataset and then runs every
baseline against it: the three heuristics (CN, AA, PPR), the
feature-only pairwise MLP, node2vec, GCN, GraphSAGE, plus the
GCN/GraphSAGE variants augmented with structural features (RQ3 of the
proposal).

Usage::

    python scripts/run_all_heart.py --dataset Cora --seed 1
    python scripts/run_all_heart.py --dataset PubMed --seed 1 \
        --gnn-eval-every 5 --feature-only-eval-every 5

Each baseline writes one JSON file to ``results/<dataset>/<method>_seed<S>.json``.
A summary line per baseline is printed to stdout.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from citegnn.baselines.feature_only import (
    FeatureOnlyBaselineConfig,
    run_feature_only_baseline,
)
from citegnn.baselines.gnn import GNNBaselineConfig, GNNMethod, run_gnn_baseline
from citegnn.baselines.heuristics import (
    HeuristicConfig,
    HeuristicMethod,
    run_heuristic_baseline,
)
from citegnn.baselines.node2vec import Node2VecConfig, run_node2vec_baseline
from citegnn.data.heart_split import (
    DEFAULT_K,
    create_heart_split,
    heart_split_artifact_name,
    load_heart_split,
    save_heart_split,
)
from citegnn.data.loaders import CitationDataset, load_citation_graph
from citegnn.evaluation import save_and_describe_experiment_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[d.value for d in CitationDataset],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--split-dir", type=Path, default=Path("artifacts/heart"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--rebuild-split",
        action="store_true",
        help="Rebuild the HeaRT split even if a cached artifact exists.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Baselines to skip. Choices: heuristics, node2vec, feature_only, gnn, gnn_pe.",
    )
    parser.add_argument("--gnn-epochs", type=int, default=200)
    parser.add_argument("--gnn-eval-every", type=int, default=1)
    parser.add_argument("--gnn-patience", type=int, default=20)
    parser.add_argument("--feature-only-epochs", type=int, default=100)
    parser.add_argument("--feature-only-eval-every", type=int, default=1)
    parser.add_argument("--feature-only-patience", type=int, default=10)
    parser.add_argument("--node2vec-epochs", type=int, default=3)
    return parser.parse_args()


def _ensure_split(args: argparse.Namespace):
    split_path = args.split_dir / heart_split_artifact_name(
        args.dataset, seed=args.seed, K=args.K
    )
    if split_path.exists() and not args.rebuild_split:
        print(f"[run_all_heart] Loading cached split: {split_path}")
        return load_heart_split(split_path)
    print(f"[run_all_heart] Building HeaRT split for {args.dataset} seed={args.seed} K={args.K}")
    graph = load_citation_graph(args.dataset)
    split = create_heart_split(graph, seed=args.seed, K=args.K)
    save_heart_split(split, output_dir=args.split_dir)
    return split


def _stage(label: str):
    print(f"\n=== {label} ===")
    return time.time()


def _done(label: str, start: float) -> None:
    elapsed = time.time() - start
    print(f"[run_all_heart] {label} done in {elapsed:.1f}s")


def main() -> None:
    args = parse_args()
    skip = set(args.skip)
    overall_start = time.time()
    split = _ensure_split(args)

    if "heuristics" not in skip:
        for method in HeuristicMethod:
            t0 = _stage(f"Heuristic: {method.value}")
            result = run_heuristic_baseline(split, method=method, config=HeuristicConfig())
            _, summary = save_and_describe_experiment_result(
                result, output_dir=args.results_dir
            )
            print(summary)
            _done(method.value, t0)

    if "node2vec" not in skip:
        t0 = _stage("Node2Vec baseline")
        result = run_node2vec_baseline(
            split,
            config=Node2VecConfig(epochs=args.node2vec_epochs),
        )
        _, summary = save_and_describe_experiment_result(
            result, output_dir=args.results_dir
        )
        print(summary)
        _done("node2vec", t0)

    if "feature_only" not in skip:
        t0 = _stage("Feature-only MLP baseline")
        config = FeatureOnlyBaselineConfig(
            epochs=args.feature_only_epochs,
            patience=args.feature_only_patience,
            eval_every=args.feature_only_eval_every,
            show_progress=False,
        )
        result = run_feature_only_baseline(split, config=config)
        _, summary = save_and_describe_experiment_result(
            result, output_dir=args.results_dir
        )
        print(summary)
        _done("feature_only_mlp", t0)

    if "gnn" not in skip:
        for method in GNNMethod:
            t0 = _stage(f"GNN: {method.value}")
            config = GNNBaselineConfig(
                epochs=args.gnn_epochs,
                patience=args.gnn_patience,
                eval_every=args.gnn_eval_every,
                decoder="dot",
                train_negatives_per_positive=5,
                learning_rate=0.01,
                dropout=0.3,
            )
            result = run_gnn_baseline(split, method=method, config=config)
            _, summary = save_and_describe_experiment_result(
                result, output_dir=args.results_dir
            )
            print(summary)
            _done(method.value, t0)

    if "gnn_pe" not in skip:
        for method in GNNMethod:
            t0 = _stage(f"GNN+PE: {method.value}")
            config = GNNBaselineConfig(
                epochs=args.gnn_epochs,
                patience=args.gnn_patience,
                eval_every=args.gnn_eval_every,
                decoder="dot",
                train_negatives_per_positive=5,
                learning_rate=0.01,
                dropout=0.3,
                use_structural_features=True,
            )
            result = run_gnn_baseline(
                split,
                method=method,
                config=config,
                method_name_override=f"{method.value}_pe",
            )
            _, summary = save_and_describe_experiment_result(
                result, output_dir=args.results_dir
            )
            print(summary)
            _done(f"{method.value}_pe", t0)

    print(f"\n[run_all_heart] TOTAL elapsed: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
