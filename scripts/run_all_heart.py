"""End-to-end HeaRT experiment runner.

Builds (or reuses) a HeaRT split for one dataset and then runs every
baseline against it: the three heuristics (CN, AA, PPR), node2vec, the
feature-only pairwise MLP, GCN, GraphSAGE and GAT, plus the three GNNs
augmented with structural features (RQ3 of the proposal).

The trainable baselines use the configurations behind the reported
numbers, i.e. the ``params`` recorded in ``results/<dataset>/*_seed1.json``
(see ``RECORDED_CONFIGS`` below). The ``--gnn-*``, ``--feature-only-*`` and
``--node2vec-epochs`` options override single fields of those
configurations for every model of that family.

Usage::

    python scripts/run_all_heart.py --dataset Cora --seed 1
    python scripts/run_all_heart.py --dataset PubMed --seed 1

Each baseline writes one JSON file to ``results/<dataset>/<method>_seed<S>.json``.
A summary line per baseline is printed to stdout.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any

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
    HeartSplit,
    create_heart_split,
    heart_split_artifact_name,
    load_heart_split,
    save_heart_split,
)
from citegnn.data.loaders import CitationDataset, load_citation_graph
from citegnn.evaluation import ExperimentResult, save_and_describe_experiment_result

# Training recipe shared by every reported GNN run: Hadamard-MLP decoder,
# HeaRT-style mini-batches with DropEdge, one random negative per positive.
_GNN_RECIPE: dict[str, Any] = {
    "decoder": "hadamard_mlp",
    "train_mode": "minibatch_dropedge",
    "train_negatives_per_positive": 1,
    "weight_decay": 0.0,
    "gat_heads": 4,
    "gat_attention_dropout": 0.1,
    "num_laplacian_eigvecs": 10,
}
_CORA_GNN: dict[str, Any] = {
    **_GNN_RECIPE,
    "hidden_dim": 128,
    "decoder_hidden_dim": 128,
    "train_batch_size": 1024,
    "eval_every": 5,
    "patience": 10,
}
_PUBMED_GNN: dict[str, Any] = {
    **_GNN_RECIPE,
    "hidden_dim": 256,
    "decoder_hidden_dim": 256,
    "train_batch_size": 4096,
    "eval_every": 10,
    "patience": 4,
    "epochs": 80,
}
_GCN = {"num_layers": 3, "learning_rate": 0.01, "dropout": 0.3}
_SAGE = {"num_layers": 2, "learning_rate": 0.005, "dropout": 0.1}
_GAT = {"num_layers": 1, "learning_rate": 0.01, "dropout": 0.1}

# Configurations behind the reported numbers, copied from the ``params``
# field of results/<dataset>/<method>_seed1.json. Keys are the saved
# method names.
RECORDED_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "Cora": {
        "node2vec": {"embedding_dim": 128, "epochs": 5},
        "feature_only_mlp": {
            "feature_mode": "hadamard",
            "hidden_dim": 128,
            "dropout": 0.3,
            "learning_rate": 0.005,
            "weight_decay": 1e-5,
            "epochs": 80,
            "patience": 10,
            "eval_every": 10,
            "batch_size": 2048,
        },
        "gcn": {**_CORA_GNN, **_GCN, "epochs": 400},
        "graphsage": {**_CORA_GNN, **_SAGE, "epochs": 400},
        "gat": {**_CORA_GNN, **_GAT, "epochs": 400},
        "gcn_pe": {**_CORA_GNN, **_GCN, "epochs": 300},
        "graphsage_pe": {**_CORA_GNN, **_SAGE, "epochs": 300},
        "gat_pe": {**_CORA_GNN, **_GAT, "epochs": 300},
    },
    "PubMed": {
        # Node2Vec has no reported PubMed result; the Cora settings are used.
        "node2vec": {"embedding_dim": 128, "epochs": 5},
        "feature_only_mlp": {
            "feature_mode": "hadamard",
            "hidden_dim": 128,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "epochs": 20,
            "patience": 10,
            "eval_every": 10,
            "batch_size": 1024,
        },
        "gcn": {
            **_PUBMED_GNN,
            **_GCN,
            "num_layers": 1,
            "epochs": 600,
            "eval_every": 5,
            "patience": 10,
            "train_batch_size": 1024,
        },
        "graphsage": {**_PUBMED_GNN, **_SAGE},
        "gat": {**_PUBMED_GNN, **_GAT},
        "gcn_pe": {**_PUBMED_GNN, **_GCN, "num_layers": 1, "epochs": 300},
        "graphsage_pe": {**_PUBMED_GNN, **_SAGE},
        "gat_pe": {**_PUBMED_GNN, **_GAT},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    override = "Overrides the recorded configuration for every model of this family."
    parser.add_argument("--gnn-epochs", type=int, default=None, help=override)
    parser.add_argument("--gnn-eval-every", type=int, default=None, help=override)
    parser.add_argument("--gnn-patience", type=int, default=None, help=override)
    parser.add_argument("--feature-only-epochs", type=int, default=None, help=override)
    parser.add_argument("--feature-only-eval-every", type=int, default=None, help=override)
    parser.add_argument("--feature-only-patience", type=int, default=None, help=override)
    parser.add_argument("--node2vec-epochs", type=int, default=None, help=override)
    return parser.parse_args()


def _ensure_split(args: argparse.Namespace) -> HeartSplit:
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


def _recorded(dataset: str, method_name: str, **overrides: int | None) -> dict[str, Any]:
    """Recorded configuration for one method, with non-None overrides applied."""

    params = dict(RECORDED_CONFIGS[dataset][method_name])
    params.update({key: value for key, value in overrides.items() if value is not None})
    return params


def _config_fields(config_type: type, params: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in dataclasses.fields(config_type)}
    unknown = set(params) - names
    if unknown:
        raise ValueError(f"Unknown {config_type.__name__} fields: {sorted(unknown)}")
    return params


def _stage(label: str) -> float:
    print(f"\n=== {label} ===", flush=True)
    return time.time()


def _report(result: ExperimentResult, results_dir: Path, label: str, start: float) -> None:
    _, summary = save_and_describe_experiment_result(result, output_dir=results_dir)
    print(summary)
    print(f"[run_all_heart] {label} done in {time.time() - start:.1f}s", flush=True)


def main() -> None:
    args = parse_args()
    skip = set(args.skip)
    overall_start = time.time()
    split = _ensure_split(args)

    if "heuristics" not in skip:
        for heuristic in HeuristicMethod:
            t0 = _stage(f"Heuristic: {heuristic.value}")
            result = run_heuristic_baseline(split, method=heuristic, config=HeuristicConfig())
            _report(result, args.results_dir, heuristic.value, t0)

    if "node2vec" not in skip:
        t0 = _stage("Node2Vec baseline")
        node2vec_params = _recorded(args.dataset, "node2vec", epochs=args.node2vec_epochs)
        node2vec_config = Node2VecConfig(**_config_fields(Node2VecConfig, node2vec_params))
        result = run_node2vec_baseline(split, config=node2vec_config)
        _report(result, args.results_dir, "node2vec", t0)

    if "feature_only" not in skip:
        t0 = _stage("Feature-only MLP baseline")
        mlp_params = _recorded(
            args.dataset,
            "feature_only_mlp",
            epochs=args.feature_only_epochs,
            patience=args.feature_only_patience,
            eval_every=args.feature_only_eval_every,
        )
        mlp_config = FeatureOnlyBaselineConfig(
            **_config_fields(FeatureOnlyBaselineConfig, mlp_params), show_progress=False
        )
        result = run_feature_only_baseline(split, config=mlp_config)
        _report(result, args.results_dir, "feature_only_mlp", t0)

    gnn_variants = [
        (stage, use_pe)
        for stage, use_pe in (("gnn", False), ("gnn_pe", True))
        if stage not in skip
    ]
    for _, use_pe in gnn_variants:
        for gnn in GNNMethod:
            method_name = f"{gnn.value}_pe" if use_pe else gnn.value
            t0 = _stage(f"GNN{'+PE' if use_pe else ''}: {gnn.value}")
            gnn_params = _recorded(
                args.dataset,
                method_name,
                epochs=args.gnn_epochs,
                patience=args.gnn_patience,
                eval_every=args.gnn_eval_every,
            )
            gnn_config = GNNBaselineConfig(
                **_config_fields(GNNBaselineConfig, gnn_params),
                use_structural_features=use_pe,
            )
            result = run_gnn_baseline(
                split,
                method=gnn,
                config=gnn_config,
                method_name_override=method_name if use_pe else None,
            )
            _report(result, args.results_dir, method_name, t0)

    print(f"\n[run_all_heart] TOTAL elapsed: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
