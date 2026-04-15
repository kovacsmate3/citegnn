"""Run the current baseline comparison sweep on the shared undirected protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from citegnn.baselines.gnn import GNNBaselineConfig, GNNMethod, run_gnn_baseline
from citegnn.baselines.heuristics import HeuristicMethod, run_heuristic_baseline
from citegnn.baselines.node2vec import Node2VecConfig, run_node2vec_baseline
from citegnn.data.loaders import CitationDataset, load_citation_graph
from citegnn.data.splits import (
    LinkPredictionSplit,
    create_link_prediction_split,
    save_split_artifact,
)
from citegnn.evaluation import save_and_describe_experiment_result
from citegnn.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the baseline ablation runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[dataset.value for dataset in CitationDataset],
        default=[dataset.value for dataset in CitationDataset],
        help="Datasets to include in the sweep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed shared by split creation and trainable baselines.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for Planetoid datasets.",
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=Path("artifacts/splits"),
        help="Directory where split artifacts are written.",
    )
    parser.add_argument(
        "--result-output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where result JSON files are written.",
    )
    parser.add_argument(
        "--gnn-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension shared by GCN and GraphSAGE.",
    )
    parser.add_argument(
        "--gnn-decoder-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension of the GNN decoder MLP.",
    )
    parser.add_argument(
        "--gnn-epochs",
        type=int,
        default=50,
        help="Maximum number of epochs for the trainable GNN baselines.",
    )
    parser.add_argument(
        "--gnn-patience",
        type=int,
        default=10,
        help="Early-stopping patience for the trainable GNN baselines.",
    )
    parser.add_argument(
        "--node2vec-epochs",
        type=int,
        default=3,
        help="Number of skip-gram training epochs for node2vec.",
    )
    return parser.parse_args()


def _build_split(
    *,
    dataset_name: str,
    seed: int,
    data_root: Path,
    split_output_dir: Path,
) -> LinkPredictionSplit:
    graph = load_citation_graph(
        dataset_name,
        root=data_root,
    )
    split = create_link_prediction_split(graph, seed=seed)
    save_split_artifact(split, output_dir=split_output_dir)
    return split


def _print_saved_result(
    result_output_dir: Path,
    *,
    result,
) -> None:
    """Persist one result and print the shared ablation summary format."""

    _, summary = save_and_describe_experiment_result(
        result,
        output_dir=result_output_dir,
        include_validation_auc=False,
        include_test_average_precision=True,
    )
    print(summary)


def main() -> None:
    """Execute the baseline comparison sweep on each requested dataset."""

    args = parse_args()
    seed_everything(args.seed)

    gnn_config = GNNBaselineConfig(
        hidden_dim=args.gnn_hidden_dim,
        decoder_hidden_dim=args.gnn_decoder_hidden_dim,
        epochs=args.gnn_epochs,
        patience=args.gnn_patience,
    )
    node2vec_config = Node2VecConfig(epochs=args.node2vec_epochs)

    for dataset_name in args.datasets:
        split = _build_split(
            dataset_name=dataset_name,
            seed=args.seed,
            data_root=args.data_root,
            split_output_dir=args.split_output_dir,
        )
        print(f"\n== {dataset_name} ==")

        for heuristic_method in HeuristicMethod:
            result = run_heuristic_baseline(split, method=heuristic_method)
            _print_saved_result(args.result_output_dir, result=result)

        node2vec_result = run_node2vec_baseline(split, config=node2vec_config)
        _print_saved_result(args.result_output_dir, result=node2vec_result)

        for gnn_method in GNNMethod:
            result = run_gnn_baseline(
                split,
                method=gnn_method,
                config=gnn_config,
                data_root=str(args.data_root),
            )
            _print_saved_result(args.result_output_dir, result=result)


if __name__ == "__main__":
    main()
