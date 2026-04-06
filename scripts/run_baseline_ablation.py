"""Run the current baseline comparison sweep on the shared undirected protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from citegnn.baselines.gnn import GNNBaselineConfig, GNNMethod, run_gnn_baseline
from citegnn.baselines.heuristics import HeuristicMethod, run_heuristic_baseline
from citegnn.baselines.node2vec import Node2VecConfig, run_node2vec_baseline
from citegnn.data.loaders import CitationDataset, load_citation_graph
from citegnn.data.splits import create_link_prediction_split, save_split_artifact
from citegnn.evaluation import save_experiment_result
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
):
    graph = load_citation_graph(
        dataset_name,
        root=data_root,
        undirected=True,
    )
    split = create_link_prediction_split(graph, seed=seed)
    save_split_artifact(split, output_dir=split_output_dir)
    return split


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
        print(f"\n== {dataset_name} / {split.graph_variant} ==")

        for method in HeuristicMethod:
            result = run_heuristic_baseline(split, method=method)
            output_path = save_experiment_result(result, output_dir=args.result_output_dir)
            print(
                f"{result.method:<22} "
                f"test_auc={result.test['roc_auc']:.4f} "
                f"test_ap={result.test['average_precision']:.4f} "
                f"-> {output_path}"
            )

        node2vec_result = run_node2vec_baseline(split, config=node2vec_config)
        node2vec_output = save_experiment_result(
            node2vec_result,
            output_dir=args.result_output_dir,
        )
        print(
            f"{node2vec_result.method:<22} "
            f"test_auc={node2vec_result.test['roc_auc']:.4f} "
            f"test_ap={node2vec_result.test['average_precision']:.4f} "
            f"-> {node2vec_output}"
        )

        for method in GNNMethod:
            result = run_gnn_baseline(
                split,
                method=method,
                config=gnn_config,
                data_root=str(args.data_root),
            )
            output_path = save_experiment_result(result, output_dir=args.result_output_dir)
            print(
                f"{result.method:<22} "
                f"test_auc={result.test['roc_auc']:.4f} "
                f"test_ap={result.test['average_precision']:.4f} "
                f"-> {output_path}"
            )


if __name__ == "__main__":
    main()
