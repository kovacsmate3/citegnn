"""Run GCN and GraphSAGE link-prediction baselines from a saved split artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from citegnn.baselines.gnn import GNNBaselineConfig, GNNMethod, run_gnn_baseline
from citegnn.data.splits import load_split_artifact
from citegnn.evaluation import save_and_describe_experiment_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["all", *(method.value for method in GNNMethod)],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--decoder-hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def _selected_methods(method_name: str) -> list[GNNMethod]:
    if method_name == "all":
        return list(GNNMethod)
    return [GNNMethod(method_name)]


def main() -> None:
    args = parse_args()
    split = load_split_artifact(args.split_path)
    config = GNNBaselineConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        decoder_hidden_dim=args.decoder_hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
    )

    for method in _selected_methods(args.method):
        result = run_gnn_baseline(
            split,
            method=method,
            config=config,
            data_root=args.data_root,
        )
        _, summary = save_and_describe_experiment_result(
            result,
            output_dir=args.output_dir,
        )
        print(summary)


if __name__ == "__main__":
    main()
