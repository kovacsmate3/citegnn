"""Run GCN and GraphSAGE link-prediction baselines from a saved HeaRT split.

Two configurations are supported per encoder:

* Plain (``--method gcn|graphsage``): the encoder receives only the raw
  node attributes.
* With structural augmentation (``--use-structural-features``): the
  encoder also receives a log-degree feature plus the top-k Laplacian
  eigenvectors of the training graph (RQ3 of the proposal).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from citegnn.baselines.gnn import GNNBaselineConfig, GNNMethod, run_gnn_baseline
from citegnn.data.heart_split import load_heart_split
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
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--decoder-hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--use-structural-features",
        action="store_true",
        help="Concatenate log-degree + Laplacian PE to node features (RQ3 ablation).",
    )
    parser.add_argument(
        "--num-laplacian-eigvecs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--gat-heads",
        type=int,
        default=4,
        help="Number of attention heads per layer for the GAT encoder.",
    )
    parser.add_argument(
        "--gat-attention-dropout",
        type=float,
        default=0.1,
        help="Dropout rate applied to GAT attention coefficients.",
    )
    parser.add_argument(
        "--train-mode",
        choices=("fullbatch", "minibatch_dropedge"),
        default="fullbatch",
        help=(
            "fullbatch: one optimizer step per epoch on the whole train "
            "set. minibatch_dropedge: HeaRT-style mini-batches with the "
            "supervision edges of each batch removed from the message-"
            "passing graph for that step."
        ),
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1024,
        help="Mini-batch size of supervision edges per optimizer step (only used in minibatch_dropedge mode).",
    )
    parser.add_argument(
        "--decoder",
        choices=("hadamard_mlp", "dot"),
        default="hadamard_mlp",
    )
    parser.add_argument(
        "--train-negatives-per-positive",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--method-suffix",
        default=None,
        help="Optional suffix appended to the saved method name (e.g. '_pe').",
    )
    return parser.parse_args()


def _selected_methods(method_name: str) -> list[GNNMethod]:
    if method_name == "all":
        return list(GNNMethod)
    return [GNNMethod(method_name)]


def main() -> None:
    args = parse_args()
    split = load_heart_split(args.split_path)
    config = GNNBaselineConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        decoder_hidden_dim=args.decoder_hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        use_structural_features=args.use_structural_features,
        num_laplacian_eigvecs=args.num_laplacian_eigvecs,
        gat_heads=args.gat_heads,
        gat_attention_dropout=args.gat_attention_dropout,
        train_mode=args.train_mode,
        train_batch_size=args.train_batch_size,
        decoder=args.decoder,
        train_negatives_per_positive=args.train_negatives_per_positive,
    )

    for method in _selected_methods(args.method):
        method_override: str | None = None
        if args.method_suffix:
            method_override = f"{method.value}{args.method_suffix}"
        elif args.use_structural_features:
            method_override = f"{method.value}_pe"
        result = run_gnn_baseline(
            split,
            method=method,
            config=config,
            data_root=args.data_root,
            method_name_override=method_override,
        )
        _, summary = save_and_describe_experiment_result(
            result, output_dir=args.output_dir
        )
        print(summary)


if __name__ == "__main__":
    main()
