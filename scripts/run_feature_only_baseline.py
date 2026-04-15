"""Run the feature-only pairwise MLP baseline from a saved split artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from citegnn.baselines.feature_only import (
    FeatureOnlyBaselineConfig,
    run_feature_only_baseline,
)
from citegnn.data.splits import load_split_artifact
from citegnn.evaluation import save_and_describe_experiment_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--feature-mode",
        choices=("concat", "hadamard", "abs_diff", "hadamard_abs_diff"),
        default="hadamard_abs_diff",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the training progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = load_split_artifact(args.split_path)
    config = FeatureOnlyBaselineConfig(
        feature_mode=args.feature_mode,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )
    result = run_feature_only_baseline(split, config=config, data_root=args.data_root)
    _, summary = save_and_describe_experiment_result(result, output_dir=args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
