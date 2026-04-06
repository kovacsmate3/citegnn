"""Run the node2vec link-prediction baseline from a saved split artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from citegnn.baselines.node2vec import Node2VecConfig, run_node2vec_baseline
from citegnn.data.splits import load_split_artifact
from citegnn.evaluation import save_experiment_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--walks-per-node", type=int, default=10)
    parser.add_argument("--context-window", type=int, default=5)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = load_split_artifact(args.split_path)
    config = Node2VecConfig(
        embedding_dim=args.embedding_dim,
        walk_length=args.walk_length,
        walks_per_node=args.walks_per_node,
        context_window=args.context_window,
        negative_samples=args.negative_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        p=args.p,
        q=args.q,
        classifier_c=args.classifier_c,
    )
    result = run_node2vec_baseline(split, config=config)
    output_path = save_experiment_result(result, output_dir=args.output_dir)
    print(
        f"{result.dataset} {result.method}: "
        f"val_auc={result.validation['roc_auc']:.4f} "
        f"test_auc={result.test['roc_auc']:.4f} "
        f"-> {output_path}"
    )


if __name__ == "__main__":
    main()
