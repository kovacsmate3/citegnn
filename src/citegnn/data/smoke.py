"""CLI entrypoint for building HeaRT splits with hard negatives.

Usage::

    python scripts/build_heart_split.py --dataset Cora --seed 1
    python scripts/build_heart_split.py --dataset PubMed --seed 1

The previous random-positive/random-negative split protocol has been
removed (see :mod:`citegnn.data.heart_split` for the replacement). This
CLI is the only supported entrypoint for producing split artifacts going
forward.
"""

from __future__ import annotations

import argparse
import json

from .heart_split import (
    DEFAULT_K,
    DEFAULT_PPR_ALPHA,
    DEFAULT_PPR_ITERATIONS,
    DEFAULT_SPLIT_RATIO,
    create_heart_split,
    save_heart_split,
)
from .loaders import CitationDataset, load_citation_graph
from ..utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and persist a HeaRT split artifact (split + hard negatives)."
    )
    parser.add_argument(
        "--dataset",
        choices=[dataset.value for dataset in CitationDataset],
        required=True,
        help="Citation dataset to load.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed controlling the train/val/test edge shuffle and negative RNG.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=DEFAULT_K,
        help="Number of HeaRT hard negatives per positive (must be even).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIO[0],
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIO[1],
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIO[2],
    )
    parser.add_argument(
        "--ppr-alpha",
        type=float,
        default=DEFAULT_PPR_ALPHA,
    )
    parser.add_argument(
        "--ppr-iterations",
        type=int,
        default=DEFAULT_PPR_ITERATIONS,
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/heart",
        help="Directory where the split artifact is stored.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    graph = load_citation_graph(args.dataset)
    split = create_heart_split(
        graph,
        seed=args.seed,
        K=args.K,
        split_ratio=(args.train_ratio, args.val_ratio, args.test_ratio),
        ppr_alpha=args.ppr_alpha,
        ppr_iterations=args.ppr_iterations,
    )
    artifact_path = save_heart_split(split, output_dir=args.output_dir)

    summary = {
        "dataset": graph.name,
        "num_nodes": graph.num_nodes,
        "num_features": graph.num_features,
        "input_graph_edge_count": int(graph.edge_index.size(1)),
        "split_ratio": list(split.split_ratio),
        "K": int(split.K),
        "seed": int(split.seed),
        "train_graph_edges_symmetric": split.train_graph_edge_count,
        "num_train_positives": split.num_train_positives,
        "num_val_positives": split.num_val_positives,
        "num_test_positives": split.num_test_positives,
        "heuristics_used": list(split.heuristics_used),
        "artifact_path": str(artifact_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
