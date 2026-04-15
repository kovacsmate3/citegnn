"""Smoke-test entrypoint for the data pipeline."""

from __future__ import annotations

import argparse
import json

from .loaders import CitationDataset, load_citation_graph
from .splits import create_link_prediction_split, save_split_artifact
from ..utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and persist a PyG RandomLinkSplit artifact."
    )
    parser.add_argument(
        "--dataset",
        choices=[dataset.value for dataset in CitationDataset],
        required=True,
        help="Citation dataset to load.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation edge ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test edge ratio.")
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Negative-to-positive sampling ratio for each supervision split.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/splits",
        help="Directory where the split artifact is stored.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    graph = load_citation_graph(
        args.dataset,
    )
    split = create_link_prediction_split(
        graph,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    artifact_path = save_split_artifact(split, output_dir=args.output_dir)

    summary = {
        "dataset": graph.name,
        "num_nodes": graph.num_nodes,
        "num_features": graph.num_features,
        "train_nodes": int(graph.train_mask.sum()),
        "val_nodes": int(graph.val_mask.sum()),
        "test_nodes": int(graph.test_mask.sum()),
        "input_graph_edge_count": int(graph.edge_index.size(1)),
        "train_graph_edge_count": int(split.train_graph_edge_index.size(1)),
        "val_graph_edge_count": int(split.val_graph_edge_index.size(1)),
        "test_graph_edge_count": int(split.test_graph_edge_index.size(1)),
        "train_pos_edges": int(split.train_pos_edge_index.size(1)),
        "val_pos_edges": int(split.val_pos_edge_index.size(1)),
        "test_pos_edges": int(split.test_pos_edge_index.size(1)),
        "train_neg_edges": int(split.train_neg_edge_index.size(1)),
        "val_neg_edges": int(split.val_neg_edge_index.size(1)),
        "test_neg_edges": int(split.test_neg_edge_index.size(1)),
        "artifact_path": str(artifact_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
