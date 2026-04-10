"""Baseline methods for link prediction."""

from .heuristics import (
    HeuristicMethod,
    run_heuristic_baseline,
    score_split_with_heuristic,
)
from .gnn import GNNBaselineConfig, GNNMethod, run_gnn_baseline
from .node2vec import Node2VecConfig, run_node2vec_baseline

__all__ = [
    "GNNBaselineConfig",
    "GNNMethod",
    "HeuristicMethod",
    "Node2VecConfig",
    "run_gnn_baseline",
    "run_heuristic_baseline",
    "run_node2vec_baseline",
    "score_split_with_heuristic",
]
