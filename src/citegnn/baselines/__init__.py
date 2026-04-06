"""Baseline methods for link prediction."""

from .heuristics import (
    HeuristicMethod,
    run_heuristic_baseline,
    score_split_with_heuristic,
)
from .node2vec import Node2VecConfig, run_node2vec_baseline

__all__ = [
    "HeuristicMethod",
    "Node2VecConfig",
    "run_heuristic_baseline",
    "run_node2vec_baseline",
    "score_split_with_heuristic",
]
