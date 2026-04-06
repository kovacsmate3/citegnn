"""Baseline methods for link prediction."""

from .heuristics import (
    HeuristicMethod,
    run_heuristic_baseline,
    score_split_with_heuristic,
)

__all__ = [
    "HeuristicMethod",
    "run_heuristic_baseline",
    "score_split_with_heuristic",
]
