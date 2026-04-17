"""Baseline methods for link prediction."""

from .feature_only import (
    FeatureOnlyBaselineConfig,
    build_pairwise_feature_dataset,
    run_feature_only_baseline,
)
from .heuristics import (
    HeuristicMethod,
    run_heuristic_baseline,
    score_split_with_heuristic,
)
from .gnn import GNNBaselineConfig, GNNMethod, run_gnn_baseline
from .node2vec import Node2VecConfig, run_node2vec_baseline

__all__ = [
    "FeatureOnlyBaselineConfig",
    "GNNBaselineConfig",
    "GNNMethod",
    "HeuristicMethod",
    "Node2VecConfig",
    "build_pairwise_feature_dataset",
    "run_feature_only_baseline",
    "run_gnn_baseline",
    "run_heuristic_baseline",
    "run_node2vec_baseline",
    "score_split_with_heuristic",
]
