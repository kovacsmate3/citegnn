"""Baseline methods for link prediction (HeaRT protocol)."""

from .augmentation import augment_features, compute_structural_features
from .feature_only import (
    FeatureOnlyBaselineConfig,
    run_feature_only_baseline,
)
from .gnn import (
    GNNBaselineConfig,
    GNNMethod,
    run_gnn_baseline,
)
from .heuristics import HeuristicConfig, HeuristicMethod, run_heuristic_baseline
from .node2vec import Node2VecConfig, run_node2vec_baseline

__all__ = [
    "FeatureOnlyBaselineConfig",
    "GNNBaselineConfig",
    "GNNMethod",
    "HeuristicConfig",
    "HeuristicMethod",
    "Node2VecConfig",
    "augment_features",
    "compute_structural_features",
    "run_feature_only_baseline",
    "run_gnn_baseline",
    "run_heuristic_baseline",
    "run_node2vec_baseline",
]
