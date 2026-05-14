"""Backwards-compatibility shim that re-exports the HeaRT split API.

The original implementation in this module produced random
positive/negative edge splits using
:class:`torch_geometric.transforms.RandomLinkSplit`. As of the third and
final milestone of the project the team has switched to the protocol
proposed by Li et al. (NeurIPS 2023) "Evaluating Graph Neural Networks
for Link Prediction: Current Pitfalls and New Benchmarking" -- see
:mod:`citegnn.data.heart_split` for the current, canonical
implementation.

This module is kept so old call sites continue to work. New code should
import from :mod:`citegnn.data.heart_split` directly.
"""

from __future__ import annotations

from .heart_split import (
    DEFAULT_K,
    DEFAULT_PPR_ALPHA,
    DEFAULT_PPR_ITERATIONS,
    DEFAULT_SPLIT_RATIO,
    HeartSplit,
    create_heart_split,
    heart_split_artifact_name,
    load_heart_split,
    save_heart_split,
)

# Aliases that older code referred to.
LinkPredictionSplit = HeartSplit
create_link_prediction_split = create_heart_split
load_split_artifact = load_heart_split
save_split_artifact = save_heart_split

__all__ = [
    "DEFAULT_K",
    "DEFAULT_PPR_ALPHA",
    "DEFAULT_PPR_ITERATIONS",
    "DEFAULT_SPLIT_RATIO",
    "HeartSplit",
    "LinkPredictionSplit",
    "create_heart_split",
    "create_link_prediction_split",
    "heart_split_artifact_name",
    "load_heart_split",
    "load_split_artifact",
    "save_heart_split",
    "save_split_artifact",
]
