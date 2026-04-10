"""Classical heuristic baselines for link prediction."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Iterable

import networkx as nx
import numpy as np
from torch import Tensor

from ..data.splits import LinkPredictionSplit
from ..evaluation import (
    ExperimentResult,
    build_experiment_result,
    evaluate_with_validation_threshold,
)


class HeuristicMethod(StrEnum):
    """Supported heuristic scoring rules."""

    COMMON_NEIGHBORS = "common_neighbors"
    ADAMIC_ADAR = "adamic_adar"
    PERSONALIZED_PAGERANK = "personalized_pagerank"


def _edge_columns(edge_index: Tensor) -> list[tuple[int, int]]:
    """Convert an edge index into integer node pairs."""

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    return [(int(source), int(target)) for source, target in edge_index.t().tolist()]


def _build_scoring_graph(split: LinkPredictionSplit) -> nx.Graph | nx.DiGraph:
    """Build the train-only graph used by all heuristic baselines."""

    graph_type = nx.Graph if split.is_undirected else nx.DiGraph
    graph = graph_type()
    graph.add_nodes_from(range(int(split.train_mask.numel())))
    graph.add_edges_from(_edge_columns(split.train_graph_edge_index))
    return graph


def _combine_labeled_edges(
    pos_edge_index: Tensor,
    neg_edge_index: Tensor,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Combine positive and negative candidate edges with binary labels."""

    pos_edges = _edge_columns(pos_edge_index)
    neg_edges = _edge_columns(neg_edge_index)
    labels = np.concatenate(
        [
            np.ones(len(pos_edges), dtype=np.int64),
            np.zeros(len(neg_edges), dtype=np.int64),
        ]
    )
    return pos_edges + neg_edges, labels


def _score_common_neighbors(
    graph: nx.Graph | nx.DiGraph,
    edges: Iterable[tuple[int, int]],
) -> np.ndarray:
    undirected_graph = graph.to_undirected(as_view=True)
    scores: list[float] = []
    for source, target in edges:
        common_count = sum(1 for _ in nx.common_neighbors(undirected_graph, source, target))
        scores.append(float(common_count))
    return np.asarray(scores, dtype=float)


def _score_adamic_adar(
    graph: nx.Graph | nx.DiGraph,
    edges: Iterable[tuple[int, int]],
) -> np.ndarray:
    undirected_graph = graph.to_undirected(as_view=True)
    edge_list = list(edges)
    scores_by_edge = {
        (int(source), int(target)): float(score)
        for source, target, score in nx.adamic_adar_index(undirected_graph, edge_list)
    }
    return np.asarray([scores_by_edge[(source, target)] for source, target in edge_list], dtype=float)


def _score_personalized_pagerank(
    graph: nx.Graph | nx.DiGraph,
    edges: Iterable[tuple[int, int]],
    *,
    alpha: float,
) -> np.ndarray:
    edge_list = list(edges)
    scores: list[float] = []
    cache: dict[int, dict[int, float]] = {}

    for source, target in edge_list:
        if source not in cache:
            personalization = {node: 0.0 for node in graph.nodes}
            personalization[source] = 1.0
            cache[source] = nx.pagerank(graph, alpha=alpha, personalization=personalization)
        scores.append(float(cache[source].get(target, 0.0)))

    return np.asarray(scores, dtype=float)


def score_edges(
    graph: nx.Graph | nx.DiGraph,
    edges: Iterable[tuple[int, int]],
    *,
    method: HeuristicMethod,
    ppr_alpha: float,
) -> np.ndarray:
    """Score candidate edges with one heuristic method."""

    if method == HeuristicMethod.COMMON_NEIGHBORS:
        return _score_common_neighbors(graph, edges)
    if method == HeuristicMethod.ADAMIC_ADAR:
        return _score_adamic_adar(graph, edges)
    if method == HeuristicMethod.PERSONALIZED_PAGERANK:
        return _score_personalized_pagerank(graph, edges, alpha=ppr_alpha)
    raise ValueError(f"unsupported heuristic method: {method}")


def score_split_with_heuristic(
    split: LinkPredictionSplit,
    *,
    method: HeuristicMethod,
    ppr_alpha: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score validation and test candidate edges from one split artifact."""

    graph = _build_scoring_graph(split)
    val_edges, y_val_true = _combine_labeled_edges(
        split.val_pos_edge_index,
        split.val_neg_edge_index,
    )
    test_edges, y_test_true = _combine_labeled_edges(
        split.test_pos_edge_index,
        split.test_neg_edge_index,
    )
    y_val_score = score_edges(graph, val_edges, method=method, ppr_alpha=ppr_alpha)
    y_test_score = score_edges(graph, test_edges, method=method, ppr_alpha=ppr_alpha)
    return y_val_true, y_val_score, y_test_true, y_test_score


def run_heuristic_baseline(
    split: LinkPredictionSplit,
    *,
    method: HeuristicMethod,
    seed: int | None = None,
    ppr_alpha: float = 0.85,
) -> ExperimentResult:
    """Run one heuristic baseline and return the standard result payload."""

    run_seed = split.seed if seed is None else seed
    y_val_true, y_val_score, y_test_true, y_test_score = score_split_with_heuristic(
        split,
        method=method,
        ppr_alpha=ppr_alpha,
    )
    validation_metrics, test_metrics = evaluate_with_validation_threshold(
        y_val_true=y_val_true,
        y_val_score=y_val_score,
        y_test_true=y_test_true,
        y_test_score=y_test_score,
    )
    params: dict[str, Any] = {
        "split_seed": int(split.seed),
        "is_undirected": bool(split.is_undirected),
        "train_graph_edges": split.train_graph_edge_count,
    }
    if method == HeuristicMethod.PERSONALIZED_PAGERANK:
        params["alpha"] = float(ppr_alpha)

    return build_experiment_result(
        dataset=split.dataset_name,
        method=method.value,
        seed=run_seed,
        params=params,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )
