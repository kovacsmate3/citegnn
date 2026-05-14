"""Heuristic link-prediction baselines under the HeaRT protocol.

Three classical heuristics are evaluated:

* **Common Neighbors (CN)** -- ``CN(u, v) = |N(u) ∩ N(v)|``.
* **Adamic-Adar (AA)** -- ``AA(u, v) = sum_{c in N(u) ∩ N(v)} 1/log(deg(c))``.
* **Personalized PageRank (PPR)** -- the stationary distribution at ``v``
  of a random walker that restarts to ``u`` with probability ``1 - alpha``.

CN and AA are computed once over the training adjacency as sparse
matrices and then sliced row-by-row to score evaluation edges. PPR is
precomputed for the endpoints of every val/test positive (the smallest
node set guaranteed to cover every evaluation edge under the HeaRT
negative layout).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import torch
from scipy import sparse as sp
from torch import Tensor

from ..data.heart_split import (
    DEFAULT_PPR_ALPHA,
    DEFAULT_PPR_ITERATIONS,
    HeartSplit,
    _build_train_adjacency,
    _collect_unique_anchors,
    _ppr_columns,
    _row_normalized,
)
from ..evaluation import (
    ExperimentResult,
    build_experiment_result,
    evaluate_with_heart_negatives,
)


class HeuristicMethod(StrEnum):
    """Supported heuristic scoring rules."""

    COMMON_NEIGHBORS = "common_neighbors"
    ADAMIC_ADAR = "adamic_adar"
    PERSONALIZED_PAGERANK = "personalized_pagerank"


@dataclass(slots=True)
class HeuristicConfig:
    """Configuration for the personalized PageRank heuristic."""

    ppr_alpha: float = DEFAULT_PPR_ALPHA
    ppr_iterations: int = DEFAULT_PPR_ITERATIONS

    def to_dict(self) -> dict[str, float | int]:
        return {
            "ppr_alpha": float(self.ppr_alpha),
            "ppr_iterations": int(self.ppr_iterations),
        }


# ---------------------------------------------------------------------------
# Sparse heuristic matrices
# ---------------------------------------------------------------------------


def _adamic_adar_weights(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Return the column-weighted ``A`` used in AA scoring.

    The result has the same sparsity pattern as ``adj`` but every column
    ``j`` is multiplied by ``1 / log(deg(j))``. Multiplying it row-wise
    against ``adj`` then summing reproduces the Adamic-Adar score.
    """

    degrees = np.asarray(adj.sum(axis=1)).flatten()
    safe = np.where(degrees > 1.0, degrees, 2.0)
    weights = 1.0 / np.log(safe)
    weights[degrees < 2.0] = 0.0
    weight_diag = sp.diags(weights)
    return adj @ weight_diag


def _score_pairwise_sparse(
    edges: Tensor,
    *,
    left: sp.csr_matrix,
    right: sp.csr_matrix,
) -> Tensor:
    """Compute ``sum(left[u] * right[v])`` per edge via sparse Hadamard products."""

    src = edges[0].cpu().numpy()
    dst = edges[1].cpu().numpy()
    L = left[src]
    R = right[dst]
    products = L.multiply(R)
    raw = np.asarray(products.sum(axis=1)).flatten()
    return torch.from_numpy(raw.astype(np.float64))


def _build_cn_score_fn(adj: sp.csr_matrix):
    def _score(edges: Tensor) -> Tensor:
        return _score_pairwise_sparse(edges, left=adj, right=adj)

    return _score


def _build_aa_score_fn(adj: sp.csr_matrix):
    weighted = _adamic_adar_weights(adj)

    def _score(edges: Tensor) -> Tensor:
        return _score_pairwise_sparse(edges, left=weighted, right=adj)

    return _score


def _build_ppr_score_fn(ppr: dict[int, np.ndarray]):
    """Symmetric PPR scoring; works as long as one endpoint has PPR."""

    def _score(edges: Tensor) -> Tensor:
        src = edges[0].cpu().numpy()
        dst = edges[1].cpu().numpy()
        scores = np.zeros(src.shape[0], dtype=np.float64)
        for i in range(src.shape[0]):
            u = int(src[i])
            v = int(dst[i])
            value = 0.0
            row_u = ppr.get(u)
            if row_u is not None:
                value += float(row_u[v])
            row_v = ppr.get(v)
            if row_v is not None:
                value += float(row_v[u])
            scores[i] = value
        return torch.from_numpy(scores)

    return _score


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_heuristic_baseline(
    split: HeartSplit,
    *,
    method: HeuristicMethod,
    config: HeuristicConfig | None = None,
    seed: int | None = None,
) -> ExperimentResult:
    """Score val/test under the HeaRT protocol with the requested heuristic."""

    resolved_config = config or HeuristicConfig()
    run_seed = split.seed if seed is None else seed

    adj = _build_train_adjacency(split.train_pos_edge_index, num_nodes=split.num_nodes)

    if method == HeuristicMethod.COMMON_NEIGHBORS:
        score_fn = _build_cn_score_fn(adj)
    elif method == HeuristicMethod.ADAMIC_ADAR:
        score_fn = _build_aa_score_fn(adj)
    elif method == HeuristicMethod.PERSONALIZED_PAGERANK:
        anchors = _collect_unique_anchors(
            split.val_pos_edge_index, split.test_pos_edge_index
        )
        P = _row_normalized(adj)
        ppr_matrix = _ppr_columns(
            P,
            anchors,
            alpha=resolved_config.ppr_alpha,
            num_iters=resolved_config.ppr_iterations,
        )
        ppr = {int(node): ppr_matrix[i] for i, node in enumerate(anchors)}
        score_fn = _build_ppr_score_fn(ppr)
    else:
        raise ValueError(f"unsupported heuristic method: {method}")

    validation, test = evaluate_with_heart_negatives(
        score_fn=score_fn,
        val_pos_edge_index=split.val_pos_edge_index,
        val_neg_nodes=split.val_neg_nodes,
        test_pos_edge_index=split.test_pos_edge_index,
        test_neg_nodes=split.test_neg_nodes,
    )

    params: dict[str, Any] = {
        "K": int(split.K),
        "split_seed": int(split.seed),
        "train_graph_edges": split.train_graph_edge_count,
        "num_train_positives": split.num_train_positives,
        "num_val_positives": split.num_val_positives,
        "num_test_positives": split.num_test_positives,
    }
    if method == HeuristicMethod.PERSONALIZED_PAGERANK:
        params.update(resolved_config.to_dict())

    return build_experiment_result(
        dataset=split.dataset_name,
        method=method.value,
        seed=run_seed,
        params=params,
        validation=validation,
        test=test,
    )


__all__ = [
    "HeuristicConfig",
    "HeuristicMethod",
    "run_heuristic_baseline",
]
