"""HeaRT-aligned split + hard negative generation for link prediction.

Implements the protocol from Li et al. (NeurIPS 2023)
"Evaluating Graph Neural Networks for Link Prediction:
Current Pitfalls and New Benchmarking".

Key differences from the legacy ``citegnn.data.splits`` module:

* The split is a single fixed 85/5/10 train/val/test partition over the
  *unique undirected edges* of the graph (no per-seed reshuffling of
  random negatives -- the seed only controls which 5%/10% of positives
  fall in val/test).
* Negatives are pre-computed *only for evaluation* (val + test). For each
  positive ``(a, b)`` we generate ``K`` hard negatives where ``K/2`` share
  the source ``a`` and ``K/2`` share the target ``b``. They are chosen by
  combining three heuristics -- Resource Allocation (RA), Personalized
  PageRank (PPR), and node-feature cosine similarity -- via Borda's
  ``min`` rank aggregation, then taking the top ``K/2`` candidates per
  side.
* Training negatives are *not* stored; the training loop draws fresh
  random negatives per epoch (HeaRT only changes the evaluation
  protocol, leaving training unchanged -- see Section 4.2 of the paper).

The serialized artifact lives at::

    artifacts/heart/<dataset>_seed<S>_K<K>_heart.pt

and is consumed by every baseline runner in :mod:`citegnn.baselines`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse as sp
from torch import Tensor

from .loaders import GraphData
from ..utils.seed import seed_everything

DEFAULT_K: int = 500
DEFAULT_SPLIT_RATIO: tuple[float, float, float] = (0.85, 0.05, 0.10)
DEFAULT_PPR_ALPHA: float = 0.85
DEFAULT_PPR_ITERATIONS: int = 50


@dataclass(slots=True)
class HeartSplit:
    """Fixed HeaRT-style split with pre-computed hard negative samples.

    Attributes
    ----------
    dataset_name:
        Human-readable dataset name (``"Cora"`` or ``"PubMed"``).
    seed:
        Seed controlling the random shuffle that assigns each unique edge
        to train / val / test. Same seed -> same split.
    split_ratio:
        ``(train, val, test)`` fractions, summing to one.
    K:
        Number of hard negatives per positive sample (``500`` per the
        HeaRT paper).
    num_nodes:
        Total node count.
    train_edge_index:
        ``[2, 2*P_train]`` symmetric (undirected) message-passing graph
        built from the training positives. Both directions of each edge
        are present so PyG aggregates correctly.
    train_pos_edge_index:
        ``[2, P_train]`` canonical-direction (``u < v``) positive
        supervision edges drawn from the training partition.
    val_pos_edge_index:
        ``[2, P_val]`` canonical-direction held-out validation edges.
    test_pos_edge_index:
        ``[2, P_test]`` canonical-direction held-out test edges.
    val_neg_nodes:
        ``[P_val, K]`` int64 node ids. For positive ``(a, b)``, the first
        ``K // 2`` columns are corruption nodes paired with ``a`` (i.e.
        edges ``(a, v_j)``); the remaining ``K - K // 2`` columns are
        corruption nodes paired with ``b`` (i.e. edges ``(u_j, b)``).
    test_neg_nodes:
        Same layout as ``val_neg_nodes`` but for test positives.
    heuristics_used:
        Heuristic names that contributed to the negatives (for
        bookkeeping / report tables).
    """

    dataset_name: str
    seed: int
    split_ratio: tuple[float, float, float]
    K: int
    num_nodes: int
    train_edge_index: Tensor
    train_pos_edge_index: Tensor
    val_pos_edge_index: Tensor
    test_pos_edge_index: Tensor
    val_neg_nodes: Tensor
    test_neg_nodes: Tensor
    heuristics_used: tuple[str, ...] = field(default_factory=tuple)

    @property
    def train_graph_edge_count(self) -> int:
        """Number of message-passing edges (counts both directions)."""

        return int(self.train_edge_index.size(1))

    @property
    def num_train_positives(self) -> int:
        return int(self.train_pos_edge_index.size(1))

    @property
    def num_val_positives(self) -> int:
        return int(self.val_pos_edge_index.size(1))

    @property
    def num_test_positives(self) -> int:
        return int(self.test_pos_edge_index.size(1))


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------


def _canonical_undirected_edges(edge_index: Tensor) -> Tensor:
    """Return one row per undirected edge with ``u < v``, deduplicated."""

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")

    src = edge_index[0]
    dst = edge_index[1]
    keep = src != dst
    src = src[keep]
    dst = dst[keep]
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    pairs = torch.stack([lo, hi], dim=1)
    pairs = torch.unique(pairs, dim=0)
    return pairs.t().contiguous()


def _shuffle_indices(num: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.arange(num)
    rng.shuffle(order)
    return order


def _split_positive_edges(
    canonical_edges: Tensor,
    *,
    seed: int,
    split_ratio: tuple[float, float, float],
) -> tuple[Tensor, Tensor, Tensor]:
    """Partition the canonical edges into train / val / test by index shuffle."""

    train_frac, val_frac, test_frac = split_ratio
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("split_ratio must sum to 1.0")
    if min(train_frac, val_frac, test_frac) <= 0.0:
        raise ValueError("split_ratio entries must be positive")

    num_edges = int(canonical_edges.size(1))
    if num_edges < 10:
        raise ValueError("not enough unique undirected edges to split")

    order = _shuffle_indices(num_edges, seed=seed)
    num_val = max(int(round(val_frac * num_edges)), 1)
    num_test = max(int(round(test_frac * num_edges)), 1)
    num_train = num_edges - num_val - num_test
    if num_train <= 0:
        raise ValueError("split ratios leave no training edges")

    train_idx = order[:num_train]
    val_idx = order[num_train : num_train + num_val]
    test_idx = order[num_train + num_val :]

    pairs = canonical_edges.t()
    train_pos = pairs[train_idx].t().contiguous()
    val_pos = pairs[val_idx].t().contiguous()
    test_pos = pairs[test_idx].t().contiguous()
    return train_pos, val_pos, test_pos


def _symmetrize(edge_index: Tensor) -> Tensor:
    """Add the reverse of every directed edge to obtain a symmetric graph."""

    reverse = torch.stack([edge_index[1], edge_index[0]], dim=0)
    return torch.cat([edge_index, reverse], dim=1).contiguous()


# ---------------------------------------------------------------------------
# Heuristic scoring (Resource Allocation, PPR, feature cosine)
# ---------------------------------------------------------------------------


def _build_train_adjacency(
    train_pos_edge_index: Tensor,
    num_nodes: int,
) -> sp.csr_matrix:
    """Build a 0/1 symmetric CSR adjacency from the training positives."""

    src = train_pos_edge_index[0].cpu().numpy()
    dst = train_pos_edge_index[1].cpu().numpy()
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(rows.shape[0], dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    A = A.tocsr()
    A.data[:] = 1.0  # collapse any duplicates
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def _row_normalized(A: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalize a CSR matrix; rows summing to zero are left unchanged."""

    degrees = np.asarray(A.sum(axis=1)).flatten()
    inv = np.zeros_like(degrees)
    nonzero = degrees > 0
    inv[nonzero] = 1.0 / degrees[nonzero]
    D_inv = sp.diags(inv)
    return D_inv @ A


def _resource_allocation_matrix(A: sp.csr_matrix) -> sp.csr_matrix:
    """Return ``RA = A @ D^-1 @ A`` with the diagonal removed.

    ``RA[a, b]`` equals the resource-allocation score for the unordered
    pair ``(a, b)``: ``sum_{c in N(a) ∩ N(b)} 1/deg(c)``.
    """

    degrees = np.asarray(A.sum(axis=1)).flatten()
    inv = np.zeros_like(degrees)
    nonzero = degrees > 0
    inv[nonzero] = 1.0 / degrees[nonzero]
    D_inv = sp.diags(inv)
    RA = A @ D_inv @ A
    RA = RA.tocsr()
    RA.setdiag(0.0)
    RA.eliminate_zeros()
    return RA


def _row_normalize_features(x: np.ndarray) -> np.ndarray:
    """Return L2-row-normalized features (zero rows mapped to zero)."""

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return x / safe


def _ppr_columns(
    P: sp.csr_matrix,
    sources: np.ndarray,
    *,
    alpha: float = DEFAULT_PPR_ALPHA,
    num_iters: int = DEFAULT_PPR_ITERATIONS,
    chunk_size: int = 64,
) -> np.ndarray:
    """Compute personalized PageRank for the requested source nodes.

    Uses power iteration on the row-normalized transition matrix ``P``
    (i.e. ``P[i, j] = 1 / deg(i)`` if ``(i, j)`` is an edge). Returns a
    dense ``[num_sources, num_nodes]`` matrix where row ``i`` is the
    stationary distribution of a random walker that restarts to
    ``sources[i]`` with probability ``1 - alpha``.
    """

    num_nodes = P.shape[0]
    num_sources = sources.shape[0]
    out = np.zeros((num_sources, num_nodes), dtype=np.float32)
    PT = P.T.tocsr()  # so that PT @ x propagates one walk step
    for start in range(0, num_sources, chunk_size):
        stop = min(start + chunk_size, num_sources)
        batch = sources[start:stop]
        teleport = np.zeros((num_nodes, batch.shape[0]), dtype=np.float32)
        teleport[batch, np.arange(batch.shape[0])] = 1.0
        scores = teleport.copy()
        for _ in range(num_iters):
            scores = (1.0 - alpha) * teleport + alpha * (PT @ scores)
        out[start:stop] = scores.T
    return out


# ---------------------------------------------------------------------------
# Hard-negative generation
# ---------------------------------------------------------------------------


def _build_filter_set(
    *,
    train_pos: Tensor,
    val_pos: Tensor | None,
    num_nodes: int,
) -> tuple[set[int], set[int]]:
    """Helper bookkeeping. Not used directly; see ``_forbidden_neighbors``."""

    raise NotImplementedError("kept for backwards-compatibility hooks")


def _forbidden_neighbors(
    *,
    anchor: int,
    train_adj: sp.csr_matrix,
    extra_forbidden: dict[int, set[int]] | None = None,
) -> set[int]:
    """All nodes that must NOT be sampled as a corruption for ``anchor``.

    This excludes:

    * ``anchor`` itself (no self-loop);
    * every training neighbour of ``anchor`` (these are observed positive
      edges, not negatives);
    * any extra positives (e.g. validation positives when generating
      test negatives).
    """

    forbidden: set[int] = {anchor}
    row = train_adj.getrow(anchor)
    forbidden.update(int(j) for j in row.indices)
    if extra_forbidden is not None and anchor in extra_forbidden:
        forbidden.update(extra_forbidden[anchor])
    return forbidden


def _argsort_descending(scores: np.ndarray) -> np.ndarray:
    """Return indices that sort ``scores`` from largest to smallest."""

    return np.argsort(-scores, kind="stable")


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Return per-node 1-based rank with optimistic tie-handling.

    Nodes with the highest score get rank ``1``. Zero-score nodes are
    pushed to the bottom (their rank is ``len(scores)``-style large)
    because Borda's ``min`` aggregation lets *any* heuristic that scored
    a node carry it forward.
    """

    order = _argsort_descending(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, order.size + 1)
    # Nodes with score == 0 in this heuristic shouldn't dominate the min
    # combination just by virtue of getting an early position from
    # argsort. Push them all the way down.
    zero_mask = scores <= 0.0
    if zero_mask.any():
        ranks[zero_mask] = order.size + 1
    return ranks


def _select_negatives_for_anchor(
    *,
    anchor: int,
    half_K: int,
    forbidden: set[int],
    score_lists: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply Algorithm 1 of the HeaRT paper to pick ``half_K`` corruptions."""

    num_nodes = score_lists[0].shape[0]
    rank_lists = [_ranks_from_scores(s) for s in score_lists]
    combined = np.minimum.reduce(rank_lists)

    if forbidden:
        forbidden_arr = np.fromiter(forbidden, dtype=np.int64)
        combined[forbidden_arr] = num_nodes + 2

    order = np.argsort(combined, kind="stable")
    valid = order[combined[order] <= num_nodes]
    if valid.size >= half_K:
        return valid[:half_K].astype(np.int64)

    # Pad with random nodes, avoiding forbidden ones and previously-selected
    # ones when possible. Tiny graphs may not have enough non-forbidden nodes
    # left; in that case fall back to sampling with replacement so the eval
    # tensor still has the requested width.
    selected = set(int(v) for v in valid.tolist())
    selected.update(int(v) for v in forbidden)
    pool = np.fromiter(
        (i for i in range(num_nodes) if i not in selected),
        dtype=np.int64,
    )
    needed = half_K - valid.size
    if pool.size >= needed:
        extras = rng.choice(pool, size=needed, replace=False)
    elif pool.size > 0:
        extras = rng.choice(pool, size=needed, replace=True)
    else:
        # Only forbidden nodes remain; reuse non-anchor nodes instead.
        fallback_pool = np.fromiter(
            (i for i in range(num_nodes) if i != anchor),
            dtype=np.int64,
        )
        if fallback_pool.size == 0:
            raise RuntimeError(
                f"cannot generate {half_K} negatives for anchor {anchor}: "
                "graph has only one node"
            )
        extras = rng.choice(fallback_pool, size=needed, replace=True)
    return np.concatenate([valid.astype(np.int64), extras.astype(np.int64)])


def _generate_negatives_for_partition(
    *,
    pos_edges: Tensor,
    train_adj: sp.csr_matrix,
    ra_matrix: sp.csr_matrix,
    ppr_scores: dict[int, np.ndarray],
    feat_norm: np.ndarray,
    K: int,
    extra_forbidden: dict[int, set[int]] | None,
    rng: np.random.Generator,
    progress_label: str,
    log_every: int,
) -> Tensor:
    """Generate the ``[P, K]`` HeaRT negatives for one partition (val/test)."""

    half_K = K // 2
    second_K = K - half_K  # in case K is odd
    num_pos = int(pos_edges.size(1))
    out = np.zeros((num_pos, K), dtype=np.int64)

    src_array = pos_edges[0].cpu().numpy()
    dst_array = pos_edges[1].cpu().numpy()

    for i in range(num_pos):
        a = int(src_array[i])
        b = int(dst_array[i])

        ra_a = np.asarray(ra_matrix.getrow(a).todense()).flatten()
        ra_b = np.asarray(ra_matrix.getrow(b).todense()).flatten()
        feat_a = feat_norm @ feat_norm[a]
        feat_b = feat_norm @ feat_norm[b]
        ppr_a = ppr_scores[a]
        ppr_b = ppr_scores[b]

        forbid_a = _forbidden_neighbors(
            anchor=a,
            train_adj=train_adj,
            extra_forbidden=extra_forbidden,
        )
        forbid_a.add(b)
        forbid_b = _forbidden_neighbors(
            anchor=b,
            train_adj=train_adj,
            extra_forbidden=extra_forbidden,
        )
        forbid_b.add(a)

        neg_left = _select_negatives_for_anchor(
            anchor=a,
            half_K=half_K,
            forbidden=forbid_a,
            score_lists=[ra_a, ppr_a, feat_a],
            rng=rng,
        )
        neg_right = _select_negatives_for_anchor(
            anchor=b,
            half_K=second_K,
            forbidden=forbid_b,
            score_lists=[ra_b, ppr_b, feat_b],
            rng=rng,
        )
        out[i, :half_K] = neg_left
        out[i, half_K:] = neg_right

        if log_every and (i + 1) % log_every == 0:
            print(f"  [{progress_label}] {i + 1}/{num_pos} positives processed")

    return torch.from_numpy(out)


def _collect_unique_anchors(*partitions: Tensor) -> np.ndarray:
    """Union of all node ids appearing as either endpoint in val/test."""

    parts: list[np.ndarray] = []
    for partition in partitions:
        if partition.numel() == 0:
            continue
        parts.append(partition[0].cpu().numpy())
        parts.append(partition[1].cpu().numpy())
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(parts)).astype(np.int64)


def _build_extra_forbidden(pos_edges: Tensor) -> dict[int, set[int]]:
    """Per-anchor lookup of all positive partners (used as filter set)."""

    out: dict[int, set[int]] = {}
    src = pos_edges[0].cpu().numpy()
    dst = pos_edges[1].cpu().numpy()
    for u, v in zip(src.tolist(), dst.tolist()):
        out.setdefault(int(u), set()).add(int(v))
        out.setdefault(int(v), set()).add(int(u))
    return out


# ---------------------------------------------------------------------------
# Public split factory + persistence
# ---------------------------------------------------------------------------


def create_heart_split(
    graph: GraphData,
    *,
    seed: int = 1,
    K: int = DEFAULT_K,
    split_ratio: tuple[float, float, float] = DEFAULT_SPLIT_RATIO,
    ppr_alpha: float = DEFAULT_PPR_ALPHA,
    ppr_iterations: int = DEFAULT_PPR_ITERATIONS,
    log_every: int = 200,
) -> HeartSplit:
    """Build a single fixed HeaRT-aligned split with hard negatives.

    The returned object is fully self-contained: it stores the exact
    message-passing graph, the val/test positive edges, and the
    pre-computed ``[P, K]`` hard negatives generated by combining
    Resource Allocation, Personalized PageRank and node-feature cosine
    similarity via Borda's ``min``.
    """

    if K <= 0 or K % 2 != 0:
        raise ValueError("K must be a positive even number")
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    canonical = _canonical_undirected_edges(graph.edge_index)
    train_pos, val_pos, test_pos = _split_positive_edges(
        canonical,
        seed=seed,
        split_ratio=split_ratio,
    )

    train_edge_index = _symmetrize(train_pos)
    train_adj = _build_train_adjacency(train_pos, num_nodes=graph.num_nodes)
    P = _row_normalized(train_adj)
    ra_matrix = _resource_allocation_matrix(train_adj)

    feat_norm = _row_normalize_features(
        graph.x.detach().cpu().to(dtype=torch.float32).numpy()
    )

    anchors = _collect_unique_anchors(val_pos, test_pos)
    print(
        f"[heart_split] {graph.name}: {anchors.size} unique anchor nodes; "
        f"running PPR (alpha={ppr_alpha}, iters={ppr_iterations})..."
    )
    ppr_dense = _ppr_columns(
        P,
        anchors,
        alpha=ppr_alpha,
        num_iters=ppr_iterations,
    )
    ppr_scores: dict[int, np.ndarray] = {
        int(node): ppr_dense[i] for i, node in enumerate(anchors)
    }
    print("[heart_split] PPR done. Generating validation negatives...")

    val_neg = _generate_negatives_for_partition(
        pos_edges=val_pos,
        train_adj=train_adj,
        ra_matrix=ra_matrix,
        ppr_scores=ppr_scores,
        feat_norm=feat_norm,
        K=K,
        extra_forbidden=None,
        rng=rng,
        progress_label="val",
        log_every=log_every,
    )

    extra_forbidden = _build_extra_forbidden(val_pos)
    print("[heart_split] Generating test negatives...")
    test_neg = _generate_negatives_for_partition(
        pos_edges=test_pos,
        train_adj=train_adj,
        ra_matrix=ra_matrix,
        ppr_scores=ppr_scores,
        feat_norm=feat_norm,
        K=K,
        extra_forbidden=extra_forbidden,
        rng=rng,
        progress_label="test",
        log_every=log_every,
    )

    return HeartSplit(
        dataset_name=graph.name,
        seed=seed,
        split_ratio=split_ratio,
        K=K,
        num_nodes=graph.num_nodes,
        train_edge_index=train_edge_index,
        train_pos_edge_index=train_pos,
        val_pos_edge_index=val_pos,
        test_pos_edge_index=test_pos,
        val_neg_nodes=val_neg,
        test_neg_nodes=test_neg,
        heuristics_used=("resource_allocation", "personalized_pagerank", "feature_cosine"),
    )


def heart_split_artifact_name(dataset_name: str, *, seed: int, K: int) -> str:
    return f"{dataset_name.lower()}_seed{seed}_K{K}_heart.pt"


def save_heart_split(
    split: HeartSplit,
    *,
    output_dir: str | Path = "artifacts/heart",
) -> Path:
    """Persist a HeaRT split to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / heart_split_artifact_name(
        split.dataset_name,
        seed=split.seed,
        K=split.K,
    )
    payload = asdict(split)
    payload["split_ratio"] = tuple(split.split_ratio)
    payload["heuristics_used"] = tuple(split.heuristics_used)
    torch.save(payload, artifact_path)
    return artifact_path


def load_heart_split(path: str | Path) -> HeartSplit:
    """Load a persisted HeaRT split from disk."""

    payload: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=False)
    payload["split_ratio"] = tuple(payload["split_ratio"])
    payload["heuristics_used"] = tuple(payload.get("heuristics_used", ()))
    return HeartSplit(**payload)


__all__ = [
    "DEFAULT_K",
    "DEFAULT_PPR_ALPHA",
    "DEFAULT_PPR_ITERATIONS",
    "DEFAULT_SPLIT_RATIO",
    "HeartSplit",
    "create_heart_split",
    "heart_split_artifact_name",
    "load_heart_split",
    "save_heart_split",
]
