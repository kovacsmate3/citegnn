"""Reusable evaluation helpers and result schema (HeaRT-aligned)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .utils.io import write_json
from .utils.metrics import (
    ClassificationMetrics,
    choose_best_threshold,
    compute_classification_metrics,
)
from .utils.ranking_metrics import (
    DEFAULT_HITS_KS,
    RankingMetrics,
    compute_ranking_metrics,
)

# ---------------------------------------------------------------------------
# Experiment result schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExperimentResult:
    """Serializable experiment result for one dataset-method-seed run.

    The ``validation`` and ``test`` dictionaries always include the HeaRT
    ranking metrics (``mrr``, ``hits_at_k``) and -- for completeness --
    the binary classification metrics (``roc_auc``, ``average_precision``)
    computed by treating all per-positive negatives as a flat negative
    set.
    """

    dataset: str
    method: str
    seed: int
    params: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, float] = field(default_factory=dict)
    test: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_experiment_result(
    result: ExperimentResult,
    *,
    metric: str = "mrr",
    include_validation: bool = True,
) -> str:
    """Build a compact CLI summary line for one experiment result."""

    parts: list[str] = []
    if include_validation:
        if metric in result.validation:
            parts.append(f"val_{metric}={result.validation[metric]:.4f}")
    if metric in result.test:
        parts.append(f"test_{metric}={result.test[metric]:.4f}")
    if "hits_at_10" in result.test:
        parts.append(f"test_hits@10={result.test['hits_at_10']:.4f}")
    if "roc_auc" in result.test:
        parts.append(f"test_auc={result.test['roc_auc']:.4f}")
    return f"{result.dataset} {result.method}: {' '.join(parts)}"


# ---------------------------------------------------------------------------
# HeaRT-aware evaluation
# ---------------------------------------------------------------------------

ScoreFn = Callable[[Tensor], Tensor]


def _to_score_tensor(x: Tensor | np.ndarray | list[float]) -> Tensor:
    if isinstance(x, Tensor):
        return x.detach().cpu().to(dtype=torch.float64).flatten()
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


def score_pos_and_heart_negatives(
    *,
    pos_edge_index: Tensor,
    neg_nodes: Tensor,
    score_fn: ScoreFn,
    batch_size: int = 65536,
) -> tuple[Tensor, Tensor]:
    """Return ``(pos_scores [P], neg_scores [P, K])`` from a generic scorer.

    ``score_fn`` is invoked with edge tensors of shape ``[2, M]`` and must
    return a one-dimensional score tensor of shape ``[M]``. Edges are
    assembled in chunks of at most ``batch_size`` columns to bound peak
    memory.
    """

    if pos_edge_index.dim() != 2 or pos_edge_index.size(0) != 2:
        raise ValueError("pos_edge_index must have shape [2, P]")
    if neg_nodes.dim() != 2:
        raise ValueError("neg_nodes must have shape [P, K]")
    if pos_edge_index.size(1) != neg_nodes.size(0):
        raise ValueError("pos_edge_index and neg_nodes disagree on the number of positives")

    P = int(pos_edge_index.size(1))
    K = int(neg_nodes.size(1))
    half_K = K // 2

    pos = pos_edge_index.cpu().to(dtype=torch.long)
    negs = neg_nodes.cpu().to(dtype=torch.long)

    # Build a single edge tensor of width P*(K+1):
    #   columns [0, P)        -> positives (a_i, b_i)
    #   columns [P, P + P*K)  -> negatives (per-positive layout)
    a = pos[0]
    b = pos[1]

    pos_left = a.unsqueeze(1).expand(-1, half_K).reshape(-1)
    neg_left_v = negs[:, :half_K].reshape(-1)
    neg_right_u = negs[:, half_K:].reshape(-1)
    pos_right = b.unsqueeze(1).expand(-1, K - half_K).reshape(-1)

    neg_src = torch.cat([pos_left, neg_right_u], dim=0)
    neg_dst = torch.cat([neg_left_v, pos_right], dim=0)

    all_src = torch.cat([a, neg_src], dim=0)
    all_dst = torch.cat([b, neg_dst], dim=0)
    all_edges = torch.stack([all_src, all_dst], dim=0)

    scores: list[Tensor] = []
    total = all_edges.size(1)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        chunk = all_edges[:, start:stop]
        chunk_scores = _to_score_tensor(score_fn(chunk))
        if chunk_scores.numel() != (stop - start):
            raise ValueError(
                f"score_fn returned {chunk_scores.numel()} values for {stop - start} edges"
            )
        scores.append(chunk_scores)
    flat = torch.cat(scores, dim=0)

    pos_scores = flat[:P]
    neg_left_scores = flat[P : P + P * half_K].reshape(P, half_K)
    neg_right_scores = flat[P + P * half_K :].reshape(P, K - half_K)
    neg_scores = torch.cat([neg_left_scores, neg_right_scores], dim=1)
    return pos_scores, neg_scores


def metrics_from_pos_neg_scores(
    pos_scores: Tensor,
    neg_scores: Tensor,
    *,
    hits_ks: Sequence[int] = DEFAULT_HITS_KS,
    include_classification: bool = True,
) -> tuple[RankingMetrics, ClassificationMetrics | None]:
    """Compute HeaRT ranking metrics (and, optionally, flat-pool AUC/AP)."""

    ranking = compute_ranking_metrics(pos_scores, neg_scores, hits_ks=hits_ks)
    if not include_classification:
        return ranking, None

    # Flat-pool classification metrics. AUC + AP only -- we skip threshold-
    # dependent metrics because the flat pool is heavily imbalanced (1
    # positive : K negatives) and the F1-optimal threshold sweep is O(N^2)
    # which dominates eval cost at K=500.
    flat_scores = torch.cat([pos_scores.flatten(), neg_scores.flatten()], dim=0)
    flat_labels = torch.cat(
        [
            torch.ones(pos_scores.numel(), dtype=torch.int64),
            torch.zeros(neg_scores.numel(), dtype=torch.int64),
        ],
        dim=0,
    )
    classification = ClassificationMetrics(
        roc_auc=_binary_roc_auc(flat_labels, flat_scores),
        average_precision=_binary_average_precision(flat_labels, flat_scores),
        precision=0.0,
        recall=0.0,
        f1=0.0,
        threshold=0.0,
        positive_rate=0.0,
    )
    return ranking, classification


def _binary_roc_auc(labels: Tensor, scores: Tensor) -> float:
    from .utils.metrics import _binary_roc_auc_torch  # local import to avoid cycle
    return _binary_roc_auc_torch(labels.to(dtype=torch.int64), scores.to(dtype=torch.float64))


def _binary_average_precision(labels: Tensor, scores: Tensor) -> float:
    from .utils.metrics import _binary_average_precision_torch
    return _binary_average_precision_torch(labels.to(dtype=torch.int64), scores.to(dtype=torch.float64))


def _merge_metrics(
    ranking: RankingMetrics,
    classification: ClassificationMetrics,
) -> dict[str, float]:
    payload: dict[str, float] = {}
    payload.update({k: float(v) for k, v in ranking.to_dict().items()})
    for key, value in classification.to_dict().items():
        payload[key] = float(value)
    return payload


# ---------------------------------------------------------------------------
# Generic helper used by every baseline
# ---------------------------------------------------------------------------


def evaluate_with_heart_negatives(
    *,
    score_fn: ScoreFn,
    val_pos_edge_index: Tensor,
    val_neg_nodes: Tensor,
    test_pos_edge_index: Tensor,
    test_neg_nodes: Tensor,
    hits_ks: Sequence[int] = DEFAULT_HITS_KS,
    batch_size: int = 65536,
) -> tuple[dict[str, float], dict[str, float]]:
    """Score val/test under the HeaRT protocol.

    Returns
    -------
    (validation_metrics, test_metrics) -- dicts ready to drop into
    :class:`ExperimentResult`.
    """

    val_pos_scores, val_neg_scores = score_pos_and_heart_negatives(
        pos_edge_index=val_pos_edge_index,
        neg_nodes=val_neg_nodes,
        score_fn=score_fn,
        batch_size=batch_size,
    )
    val_ranking, val_classification = metrics_from_pos_neg_scores(
        val_pos_scores, val_neg_scores, hits_ks=hits_ks
    )

    test_pos_scores, test_neg_scores = score_pos_and_heart_negatives(
        pos_edge_index=test_pos_edge_index,
        neg_nodes=test_neg_nodes,
        score_fn=score_fn,
        batch_size=batch_size,
    )
    test_ranking, test_classification = metrics_from_pos_neg_scores(
        test_pos_scores, test_neg_scores, hits_ks=hits_ks
    )

    return _merge_metrics(val_ranking, val_classification), _merge_metrics(
        test_ranking, test_classification
    )


def build_experiment_result(
    *,
    dataset: str,
    method: str,
    seed: int,
    params: dict[str, Any],
    validation: dict[str, float],
    test: dict[str, float],
) -> ExperimentResult:
    """Create the standard experiment result payload."""

    return ExperimentResult(
        dataset=dataset,
        method=method,
        seed=seed,
        params=params,
        validation=validation,
        test=test,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_experiment_result(
    result: ExperimentResult,
    *,
    output_dir: str | Path = "results",
) -> Path:
    output_path = (
        Path(output_dir)
        / result.dataset.lower()
        / f"{result.method}_seed{result.seed}.json"
    )
    return write_json(result.to_dict(), output_path)


def save_and_describe_experiment_result(
    result: ExperimentResult,
    *,
    output_dir: str | Path = "results",
    metric: str = "mrr",
) -> tuple[Path, str]:
    output_path = save_experiment_result(result, output_dir=output_dir)
    summary = describe_experiment_result(result, metric=metric)
    return output_path, f"{summary} -> {output_path}"


# ---------------------------------------------------------------------------
# Backwards-compatibility helpers (used by tests / older notebook code)
# ---------------------------------------------------------------------------


def evaluate_with_validation_threshold(
    *,
    y_val_true: list[int] | Tensor | np.ndarray,
    y_val_score: list[float] | Tensor | np.ndarray,
    y_test_true: list[int] | Tensor | np.ndarray,
    y_test_score: list[float] | Tensor | np.ndarray,
) -> tuple[ClassificationMetrics, ClassificationMetrics]:
    """Tune a threshold on validation data, then score val and test."""

    threshold = choose_best_threshold(y_val_true, y_val_score)
    validation_metrics = compute_classification_metrics(
        y_val_true, y_val_score, threshold=threshold
    )
    test_metrics = compute_classification_metrics(
        y_test_true, y_test_score, threshold=threshold
    )
    return validation_metrics, test_metrics


__all__ = [
    "ExperimentResult",
    "build_experiment_result",
    "describe_experiment_result",
    "evaluate_with_heart_negatives",
    "evaluate_with_validation_threshold",
    "metrics_from_pos_neg_scores",
    "save_and_describe_experiment_result",
    "save_experiment_result",
    "score_pos_and_heart_negatives",
]
