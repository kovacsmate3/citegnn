"""Ranking metrics for HeaRT-style link prediction evaluation.

For each positive sample we are given ``K`` hard negatives sampled by HeaRT.
The model produces one score per candidate edge. The positive is then ranked
against its ``K`` negatives using the OGB-style optimistic rank
(``rank = (#negatives strictly greater than positive) + 1``).

We report:

* MRR -- mean of ``1 / rank`` across positives.
* Hits@k -- fraction of positives ranked in the top ``k``.

These metrics are the primary evaluation signal under the HeaRT protocol,
following Li et al. (NeurIPS 2023) "Evaluating Graph Neural Networks for
Link Prediction: Current Pitfalls and New Benchmarking".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

DEFAULT_HITS_KS: tuple[int, ...] = (1, 3, 10, 100)


@dataclass(slots=True)
class RankingMetrics:
    """HeaRT-aligned ranking metrics for one evaluation set."""

    mrr: float
    hits_at: dict[int, float] = field(default_factory=dict)
    num_positives: int = 0
    num_negatives_per_positive: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Serialize the metrics to a flat JSON-friendly dictionary."""

        payload: dict[str, float | int] = {
            "mrr": float(self.mrr),
            "num_positives": int(self.num_positives),
            "num_negatives_per_positive": int(self.num_negatives_per_positive),
        }
        for k, value in sorted(self.hits_at.items()):
            payload[f"hits_at_{k}"] = float(value)
        return payload


def _rank_positive_against_negatives(
    pos_scores: Tensor,
    neg_scores: Tensor,
) -> Tensor:
    """Compute one tie-aware rank per positive sample.

    The rank uses the *average* convention (matching ``scipy.stats.rankdata``
    with ``method='average'``)::

        rank = 0.5 * (#{neg > pos} + #{neg >= pos}) + 1

    With all-equal scores this yields ``rank = K/2 + 1`` (worst-case is
    ``K + 1`` and best-case is ``1``). The previous optimistic convention
    rewarded models that collapsed to a constant score by giving them
    ``rank = 1`` for every positive.
    """

    if pos_scores.dim() != 1:
        raise ValueError("pos_scores must be one-dimensional")
    if neg_scores.dim() != 2:
        raise ValueError("neg_scores must be two-dimensional [P, K]")
    if pos_scores.size(0) != neg_scores.size(0):
        raise ValueError("pos_scores and neg_scores must have the same number of positives")

    pos = pos_scores.to(dtype=torch.float64).unsqueeze(1)
    neg = neg_scores.to(dtype=torch.float64)
    strictly_greater = (neg > pos).sum(dim=1).to(dtype=torch.float64)
    geq = (neg >= pos).sum(dim=1).to(dtype=torch.float64)
    return 0.5 * (strictly_greater + geq) + 1.0


def compute_ranking_metrics(
    pos_scores: Tensor,
    neg_scores: Tensor,
    *,
    hits_ks: Sequence[int] = DEFAULT_HITS_KS,
) -> RankingMetrics:
    """Compute MRR and Hits@k for one evaluation set.

    Parameters
    ----------
    pos_scores:
        ``[P]`` tensor of scores for the ``P`` positive samples.
    neg_scores:
        ``[P, K]`` tensor of scores for the ``K`` HeaRT negatives associated
        with each positive sample.
    hits_ks:
        Sequence of ``k`` values to report Hits@k for. ``k`` may exceed ``K``;
        in that case Hits@k is trivially ``1``.

    Returns
    -------
    RankingMetrics
        Computed MRR, Hits@k, and bookkeeping counts.
    """

    if pos_scores.numel() == 0:
        raise ValueError("ranking metric computation requires at least one positive sample")
    if any(k <= 0 for k in hits_ks):
        raise ValueError("hits_ks values must be positive")

    ranks = _rank_positive_against_negatives(pos_scores, neg_scores)
    mrr = (1.0 / ranks).mean().item()
    # For Hits@k a tie at the boundary is counted as a half-hit (matches the
    # average-rank convention).
    hits_at: dict[int, float] = {}
    for k in hits_ks:
        hits_at[int(k)] = float((ranks <= float(k)).to(dtype=torch.float64).mean().item())

    return RankingMetrics(
        mrr=float(mrr),
        hits_at=hits_at,
        num_positives=int(pos_scores.numel()),
        num_negatives_per_positive=int(neg_scores.size(1)),
    )


def aggregate_ranking_metrics_dict(metrics: RankingMetrics) -> dict[str, Any]:
    """Public helper used when building ExperimentResult payloads."""

    return metrics.to_dict()


__all__ = [
    "DEFAULT_HITS_KS",
    "RankingMetrics",
    "compute_ranking_metrics",
    "aggregate_ranking_metrics_dict",
]
