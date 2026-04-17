"""Shared metric computation for link prediction experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from torch import Tensor


@dataclass(slots=True)
class ClassificationMetrics:
    """Evaluation metrics for binary link prediction."""

    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float
    threshold: float
    positive_rate: float

    def to_dict(self) -> dict[str, float]:
        """Convert the metrics into a JSON-friendly dictionary."""

        return asdict(self)


def _to_numpy(array: np.ndarray | list[float] | list[int] | Tensor) -> np.ndarray:
    """Convert common array-like inputs into a one-dimensional NumPy array."""

    if isinstance(array, np.ndarray):
        result = array
    else:
        try:
            import torch
        except ModuleNotFoundError:
            result = np.asarray(array)
        else:
            if isinstance(array, torch.Tensor):
                result = array.detach().cpu().numpy()
            else:
                result = np.asarray(array)

    return np.ravel(result)


def _to_tensor(
    array: np.ndarray | list[float] | list[int] | Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Convert common array-like inputs into a one-dimensional Torch tensor."""

    if isinstance(array, Tensor):
        result = array.detach()
    elif isinstance(array, np.ndarray):
        result = torch.from_numpy(array)
    else:
        result = torch.as_tensor(array)

    result = torch.ravel(result)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


def _binary_roc_auc_torch(labels: Tensor, scores: Tensor) -> float:
    """Compute tie-aware ROC-AUC for binary labels using average ranks."""

    order = torch.argsort(scores, stable=True)
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    positive_count = int(sorted_labels.sum().item())
    negative_count = int(sorted_labels.numel() - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("metric computation requires both positive and negative labels")

    ranks = torch.arange(1, sorted_labels.numel() + 1, device=scores.device, dtype=torch.float64)
    average_ranks = torch.empty_like(ranks)
    unique_scores, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    del unique_scores
    start = 0
    for count in counts.tolist():
        end = start + count
        average_rank = (ranks[start] + ranks[end - 1]) / 2.0
        average_ranks[start:end] = average_rank
        start = end

    positive_rank_sum = average_ranks[sorted_labels.to(torch.bool)].sum()
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc.item())


def _binary_average_precision_torch(labels: Tensor, scores: Tensor) -> float:
    """Compute tie-aware average precision for binary labels."""

    positive_count = int(labels.sum().item())
    if positive_count == 0:
        raise ValueError("metric computation requires both positive and negative labels")

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    cumulative_true_positives = torch.cumsum(sorted_labels, dim=0).to(torch.float64)
    cumulative_false_positives = (
        torch.cumsum(1 - sorted_labels, dim=0).to(torch.float64)
    )

    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    group_last_indices = torch.cumsum(counts, dim=0) - 1
    true_positives = cumulative_true_positives[group_last_indices]
    false_positives = cumulative_false_positives[group_last_indices]
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / float(positive_count)

    previous_recall = torch.cat(
        [torch.zeros(1, device=scores.device, dtype=torch.float64), recall[:-1]]
    )
    average_precision = torch.sum((recall - previous_recall) * precision)
    return float(average_precision.item())


def _classification_counts(labels: Tensor, predictions: Tensor) -> tuple[float, float, float]:
    """Return true positives, predicted positives, and actual positives."""

    labels_bool = labels.to(torch.bool)
    predictions_bool = predictions.to(torch.bool)
    true_positives = float((labels_bool & predictions_bool).sum().item())
    predicted_positives = float(predictions_bool.sum().item())
    actual_positives = float(labels_bool.sum().item())
    return true_positives, predicted_positives, actual_positives


def choose_best_threshold(
    y_true: np.ndarray | list[int] | Tensor,
    y_score: np.ndarray | list[float] | Tensor,
) -> float:
    """Select the score threshold that maximizes F1 on validation data."""

    if isinstance(y_true, Tensor) or isinstance(y_score, Tensor):
        label_tensor = _to_tensor(y_true, dtype=torch.int64)
        score_tensor = _to_tensor(y_score, dtype=torch.float64).to(device=label_tensor.device)
        if label_tensor.shape != score_tensor.shape:
            raise ValueError("y_true and y_score must have the same shape")
        if label_tensor.numel() == 0:
            raise ValueError("threshold selection requires at least one example")

        candidates = torch.unique(score_tensor)
        if candidates.numel() == 1:
            return float(candidates[0].item())

        best_threshold = float(candidates[0].item())
        best_f1 = -1.0

        for threshold in candidates:
            predictions = (score_tensor >= threshold).to(torch.int64)
            true_positives, predicted_positives, actual_positives = _classification_counts(
                label_tensor,
                predictions,
            )
            precision = (
                true_positives / predicted_positives if predicted_positives > 0.0 else 0.0
            )
            recall = true_positives / actual_positives if actual_positives > 0.0 else 0.0
            score = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0.0
                else 0.0
            )
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold.item())

        return best_threshold

    labels = _to_numpy(y_true).astype(int)
    scores = _to_numpy(y_score).astype(float)

    if labels.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if labels.size == 0:
        raise ValueError("threshold selection requires at least one example")

    candidates = np.unique(scores)
    if candidates.size == 1:
        return float(candidates[0])

    best_threshold = float(candidates[0])
    best_f1 = -1.0

    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        score = f1_score(labels, predictions, zero_division=0.0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def compute_classification_metrics(
    y_true: np.ndarray | list[int] | Tensor,
    y_score: np.ndarray | list[float] | Tensor,
    *,
    threshold: float,
) -> ClassificationMetrics:
    """Compute ranking and thresholded classification metrics."""

    if isinstance(y_true, Tensor) or isinstance(y_score, Tensor):
        labels = _to_tensor(y_true, dtype=torch.int64)
        scores = _to_tensor(y_score, dtype=torch.float64).to(device=labels.device)

        if labels.shape != scores.shape:
            raise ValueError("y_true and y_score must have the same shape")
        if labels.numel() == 0:
            raise ValueError("metric computation requires at least one example")
        if torch.unique(labels).numel() < 2:
            raise ValueError("metric computation requires both positive and negative labels")

        predictions = (scores >= threshold).to(torch.int64)
        true_positives, predicted_positives, actual_positives = _classification_counts(
            labels,
            predictions,
        )
        precision = true_positives / predicted_positives if predicted_positives > 0.0 else 0.0
        recall = true_positives / actual_positives if actual_positives > 0.0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )

        return ClassificationMetrics(
            roc_auc=_binary_roc_auc_torch(labels, scores),
            average_precision=_binary_average_precision_torch(labels, scores),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            threshold=float(threshold),
            positive_rate=float(predictions.to(torch.float64).mean().item()),
        )

    labels = _to_numpy(y_true).astype(int)
    scores = _to_numpy(y_score).astype(float)

    if labels.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if labels.size == 0:
        raise ValueError("metric computation requires at least one example")
    if np.unique(labels).size < 2:
        raise ValueError("metric computation requires both positive and negative labels")

    predictions = (scores >= threshold).astype(int)

    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(labels, scores)),
        average_precision=float(average_precision_score(labels, scores)),
        precision=float(precision_score(labels, predictions, zero_division=0.0)),
        recall=float(recall_score(labels, predictions, zero_division=0.0)),
        f1=float(f1_score(labels, predictions, zero_division=0.0)),
        threshold=float(threshold),
        positive_rate=float(predictions.mean()),
    )
