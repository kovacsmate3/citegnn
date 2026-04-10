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


def choose_best_threshold(
    y_true: np.ndarray | list[int] | Tensor,
    y_score: np.ndarray | list[float] | Tensor,
) -> float:
    """Select the score threshold that maximizes F1 on validation data."""

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
