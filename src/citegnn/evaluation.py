"""Reusable evaluation helpers and result schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor

from .utils.io import write_json
from .utils.metrics import (
    ClassificationMetrics,
    choose_best_threshold,
    compute_classification_metrics,
)


@dataclass(slots=True)
class ExperimentResult:
    """Serializable experiment result for one dataset-method-seed run."""

    dataset: str
    method: str
    seed: int
    params: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, float] = field(default_factory=dict)
    test: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert this result into a JSON-friendly dictionary."""

        return asdict(self)


def describe_experiment_result(
    result: ExperimentResult,
    *,
    include_validation_auc: bool = True,
    include_test_average_precision: bool = False,
) -> str:
    """Build a compact CLI summary line for one experiment result."""

    metrics: list[str] = []
    if include_validation_auc:
        metrics.append(f"val_auc={result.validation['roc_auc']:.4f}")
    metrics.append(f"test_auc={result.test['roc_auc']:.4f}")
    if include_test_average_precision:
        metrics.append(f"test_ap={result.test['average_precision']:.4f}")
    return f"{result.dataset} {result.method}: {' '.join(metrics)}"


def evaluate_with_validation_threshold(
    *,
    y_val_true: list[int] | Tensor | np.ndarray,
    y_val_score: list[float] | Tensor | np.ndarray,
    y_test_true: list[int] | Tensor | np.ndarray,
    y_test_score: list[float] | Tensor | np.ndarray,
) -> tuple[ClassificationMetrics, ClassificationMetrics]:
    """Tune a threshold on validation data, then score validation and test."""

    threshold = choose_best_threshold(y_val_true, y_val_score)
    validation_metrics = compute_classification_metrics(
        y_val_true,
        y_val_score,
        threshold=threshold,
    )
    test_metrics = compute_classification_metrics(
        y_test_true,
        y_test_score,
        threshold=threshold,
    )
    return validation_metrics, test_metrics


def build_experiment_result(
    *,
    dataset: str,
    method: str,
    seed: int,
    params: dict[str, Any],
    validation_metrics: ClassificationMetrics,
    test_metrics: ClassificationMetrics,
) -> ExperimentResult:
    """Create the standard experiment result payload."""

    return ExperimentResult(
        dataset=dataset,
        method=method,
        seed=seed,
        params=params,
        validation=validation_metrics.to_dict(),
        test=test_metrics.to_dict(),
    )


def save_experiment_result(
    result: ExperimentResult,
    *,
    output_dir: str | Path = "results",
) -> Path:
    """Persist one experiment result as JSON."""

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
    include_validation_auc: bool = True,
    include_test_average_precision: bool = False,
) -> tuple[Path, str]:
    """Persist one result and return the matching CLI summary line."""

    output_path = save_experiment_result(result, output_dir=output_dir)
    summary = describe_experiment_result(
        result,
        include_validation_auc=include_validation_auc,
        include_test_average_precision=include_test_average_precision,
    )
    return output_path, f"{summary} -> {output_path}"
