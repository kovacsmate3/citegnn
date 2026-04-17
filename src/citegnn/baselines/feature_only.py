"""Feature-only pairwise baseline for link prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data.loaders import GraphData, load_citation_graph
from ..data.splits import LinkPredictionSplit
from ..evaluation import (
    ExperimentResult,
    build_experiment_result,
    evaluate_with_validation_threshold,
)
from ..utils.seed import seed_everything

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _default_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class PairwiseFeatureDataset:
    """Pairwise endpoint features derived from one link-prediction split."""

    dataset_name: str
    feature_mode: str
    train_features: Tensor
    train_labels: Tensor
    val_features: Tensor
    val_labels: Tensor
    test_features: Tensor
    test_labels: Tensor

    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of one pairwise feature vector."""

        return int(self.train_features.size(1))

    def to(self, device: torch.device) -> PairwiseFeatureDataset:
        """Move every tensor field in the dataset to one device."""

        return PairwiseFeatureDataset(
            dataset_name=self.dataset_name,
            feature_mode=self.feature_mode,
            train_features=self.train_features.to(device),
            train_labels=self.train_labels.to(device),
            val_features=self.val_features.to(device),
            val_labels=self.val_labels.to(device),
            test_features=self.test_features.to(device),
            test_labels=self.test_labels.to(device),
        )


@dataclass(slots=True)
class FeatureOnlyBaselineConfig:
    """Configuration for the feature-only pairwise MLP baseline."""

    feature_mode: str = "hadamard_abs_diff"
    hidden_dim: int = 256
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10
    batch_size: int = 1024
    show_progress: bool = True

    def validate(self) -> None:
        valid_feature_modes = {"concat", "hadamard", "abs_diff", "hadamard_abs_diff"}
        if self.feature_mode not in valid_feature_modes:
            raise ValueError(
                f"feature_mode must be one of {sorted(valid_feature_modes)}, got {self.feature_mode!r}"
            )
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "feature_mode": self.feature_mode,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "show_progress": self.show_progress,
        }


class _PairwiseMLP(nn.Module):
    """Small MLP that scores endpoint-feature pairs."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


def _edge_features(node_features: Tensor, edge_index: Tensor, *, feature_mode: str) -> Tensor:
    """Build pairwise endpoint features without using graph neighborhoods."""

    source_features = node_features[edge_index[0]]
    target_features = node_features[edge_index[1]]

    if feature_mode == "concat":
        return torch.cat([source_features, target_features], dim=1)
    if feature_mode == "hadamard":
        return source_features * target_features
    if feature_mode == "abs_diff":
        return torch.abs(source_features - target_features)
    if feature_mode == "hadamard_abs_diff":
        return torch.cat(
            [
                source_features * target_features,
                torch.abs(source_features - target_features),
            ],
            dim=1,
        )
    raise ValueError(f"unsupported feature_mode: {feature_mode}")


def _labeled_edge_features(
    node_features: Tensor,
    pos_edge_index: Tensor,
    neg_edge_index: Tensor,
    *,
    feature_mode: str,
) -> tuple[Tensor, Tensor]:
    """Combine positive and negative endpoint pairs into one labeled matrix."""

    pos_features = _edge_features(node_features, pos_edge_index, feature_mode=feature_mode)
    neg_features = _edge_features(node_features, neg_edge_index, feature_mode=feature_mode)
    features = torch.cat([pos_features, neg_features], dim=0)
    labels = torch.cat(
        [
            torch.ones(pos_features.size(0), dtype=torch.float32, device=pos_features.device),
            torch.zeros(neg_features.size(0), dtype=torch.float32, device=neg_features.device),
        ]
    )
    return features, labels


def build_pairwise_feature_dataset(
    split: LinkPredictionSplit,
    graph: GraphData,
    *,
    feature_mode: str = "hadamard_abs_diff",
) -> PairwiseFeatureDataset:
    """Build train/validation/test matrices from raw node attributes."""

    train_features, train_labels = _labeled_edge_features(
        graph.x,
        split.train_pos_edge_index,
        split.train_neg_edge_index,
        feature_mode=feature_mode,
    )
    val_features, val_labels = _labeled_edge_features(
        graph.x,
        split.val_pos_edge_index,
        split.val_neg_edge_index,
        feature_mode=feature_mode,
    )
    test_features, test_labels = _labeled_edge_features(
        graph.x,
        split.test_pos_edge_index,
        split.test_neg_edge_index,
        feature_mode=feature_mode,
    )
    return PairwiseFeatureDataset(
        dataset_name=split.dataset_name,
        feature_mode=feature_mode,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
    )


def _predict_scores(model: _PairwiseMLP, features: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Run one scoring pass and keep labels plus probabilities on the active device."""

    model.eval()
    with torch.no_grad():
        logits = model(features)
        scores = torch.sigmoid(logits).detach().to(dtype=torch.float64, device=features.device)
    return labels.detach().to(dtype=torch.int64, device=features.device), scores


def run_feature_only_baseline(
    split: LinkPredictionSplit,
    *,
    config: FeatureOnlyBaselineConfig | None = None,
    seed: int | None = None,
    data_root: str = "data",
) -> ExperimentResult:
    """Train a pairwise MLP using only endpoint node attributes."""

    resolved_config = config or FeatureOnlyBaselineConfig()
    resolved_config.validate()
    run_seed = split.seed if seed is None else seed
    seed_everything(run_seed)
    device = _default_device()

    graph = load_citation_graph(split.dataset_name, root=data_root, remove_self_loops=True)
    dataset = build_pairwise_feature_dataset(
        split,
        graph,
        feature_mode=resolved_config.feature_mode,
    ).to(device)

    model = _PairwiseMLP(
        input_dim=dataset.feature_dim,
        hidden_dim=resolved_config.hidden_dim,
        dropout=resolved_config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=resolved_config.learning_rate,
        weight_decay=resolved_config.weight_decay,
    )

    best_validation = None
    best_test = None
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0

    epoch_iterator = range(1, resolved_config.epochs + 1)
    progress_bar = None
    if resolved_config.show_progress and tqdm is not None:
        progress_bar = tqdm(
            epoch_iterator,
            desc=f"{split.dataset_name} feature_only_mlp",
            unit="epoch",
        )
        epoch_iterator = progress_bar

    for epoch in epoch_iterator:
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        permutation = torch.randperm(dataset.train_features.size(0), device=device)
        for batch_start in range(0, permutation.size(0), resolved_config.batch_size):
            batch_indices = permutation[batch_start : batch_start + resolved_config.batch_size]
            batch_features = dataset.train_features[batch_indices]
            batch_labels = dataset.train_labels[batch_indices]
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            batch_count += 1

        y_val_true, y_val_score = _predict_scores(model, dataset.val_features, dataset.val_labels)
        y_test_true, y_test_score = _predict_scores(model, dataset.test_features, dataset.test_labels)
        validation_metrics, test_metrics = evaluate_with_validation_threshold(
            y_val_true=y_val_true,
            y_val_score=y_val_score,
            y_test_true=y_test_true,
            y_test_score=y_test_score,
        )

        if progress_bar is not None:
            progress_bar.set_postfix(
                loss=f"{epoch_loss / max(batch_count, 1):.4f}",
                val_auc=f"{validation_metrics.roc_auc:.4f}",
                best_auc=f"{best_score if best_score > float('-inf') else validation_metrics.roc_auc:.4f}",
            )

        if validation_metrics.roc_auc > best_score:
            best_score = validation_metrics.roc_auc
            best_validation = validation_metrics
            best_test = test_metrics
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= resolved_config.patience:
                break

    if progress_bar is not None:
        progress_bar.close()

    if best_validation is None or best_test is None:
        raise RuntimeError("training did not produce any validation metrics")

    params: dict[str, Any] = resolved_config.to_dict()
    params.update(
        {
            "device": str(device),
            "split_seed": int(split.seed),
            "train_supervision_edges": split.train_supervision_edge_count,
            "pairwise_feature_dim": dataset.feature_dim,
            "best_epoch": int(best_epoch),
        }
    )
    return build_experiment_result(
        dataset=split.dataset_name,
        method="feature_only_mlp",
        seed=run_seed,
        params=params,
        validation_metrics=best_validation,
        test_metrics=best_test,
    )
