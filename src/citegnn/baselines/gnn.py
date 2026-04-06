"""Shared GNN baselines for link prediction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, SAGEConv

from ..data.loaders import load_citation_graph
from ..data.splits import LinkPredictionSplit
from ..evaluation import (
    ExperimentResult,
    build_experiment_result,
    evaluate_with_validation_threshold,
)
from ..utils.seed import seed_everything


class GNNMethod(StrEnum):
    """Supported GNN baseline variants."""

    GCN = "gcn"
    GRAPHSAGE = "graphsage"


@dataclass(slots=True)
class GNNBaselineConfig:
    """Configuration for the shared GNN link-prediction baselines."""

    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    decoder_hidden_dim: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 20

    def validate(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.decoder_hidden_dim <= 0:
            raise ValueError("decoder_hidden_dim must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
        }


class _GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, config: GNNBaselineConfig) -> None:
        super().__init__()
        dims = [in_dim] + [config.hidden_dim] * config.num_layers
        self.layers = nn.ModuleList(
            GCNConv(dims[index], dims[index + 1]) for index in range(config.num_layers)
        )
        self.dropout = float(config.dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        hidden = x
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, edge_index)
            if index < len(self.layers) - 1:
                hidden = F.relu(hidden)
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return hidden


class _GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, config: GNNBaselineConfig) -> None:
        super().__init__()
        dims = [in_dim] + [config.hidden_dim] * config.num_layers
        self.layers = nn.ModuleList(
            SAGEConv(dims[index], dims[index + 1]) for index in range(config.num_layers)
        )
        self.dropout = float(config.dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        hidden = x
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, edge_index)
            if index < len(self.layers) - 1:
                hidden = F.relu(hidden)
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return hidden


class _HadamardMLPDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: Tensor, edge_index: Tensor) -> Tensor:
        source_embeddings = node_embeddings[edge_index[0]]
        target_embeddings = node_embeddings[edge_index[1]]
        pair_embeddings = source_embeddings * target_embeddings
        return self.network(pair_embeddings).squeeze(-1)


class _LinkPredictionModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: _HadamardMLPDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def score_edges(self, node_embeddings: Tensor, edge_index: Tensor) -> Tensor:
        return self.decoder(node_embeddings, edge_index)


def _edge_labels(pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tensor:
    return torch.cat(
        [
            torch.ones(pos_edge_index.size(1), dtype=torch.float32),
            torch.zeros(neg_edge_index.size(1), dtype=torch.float32),
        ]
    )


def _combined_edge_index(pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tensor:
    return torch.cat([pos_edge_index, neg_edge_index], dim=1)


def _build_model(
    *,
    method: GNNMethod,
    input_dim: int,
    config: GNNBaselineConfig,
) -> _LinkPredictionModel:
    if method == GNNMethod.GCN:
        encoder: nn.Module = _GCNEncoder(input_dim, config)
    elif method == GNNMethod.GRAPHSAGE:
        encoder = _GraphSAGEEncoder(input_dim, config)
    else:
        raise ValueError(f"unsupported GNN method: {method}")

    decoder = _HadamardMLPDecoder(config.hidden_dim, config.decoder_hidden_dim)
    return _LinkPredictionModel(encoder, decoder)


def _predict_scores(
    model: _LinkPredictionModel,
    *,
    x: Tensor,
    message_passing_edge_index: Tensor,
    pos_edge_index: Tensor,
    neg_edge_index: Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode(x, message_passing_edge_index)
        edge_index = _combined_edge_index(pos_edge_index, neg_edge_index)
        logits = model.score_edges(node_embeddings, edge_index)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        labels = _edge_labels(pos_edge_index, neg_edge_index).detach().cpu().numpy()
    return labels.astype(np.int64), scores.astype(np.float64)


def run_gnn_baseline(
    split: LinkPredictionSplit,
    *,
    method: GNNMethod,
    config: GNNBaselineConfig | None = None,
    seed: int | None = None,
    data_root: str = "data",
) -> ExperimentResult:
    """Train one GNN baseline and evaluate it on the shared split artifact."""

    resolved_config = config or GNNBaselineConfig()
    resolved_config.validate()
    run_seed = split.seed if seed is None else seed
    seed_everything(run_seed)

    graph = load_citation_graph(
        split.dataset_name,
        root=data_root,
        undirected=split.is_undirected,
        remove_self_loops=True,
    )
    x = graph.x
    train_graph_edge_index = split.train_graph_edge_index
    train_edge_index = _combined_edge_index(split.train_pos_edge_index, split.train_neg_edge_index)
    train_labels = _edge_labels(split.train_pos_edge_index, split.train_neg_edge_index)

    model = _build_model(
        method=method,
        input_dim=graph.num_features,
        config=resolved_config,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=resolved_config.learning_rate,
        weight_decay=resolved_config.weight_decay,
    )

    best_state = deepcopy(model.state_dict())
    best_validation_metrics: dict[str, float] | None = None
    best_test_metrics: dict[str, float] | None = None
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, resolved_config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        node_embeddings = model.encode(x, train_graph_edge_index)
        logits = model.score_edges(node_embeddings, train_edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels)
        loss.backward()
        optimizer.step()

        y_val_true, y_val_score = _predict_scores(
            model,
            x=x,
            message_passing_edge_index=train_graph_edge_index,
            pos_edge_index=split.val_pos_edge_index,
            neg_edge_index=split.val_neg_edge_index,
        )
        y_test_true, y_test_score = _predict_scores(
            model,
            x=x,
            message_passing_edge_index=train_graph_edge_index,
            pos_edge_index=split.test_pos_edge_index,
            neg_edge_index=split.test_neg_edge_index,
        )
        validation_metrics, test_metrics = evaluate_with_validation_threshold(
            y_val_true=y_val_true,
            y_val_score=y_val_score,
            y_test_true=y_test_true,
            y_test_score=y_test_score,
        )

        if validation_metrics.roc_auc > best_score:
            best_score = validation_metrics.roc_auc
            best_state = deepcopy(model.state_dict())
            best_validation_metrics = validation_metrics.to_dict()
            best_test_metrics = test_metrics.to_dict()
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= resolved_config.patience:
                break

    model.load_state_dict(best_state)
    if best_validation_metrics is None or best_test_metrics is None:
        raise RuntimeError("training did not produce any validation metrics")

    params: dict[str, Any] = resolved_config.to_dict()
    params.update(
        {
            "split_seed": int(split.seed),
            "is_undirected": bool(split.is_undirected),
            "train_graph_edges": int(split.train_graph_edge_index.size(1)),
            "train_supervision_edges": int(
                split.train_pos_edge_index.size(1) + split.train_neg_edge_index.size(1)
            ),
            "best_epoch": int(best_epoch),
        }
    )

    return ExperimentResult(
        dataset=split.dataset_name,
        method=method.value,
        seed=run_seed,
        params=params,
        validation=best_validation_metrics,
        test=best_test_metrics,
    )
