"""CPU-friendly node2vec baseline for link prediction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch import Tensor, nn

from ..data.splits import LinkPredictionSplit
from ..evaluation import (
    ExperimentResult,
    build_experiment_result,
    evaluate_with_validation_threshold,
)
from ..utils.seed import seed_everything


@dataclass(slots=True)
class Node2VecConfig:
    """Configuration for the node2vec baseline."""

    embedding_dim: int = 64
    walk_length: int = 20
    walks_per_node: int = 10
    context_window: int = 5
    negative_samples: int = 5
    epochs: int = 3
    batch_size: int = 512
    learning_rate: float = 0.025
    p: float = 1.0
    q: float = 1.0
    classifier_c: float = 1.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "embedding_dim": self.embedding_dim,
            "walk_length": self.walk_length,
            "walks_per_node": self.walks_per_node,
            "context_window": self.context_window,
            "negative_samples": self.negative_samples,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "p": self.p,
            "q": self.q,
            "classifier_c": self.classifier_c,
        }

    def validate(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.walk_length <= 0:
            raise ValueError("walk_length must be positive")
        if self.walks_per_node <= 0:
            raise ValueError("walks_per_node must be positive")
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        if self.negative_samples <= 0:
            raise ValueError("negative_samples must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.p <= 0:
            raise ValueError("p must be positive")
        if self.q <= 0:
            raise ValueError("q must be positive")
        if self.classifier_c <= 0:
            raise ValueError("classifier_c must be positive")


class _SkipGramModel(nn.Module):
    """Small skip-gram model with separate source and context embeddings."""

    def __init__(self, num_nodes: int, embedding_dim: int) -> None:
        super().__init__()
        self.source_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.context_embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.source_embedding.weight)
        nn.init.xavier_uniform_(self.context_embedding.weight)

    def positive_loss(self, sources: Tensor, contexts: Tensor) -> Tensor:
        src = self.source_embedding(sources)
        ctx = self.context_embedding(contexts)
        logits = (src * ctx).sum(dim=1)
        return F.softplus(-logits).mean()

    def negative_loss(self, sources: Tensor, negatives: Tensor) -> Tensor:
        src = self.source_embedding(sources)
        neg = self.context_embedding(negatives)
        logits = (src.unsqueeze(1) * neg).sum(dim=2)
        return F.softplus(logits).mean()

    def embeddings(self) -> np.ndarray:
        weight = self.source_embedding.weight.detach().cpu().numpy()
        return np.asarray(weight, dtype=np.float32)


def _edge_columns(edge_index: Tensor) -> list[tuple[int, int]]:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    return [(int(source), int(target)) for source, target in edge_index.t().tolist()]


def _build_adjacency(split: LinkPredictionSplit) -> dict[int, list[int]]:
    adjacency_sets: dict[int, set[int]] = defaultdict(set)
    for source, target in _edge_columns(split.train_graph_edge_index):
        adjacency_sets[source].add(target)
    num_nodes = int(split.train_mask.numel())
    return {node: sorted(adjacency_sets.get(node, set())) for node in range(num_nodes)}


def _edge_set(adjacency: dict[int, list[int]]) -> set[tuple[int, int]]:
    return {(source, target) for source, neighbors in adjacency.items() for target in neighbors}


def _node2vec_transition_weights(
    previous: int,
    current: int,
    neighbors: list[int],
    edge_pairs: set[tuple[int, int]],
    *,
    p: float,
    q: float,
) -> np.ndarray:
    weights: list[float] = []
    for neighbor in neighbors:
        if neighbor == previous:
            weights.append(1.0 / p)
        elif (previous, neighbor) in edge_pairs or (neighbor, previous) in edge_pairs:
            weights.append(1.0)
        else:
            weights.append(1.0 / q)
    result = np.asarray(weights, dtype=np.float64)
    return result / result.sum()


def _generate_random_walks(
    adjacency: dict[int, list[int]],
    *,
    walk_length: int,
    walks_per_node: int,
    p: float,
    q: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    walks: list[list[int]] = []
    nodes = list(adjacency.keys())
    edge_pairs = _edge_set(adjacency)

    for _ in range(walks_per_node):
        shuffled_nodes = list(nodes)
        rng.shuffle(shuffled_nodes)
        for start in shuffled_nodes:
            neighbors = adjacency[start]
            if not neighbors:
                walks.append([start])
                continue

            walk = [start]
            current = start
            previous: int | None = None

            for _step in range(1, walk_length):
                current_neighbors = adjacency[current]
                if not current_neighbors:
                    break
                if previous is None:
                    next_node = int(rng.choice(current_neighbors))
                else:
                    probabilities = _node2vec_transition_weights(
                        previous,
                        current,
                        current_neighbors,
                        edge_pairs,
                        p=p,
                        q=q,
                    )
                    next_node = int(rng.choice(current_neighbors, p=probabilities))
                walk.append(next_node)
                previous, current = current, next_node

            walks.append(walk)

    return walks


def _positive_pairs_from_walks(
    walks: list[list[int]],
    *,
    context_window: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for walk in walks:
        for center_index, source in enumerate(walk):
            start = max(0, center_index - context_window)
            end = min(len(walk), center_index + context_window + 1)
            for context_index in range(start, end):
                if context_index == center_index:
                    continue
                pairs.append((source, walk[context_index]))
    return pairs


def _negative_sampling_distribution(adjacency: dict[int, list[int]]) -> np.ndarray:
    degrees = np.asarray(
        [max(len(adjacency[node]), 1) for node in range(len(adjacency))],
        dtype=np.float64,
    )
    weights = np.power(degrees, 0.75)
    return weights / weights.sum()


def _train_node2vec_embeddings(
    split: LinkPredictionSplit,
    config: Node2VecConfig,
    *,
    seed: int,
) -> np.ndarray:
    adjacency = _build_adjacency(split)
    rng = np.random.default_rng(seed)
    walks = _generate_random_walks(
        adjacency,
        walk_length=config.walk_length,
        walks_per_node=config.walks_per_node,
        p=config.p,
        q=config.q,
        rng=rng,
    )
    positive_pairs = _positive_pairs_from_walks(
        walks,
        context_window=config.context_window,
    )
    if not positive_pairs:
        raise ValueError("node2vec training produced no positive context pairs")

    model = _SkipGramModel(len(adjacency), config.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    negative_distribution = _negative_sampling_distribution(adjacency)

    pair_array = np.asarray(positive_pairs, dtype=np.int64)
    for _epoch in range(config.epochs):
        rng.shuffle(pair_array)
        for start in range(0, len(pair_array), config.batch_size):
            batch = pair_array[start : start + config.batch_size]
            sources = torch.as_tensor(batch[:, 0], dtype=torch.long)
            contexts = torch.as_tensor(batch[:, 1], dtype=torch.long)
            negatives = torch.as_tensor(
                rng.choice(
                    len(adjacency),
                    size=(len(batch), config.negative_samples),
                    p=negative_distribution,
                ),
                dtype=torch.long,
            )

            optimizer.zero_grad()
            loss = model.positive_loss(sources, contexts) + model.negative_loss(sources, negatives)
            loss.backward()
            optimizer.step()

    return model.embeddings()


def _edge_features(edge_index: Tensor, embeddings: np.ndarray) -> np.ndarray:
    sources = edge_index[0].detach().cpu().numpy().astype(np.int64)
    targets = edge_index[1].detach().cpu().numpy().astype(np.int64)
    return embeddings[sources] * embeddings[targets]


def _labeled_edge_features(
    pos_edge_index: Tensor,
    neg_edge_index: Tensor,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos_features = _edge_features(pos_edge_index, embeddings)
    neg_features = _edge_features(neg_edge_index, embeddings)
    features = np.vstack([pos_features, neg_features]).astype(np.float32)
    labels = np.concatenate(
        [
            np.ones(pos_features.shape[0], dtype=np.int64),
            np.zeros(neg_features.shape[0], dtype=np.int64),
        ]
    )
    return features, labels


def run_node2vec_baseline(
    split: LinkPredictionSplit,
    *,
    config: Node2VecConfig | None = None,
    seed: int | None = None,
) -> ExperimentResult:
    """Train node2vec on the train graph and evaluate a logistic edge classifier."""

    resolved_config = config or Node2VecConfig()
    resolved_config.validate()
    run_seed = split.seed if seed is None else seed
    seed_everything(run_seed)

    embeddings = _train_node2vec_embeddings(
        split,
        resolved_config,
        seed=run_seed,
    )

    x_train, y_train = _labeled_edge_features(
        split.train_pos_edge_index,
        split.train_neg_edge_index,
        embeddings,
    )
    x_val, y_val = _labeled_edge_features(
        split.val_pos_edge_index,
        split.val_neg_edge_index,
        embeddings,
    )
    x_test, y_test = _labeled_edge_features(
        split.test_pos_edge_index,
        split.test_neg_edge_index,
        embeddings,
    )

    classifier = LogisticRegression(
        C=resolved_config.classifier_c,
        max_iter=1000,
        random_state=run_seed,
    )
    classifier.fit(x_train, y_train)

    y_val_score = classifier.predict_proba(x_val)[:, 1]
    y_test_score = classifier.predict_proba(x_test)[:, 1]
    validation_metrics, test_metrics = evaluate_with_validation_threshold(
        y_val_true=y_val,
        y_val_score=y_val_score,
        y_test_true=y_test,
        y_test_score=y_test_score,
    )

    params: dict[str, Any] = resolved_config.to_dict()
    params.update(
        {
            "split_seed": int(split.seed),
            "train_graph_edges": split.train_graph_edge_count,
            "train_supervision_edges": split.train_supervision_edge_count,
        }
    )

    return build_experiment_result(
        dataset=split.dataset_name,
        method="node2vec",
        seed=run_seed,
        params=params,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )
