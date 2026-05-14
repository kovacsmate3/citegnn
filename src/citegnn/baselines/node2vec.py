"""CPU-friendly node2vec baseline (HeaRT protocol)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch import Tensor, nn

from ..data.heart_split import HeartSplit
from ..evaluation import (
    ExperimentResult,
    build_experiment_result,
    evaluate_with_heart_negatives,
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
        for name in (
            "embedding_dim",
            "walk_length",
            "walks_per_node",
            "context_window",
            "negative_samples",
            "epochs",
            "batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "p", "q", "classifier_c"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


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


def _build_adjacency(split: HeartSplit) -> dict[int, list[int]]:
    adjacency_sets: dict[int, set[int]] = defaultdict(set)
    src = split.train_edge_index[0].tolist()
    dst = split.train_edge_index[1].tolist()
    for u, v in zip(src, dst):
        adjacency_sets[int(u)].add(int(v))
    return {
        node: sorted(adjacency_sets.get(node, set()))
        for node in range(split.num_nodes)
    }


def _edge_set(adjacency: dict[int, list[int]]) -> set[tuple[int, int]]:
    return {(s, t) for s, neighbors in adjacency.items() for t in neighbors}


def _node2vec_transition_weights(
    previous: int,
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
    split: HeartSplit,
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
    positive_pairs = _positive_pairs_from_walks(walks, context_window=config.context_window)
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


def _hadamard_features(edge_index: Tensor, embeddings: np.ndarray) -> np.ndarray:
    src = edge_index[0].detach().cpu().numpy().astype(np.int64)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64)
    return embeddings[src] * embeddings[dst]


def _sample_random_negative_edges(
    *,
    num_samples: int,
    num_nodes: int,
    forbidden: set[tuple[int, int]],
    rng: np.random.Generator,
) -> Tensor:
    samples: list[tuple[int, int]] = []
    cap = 10 * num_samples
    attempts = 0
    while len(samples) < num_samples and attempts < cap:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u == v:
            attempts += 1
            continue
        key = (u, v) if u < v else (v, u)
        if key in forbidden:
            attempts += 1
            continue
        samples.append((u, v))
        attempts += 1
    while len(samples) < num_samples:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u != v:
            samples.append((u, v))
    arr = np.asarray(samples, dtype=np.int64).T
    return torch.from_numpy(arr).contiguous()


def run_node2vec_baseline(
    split: HeartSplit,
    *,
    config: Node2VecConfig | None = None,
    seed: int | None = None,
) -> ExperimentResult:
    """Train node2vec embeddings + logistic edge classifier; eval with HeaRT."""

    resolved_config = config or Node2VecConfig()
    resolved_config.validate()
    run_seed = split.seed if seed is None else seed
    seed_everything(run_seed)

    embeddings = _train_node2vec_embeddings(split, resolved_config, seed=run_seed)

    rng = np.random.default_rng(run_seed)
    forbidden_train: set[tuple[int, int]] = set()
    src = split.train_pos_edge_index[0].tolist()
    dst = split.train_pos_edge_index[1].tolist()
    for u, v in zip(src, dst):
        a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        forbidden_train.add((a, b))

    train_neg = _sample_random_negative_edges(
        num_samples=int(split.train_pos_edge_index.size(1)),
        num_nodes=split.num_nodes,
        forbidden=forbidden_train,
        rng=rng,
    )

    pos_features = _hadamard_features(split.train_pos_edge_index, embeddings)
    neg_features = _hadamard_features(train_neg, embeddings)
    x_train = np.vstack([pos_features, neg_features]).astype(np.float32)
    y_train = np.concatenate(
        [
            np.ones(pos_features.shape[0], dtype=np.int64),
            np.zeros(neg_features.shape[0], dtype=np.int64),
        ]
    )

    classifier = LogisticRegression(
        C=resolved_config.classifier_c,
        max_iter=1000,
        random_state=run_seed,
    )
    classifier.fit(x_train, y_train)

    def score_fn(edges: Tensor) -> Tensor:
        feats = _hadamard_features(edges, embeddings).astype(np.float32)
        probs = classifier.predict_proba(feats)[:, 1]
        return torch.from_numpy(probs.astype(np.float64))

    validation, test = evaluate_with_heart_negatives(
        score_fn=score_fn,
        val_pos_edge_index=split.val_pos_edge_index,
        val_neg_nodes=split.val_neg_nodes,
        test_pos_edge_index=split.test_pos_edge_index,
        test_neg_nodes=split.test_neg_nodes,
    )

    params: dict[str, Any] = resolved_config.to_dict()
    params.update(
        {
            "split_seed": int(split.seed),
            "K": int(split.K),
            "train_graph_edges": split.train_graph_edge_count,
            "num_train_positives": int(split.num_train_positives),
        }
    )

    return build_experiment_result(
        dataset=split.dataset_name,
        method="node2vec",
        seed=run_seed,
        params=params,
        validation=validation,
        test=test,
    )


__all__ = [
    "Node2VecConfig",
    "run_node2vec_baseline",
]
