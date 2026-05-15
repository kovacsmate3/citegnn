"""Feature-only pairwise baseline (HeaRT protocol)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data.heart_split import HeartSplit
from ..data.loaders import load_citation_graph
from ..evaluation import (
    ExperimentResult,
    build_experiment_result,
    evaluate_with_heart_negatives,
)
from ..utils.seed import seed_everything

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class FeatureOnlyBaselineConfig:
    """Configuration for the feature-only pairwise MLP baseline."""

    feature_mode: str = "hadamard"
    hidden_dim: int = 128
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10
    eval_every: int = 1
    train_negatives_per_positive: int = 1
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
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.train_negatives_per_positive <= 0:
            raise ValueError("train_negatives_per_positive must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "feature_mode": self.feature_mode,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "eval_every": self.eval_every,
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


def _edge_features(
    node_features: Tensor,
    edge_index: Tensor,
    *,
    feature_mode: str,
) -> Tensor:
    source = node_features[edge_index[0]]
    target = node_features[edge_index[1]]
    if feature_mode == "concat":
        return torch.cat([source, target], dim=1)
    if feature_mode == "hadamard":
        return source * target
    if feature_mode == "abs_diff":
        return torch.abs(source - target)
    if feature_mode == "hadamard_abs_diff":
        return torch.cat([source * target, torch.abs(source - target)], dim=1)
    raise ValueError(f"unsupported feature_mode: {feature_mode}")


def _pairwise_input_dim(node_feature_dim: int, feature_mode: str) -> int:
    if feature_mode == "concat":
        return 2 * node_feature_dim
    if feature_mode in {"hadamard", "abs_diff"}:
        return node_feature_dim
    if feature_mode == "hadamard_abs_diff":
        return 2 * node_feature_dim
    raise ValueError(f"unsupported feature_mode: {feature_mode}")


def _sample_random_negative_edges(
    *,
    num_samples: int,
    num_nodes: int,
    forbidden: set[tuple[int, int]],
    rng: torch.Generator,
    device: torch.device,
    max_attempts_per_sample: int = 10,
) -> Tensor:
    """Draw ``num_samples`` random non-edges that aren't in ``forbidden``."""

    samples: list[tuple[int, int]] = []
    attempts = 0
    cap = max_attempts_per_sample * num_samples
    while len(samples) < num_samples and attempts < cap:
        u = int(torch.randint(0, num_nodes, (1,), generator=rng, device=device).item())
        v = int(torch.randint(0, num_nodes, (1,), generator=rng, device=device).item())
        if u == v:
            attempts += 1
            continue
        key = (u, v) if u < v else (v, u)
        if key in forbidden:
            attempts += 1
            continue
        samples.append((u, v))
        attempts += 1
    if len(samples) < num_samples:
        # Fallback for tiny graphs: keep going without uniqueness.
        while len(samples) < num_samples:
            u = int(torch.randint(0, num_nodes, (1,), generator=rng, device=device).item())
            v = int(torch.randint(0, num_nodes, (1,), generator=rng, device=device).item())
            if u != v:
                samples.append((u, v))
    arr = torch.tensor(samples, dtype=torch.long, device=device).t().contiguous()
    return arr


def _train_positive_set(train_pos: Tensor) -> set[tuple[int, int]]:
    src = train_pos[0].tolist()
    dst = train_pos[1].tolist()
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in zip(src, dst)}


def run_feature_only_baseline(
    split: HeartSplit,
    *,
    config: FeatureOnlyBaselineConfig | None = None,
    seed: int | None = None,
    data_root: str = "data",
) -> ExperimentResult:
    """Train a pairwise MLP on endpoint attributes; evaluate with HeaRT."""

    resolved_config = config or FeatureOnlyBaselineConfig()
    resolved_config.validate()
    run_seed = split.seed if seed is None else seed
    seed_everything(run_seed)
    device = _default_device()

    graph = load_citation_graph(split.dataset_name, root=data_root, remove_self_loops=True)
    node_features = graph.x.to(device)
    train_pos_edges = split.train_pos_edge_index.to(device)
    forbidden = _train_positive_set(split.train_pos_edge_index)

    input_dim = _pairwise_input_dim(node_features.size(1), resolved_config.feature_mode)
    model = _PairwiseMLP(
        input_dim=input_dim,
        hidden_dim=resolved_config.hidden_dim,
        dropout=resolved_config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=resolved_config.learning_rate,
        weight_decay=resolved_config.weight_decay,
    )

    num_pos = int(train_pos_edges.size(1))
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(run_seed)

    @torch.no_grad()
    def _score_edges(edges: Tensor) -> Tensor:
        edges_dev = edges.to(device=device, dtype=torch.long)
        feats = _edge_features(
            node_features,
            edges_dev,
            feature_mode=resolved_config.feature_mode,
        )
        model.eval()
        return torch.sigmoid(model(feats)).cpu().to(dtype=torch.float64)

    best_validation: dict[str, float] | None = None
    best_test: dict[str, float] | None = None
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

    neg_per_pos = int(resolved_config.train_negatives_per_positive)
    pos_weight = torch.tensor(float(neg_per_pos), device=device)
    for epoch in epoch_iterator:
        model.train()
        # Fresh random training negatives per epoch (HeaRT keeps training
        # untouched -- only evaluation uses hard negatives).
        train_neg = _sample_random_negative_edges(
            num_samples=num_pos * neg_per_pos,
            num_nodes=split.num_nodes,
            forbidden=forbidden,
            rng=cpu_rng,
            device=torch.device("cpu"),
        ).to(device)

        all_edges = torch.cat([train_pos_edges, train_neg], dim=1)
        labels = torch.cat(
            [
                torch.ones(num_pos, dtype=torch.float32, device=device),
                torch.zeros(num_pos * neg_per_pos, dtype=torch.float32, device=device),
            ]
        )
        permutation = torch.randperm(all_edges.size(1), device=device)
        all_edges = all_edges[:, permutation]
        labels = labels[permutation]

        epoch_loss = 0.0
        batch_count = 0
        for batch_start in range(0, all_edges.size(1), resolved_config.batch_size):
            batch_edges = all_edges[:, batch_start : batch_start + resolved_config.batch_size]
            batch_labels = labels[batch_start : batch_start + resolved_config.batch_size]
            batch_features = _edge_features(
                node_features,
                batch_edges,
                feature_mode=resolved_config.feature_mode,
            )
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch_labels, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            batch_count += 1

        if epoch % resolved_config.eval_every != 0 and epoch != resolved_config.epochs:
            continue

        validation, test = evaluate_with_heart_negatives(
            score_fn=_score_edges,
            val_pos_edge_index=split.val_pos_edge_index,
            val_neg_nodes=split.val_neg_nodes,
            test_pos_edge_index=split.test_pos_edge_index,
            test_neg_nodes=split.test_neg_nodes,
        )

        if progress_bar is not None:
            progress_bar.set_postfix(
                loss=f"{epoch_loss / max(batch_count, 1):.4f}",
                val_mrr=f"{validation['mrr']:.4f}",
                best_mrr=f"{best_score if best_score > float('-inf') else validation['mrr']:.4f}",
            )

        if validation["mrr"] > best_score:
            best_score = validation["mrr"]
            best_validation = validation
            best_test = test
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
            "K": int(split.K),
            "train_positive_edges": int(num_pos),
            "pairwise_feature_dim": int(input_dim),
            "best_epoch": int(best_epoch),
        }
    )
    return build_experiment_result(
        dataset=split.dataset_name,
        method="feature_only_mlp",
        seed=run_seed,
        params=params,
        validation=best_validation,
        test=best_test,
    )


__all__ = [
    "FeatureOnlyBaselineConfig",
    "run_feature_only_baseline",
]
