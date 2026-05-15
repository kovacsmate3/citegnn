"""Shared GNN baselines for link prediction (HeaRT protocol).

Encoders supported:

* ``gcn`` -- :class:`torch_geometric.nn.GCNConv` stack.
* ``graphsage`` -- :class:`torch_geometric.nn.SAGEConv` stack.

Decoder is a small MLP applied to the Hadamard product of the endpoint
embeddings.

The optional structural augmentation concatenates log-degree and the
top-k Laplacian eigenvectors of the training graph to the raw node
features, addressing RQ3 of the project proposal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from ..data.heart_split import HeartSplit
from ..data.loaders import load_citation_graph
from ..evaluation import (
    ExperimentResult,
    build_experiment_result,
    evaluate_with_heart_negatives,
)
from ..utils.seed import seed_everything
from .augmentation import augment_features, compute_structural_features


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GNNMethod(StrEnum):
    """Supported GNN baseline variants."""

    GCN = "gcn"
    GRAPHSAGE = "graphsage"
    GAT = "gat"


@dataclass(slots=True)
class GNNBaselineConfig:
    """Configuration for the shared GNN link-prediction baselines."""

    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    decoder_hidden_dim: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 20
    eval_every: int = 1
    decoder: str = "hadamard_mlp"  # one of {"hadamard_mlp", "dot"}
    train_negatives_per_positive: int = 1
    # Training mode: "fullbatch" (one optimizer step per epoch on the
    # whole train set) or "minibatch_dropedge" (HeaRT-style: many
    # optimizer steps per epoch with batch_size supervision edges per
    # step, and the supervision edges of each batch are removed from
    # the message-passing graph for that step).
    train_mode: str = "fullbatch"
    train_batch_size: int = 1024
    use_structural_features: bool = False
    num_laplacian_eigvecs: int = 10
    # GAT-specific: number of attention heads per layer. Hidden activations
    # within the encoder always end up at ``hidden_dim`` regardless of how
    # many heads are used (we average across heads on the last layer and
    # split ``hidden_dim`` across heads on intermediate ones).
    gat_heads: int = 4
    gat_attention_dropout: float = 0.1

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
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.decoder not in {"hadamard_mlp", "dot"}:
            raise ValueError("decoder must be 'hadamard_mlp' or 'dot'")
        if self.train_negatives_per_positive <= 0:
            raise ValueError("train_negatives_per_positive must be positive")
        if self.train_mode not in {"fullbatch", "minibatch_dropedge"}:
            raise ValueError("train_mode must be 'fullbatch' or 'minibatch_dropedge'")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if self.num_laplacian_eigvecs < 0:
            raise ValueError("num_laplacian_eigvecs must be non-negative")
        if self.gat_heads <= 0:
            raise ValueError("gat_heads must be positive")
        if not 0.0 <= self.gat_attention_dropout < 1.0:
            raise ValueError("gat_attention_dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "eval_every": self.eval_every,
            "decoder": self.decoder,
            "train_negatives_per_positive": self.train_negatives_per_positive,
            "train_mode": self.train_mode,
            "train_batch_size": self.train_batch_size,
            "use_structural_features": self.use_structural_features,
            "num_laplacian_eigvecs": self.num_laplacian_eigvecs,
            "gat_heads": self.gat_heads,
            "gat_attention_dropout": self.gat_attention_dropout,
        }


class _StackedEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        config: GNNBaselineConfig,
        layer_factory: Callable[[int, int], nn.Module],
    ) -> None:
        super().__init__()
        dims = [in_dim] + [config.hidden_dim] * config.num_layers
        self.layers = nn.ModuleList(
            layer_factory(dims[index], dims[index + 1])
            for index in range(config.num_layers)
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


class _GATEncoder(nn.Module):
    """Multi-layer GAT encoder.

    Hidden layers use ``concat=True`` with ``hidden_dim // heads`` channels
    so the output is always ``hidden_dim``. The final layer uses
    ``concat=False`` (head averaging) so it also returns ``hidden_dim``.
    """

    def __init__(self, in_dim: int, config: GNNBaselineConfig) -> None:
        super().__init__()
        heads = int(config.gat_heads)
        hidden_dim = int(config.hidden_dim)
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by gat_heads ({heads})"
            )
        per_head = hidden_dim // heads
        attn_dropout = float(config.gat_attention_dropout)
        layers: list[nn.Module] = []
        prev = in_dim
        for index in range(int(config.num_layers)):
            is_last = index == int(config.num_layers) - 1
            if is_last:
                layers.append(
                    GATConv(
                        prev,
                        hidden_dim,
                        heads=heads,
                        concat=False,
                        dropout=attn_dropout,
                    )
                )
            else:
                layers.append(
                    GATConv(
                        prev,
                        per_head,
                        heads=heads,
                        concat=True,
                        dropout=attn_dropout,
                    )
                )
            prev = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.dropout = float(config.dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        hidden = x
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, edge_index)
            if index < len(self.layers) - 1:
                hidden = F.elu(hidden)
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
        source = node_embeddings[edge_index[0]]
        target = node_embeddings[edge_index[1]]
        pair = source * target
        return self.network(pair).squeeze(-1)


class _DotProductDecoder(nn.Module):
    """Inner-product decoder used by VGAE-style link prediction baselines."""

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        # No learned parameters; hidden_dim is accepted for interface parity.
        del embedding_dim, hidden_dim

    def forward(self, node_embeddings: Tensor, edge_index: Tensor) -> Tensor:
        source = node_embeddings[edge_index[0]]
        target = node_embeddings[edge_index[1]]
        return (source * target).sum(dim=-1)


class _LinkPredictionModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: _HadamardMLPDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def score_edges(self, node_embeddings: Tensor, edge_index: Tensor) -> Tensor:
        return self.decoder(node_embeddings, edge_index)


def _build_model(
    *,
    method: GNNMethod,
    input_dim: int,
    config: GNNBaselineConfig,
) -> _LinkPredictionModel:
    encoder: nn.Module
    if method == GNNMethod.GCN:
        encoder = _StackedEncoder(input_dim, config, GCNConv)
    elif method == GNNMethod.GRAPHSAGE:
        encoder = _StackedEncoder(input_dim, config, SAGEConv)
    elif method == GNNMethod.GAT:
        encoder = _GATEncoder(input_dim, config)
    else:
        raise ValueError(f"unsupported GNN method: {method}")
    if config.decoder == "dot":
        decoder: nn.Module = _DotProductDecoder(config.hidden_dim, config.decoder_hidden_dim)
    else:
        decoder = _HadamardMLPDecoder(config.hidden_dim, config.decoder_hidden_dim)
    return _LinkPredictionModel(encoder, decoder)


def _sample_random_negative_edges(
    *,
    num_samples: int,
    num_nodes: int,
    forbidden: set[tuple[int, int]],
    rng: torch.Generator,
) -> Tensor:
    samples: list[tuple[int, int]] = []
    cap = 10 * num_samples
    attempts = 0
    while len(samples) < num_samples and attempts < cap:
        u = int(torch.randint(0, num_nodes, (1,), generator=rng).item())
        v = int(torch.randint(0, num_nodes, (1,), generator=rng).item())
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
        u = int(torch.randint(0, num_nodes, (1,), generator=rng).item())
        v = int(torch.randint(0, num_nodes, (1,), generator=rng).item())
        if u != v:
            samples.append((u, v))
    arr = torch.tensor(samples, dtype=torch.long).t().contiguous()
    return arr


def _train_positive_set(train_pos: Tensor) -> set[tuple[int, int]]:
    src = train_pos[0].tolist()
    dst = train_pos[1].tolist()
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in zip(src, dst)}


def run_gnn_baseline(
    split: HeartSplit,
    *,
    method: GNNMethod,
    config: GNNBaselineConfig | None = None,
    seed: int | None = None,
    data_root: str = "data",
    method_name_override: str | None = None,
) -> ExperimentResult:
    """Train one GNN baseline and evaluate it under HeaRT."""

    resolved_config = config or GNNBaselineConfig()
    resolved_config.validate()
    run_seed = split.seed if seed is None else seed
    seed_everything(run_seed)
    device = _default_device()

    graph = load_citation_graph(
        split.dataset_name,
        root=data_root,
        remove_self_loops=True,
    )

    node_features = graph.x
    if resolved_config.use_structural_features:
        structural = compute_structural_features(
            num_nodes=split.num_nodes,
            train_edge_index=split.train_edge_index,
            num_eigvecs=resolved_config.num_laplacian_eigvecs,
        )
        node_features = augment_features(node_features, structural=structural)

    x = node_features.to(device)
    train_edge_index = split.train_edge_index.to(device)
    train_pos_edges = split.train_pos_edge_index.to(device)
    forbidden = _train_positive_set(split.train_pos_edge_index)

    model = _build_model(
        method=method,
        input_dim=int(x.size(1)),
        config=resolved_config,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=resolved_config.learning_rate,
        weight_decay=resolved_config.weight_decay,
    )

    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(run_seed)

    # Cache the node embeddings for the duration of one eval cycle so we
    # do not re-encode the full graph per scoring batch.
    cached_emb: dict[str, Tensor] = {}

    def _refresh_cache() -> None:
        model.eval()
        with torch.no_grad():
            cached_emb["emb"] = model.encode(x, train_edge_index)

    @torch.no_grad()
    def _score_edges(edges: Tensor) -> Tensor:
        edges_dev = edges.to(device=device, dtype=torch.long)
        emb = cached_emb["emb"]
        return torch.sigmoid(model.score_edges(emb, edges_dev)).cpu().to(dtype=torch.float64)

    best_validation: dict[str, float] | None = None
    best_test: dict[str, float] | None = None
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    num_pos = int(train_pos_edges.size(1))

    neg_per_pos = int(resolved_config.train_negatives_per_positive)
    train_mode = resolved_config.train_mode
    batch_size = int(resolved_config.train_batch_size)

    # Pre-build the canonical (one-direction) train positives so we can
    # mask them per batch in the HeaRT-style loop.
    canonical_train_pos = split.train_pos_edge_index.to(device).t().contiguous()
    num_canonical = int(canonical_train_pos.size(0))

    def _train_one_epoch_fullbatch() -> float:
        model.train()
        train_neg = _sample_random_negative_edges(
            num_samples=num_pos * neg_per_pos,
            num_nodes=split.num_nodes,
            forbidden=forbidden,
            rng=cpu_rng,
        ).to(device)
        train_edges = torch.cat([train_pos_edges, train_neg], dim=1)
        labels = torch.cat(
            [
                torch.ones(num_pos, dtype=torch.float32, device=device),
                torch.zeros(num_pos * neg_per_pos, dtype=torch.float32, device=device),
            ]
        )
        pos_weight = torch.tensor(float(neg_per_pos), device=device)
        optimizer.zero_grad()
        emb = model.encode(x, train_edge_index)
        logits = model.score_edges(emb, train_edges)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())

    def _train_one_epoch_minibatch_dropedge() -> float:
        """HeaRT-style: mini-batches + per-batch removal of supervision edges
        from the message-passing graph. See Li et al.'s
        ``benchmarking/HeaRT_small/main_gnn_CoraCiteseerPubmed.py::train``.
        """
        model.train()
        total_loss = 0.0
        total_examples = 0
        perm_order = torch.randperm(num_canonical, device=device)
        for start in range(0, num_canonical, batch_size):
            perm = perm_order[start : start + batch_size]
            optimizer.zero_grad()
            mask = torch.ones(num_canonical, dtype=torch.bool, device=device)
            mask[perm] = False
            kept = canonical_train_pos[mask].t().contiguous()
            mp_edges = torch.cat([kept, kept.flip(0)], dim=1)

            emb = model.encode(x, mp_edges)
            batch_pos = canonical_train_pos[perm].t().contiguous()
            pos_logits = model.score_edges(emb, batch_pos)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits)
            )

            num_batch_pos = int(batch_pos.size(1))
            neg_src = torch.randint(
                0, split.num_nodes, (num_batch_pos * neg_per_pos,), device=device
            )
            neg_dst = torch.randint(
                0, split.num_nodes, (num_batch_pos * neg_per_pos,), device=device
            )
            neg_edges = torch.stack([neg_src, neg_dst], dim=0)
            neg_logits = model.score_edges(emb, neg_edges)
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits)
            )

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * num_batch_pos
            total_examples += num_batch_pos
        return total_loss / max(total_examples, 1)

    train_step = (
        _train_one_epoch_minibatch_dropedge
        if train_mode == "minibatch_dropedge"
        else _train_one_epoch_fullbatch
    )

    for epoch in range(1, resolved_config.epochs + 1):
        train_step()

        if epoch % resolved_config.eval_every != 0 and epoch != resolved_config.epochs:
            continue

        _refresh_cache()
        validation, test = evaluate_with_heart_negatives(
            score_fn=_score_edges,
            val_pos_edge_index=split.val_pos_edge_index,
            val_neg_nodes=split.val_neg_nodes,
            test_pos_edge_index=split.test_pos_edge_index,
            test_neg_nodes=split.test_neg_nodes,
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

    if best_validation is None or best_test is None:
        raise RuntimeError("training did not produce any validation metrics")

    params: dict[str, Any] = resolved_config.to_dict()
    params.update(
        {
            "device": str(device),
            "split_seed": int(split.seed),
            "K": int(split.K),
            "input_feature_dim": int(x.size(1)),
            "train_graph_edges": split.train_graph_edge_count,
            "num_train_positives": int(num_pos),
            "best_epoch": int(best_epoch),
        }
    )

    return build_experiment_result(
        dataset=split.dataset_name,
        method=method_name_override or method.value,
        seed=run_seed,
        params=params,
        validation=best_validation,
        test=best_test,
    )


__all__ = [
    "GNNBaselineConfig",
    "GNNMethod",
    "run_gnn_baseline",
]
