"""Link prediction split creation and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from .loaders import GraphData

@dataclass(slots=True)
class LinkPredictionSplit:
    """Serialized edge splits for reproducible link prediction experiments."""

    dataset_name: str
    is_undirected: bool
    seed: int
    val_ratio: float
    test_ratio: float
    negative_ratio: float
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor
    train_graph_edge_index: Tensor
    val_graph_edge_index: Tensor
    test_graph_edge_index: Tensor
    train_pos_edge_index: Tensor
    val_pos_edge_index: Tensor
    test_pos_edge_index: Tensor
    train_neg_edge_index: Tensor
    val_neg_edge_index: Tensor
    test_neg_edge_index: Tensor


def _validate_ratios(val_ratio: float, test_ratio: float) -> None:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")


def _split_labeled_edges(
    edge_label_index: Tensor,
    edge_label: Tensor,
) -> tuple[Tensor, Tensor]:
    """Separate positive and negative supervision edges from PyG split outputs."""

    if edge_label_index.dim() != 2 or edge_label_index.size(0) != 2:
        raise ValueError("edge_label_index must have shape [2, num_edges]")

    labels = edge_label.to(torch.bool)
    pos_edges = edge_label_index[:, labels]
    neg_edges = edge_label_index[:, ~labels]
    return pos_edges.contiguous(), neg_edges.contiguous()


def create_link_prediction_split(
    graph: GraphData,
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> LinkPredictionSplit:
    """Create deterministic train/validation/test splits using PyG RandomLinkSplit."""

    _validate_ratios(val_ratio=val_ratio, test_ratio=test_ratio)
    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive")

    base_data = Data(
        x=graph.x,
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
    )
    splitter = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=not graph.is_directed,
        add_negative_train_samples=True,
        neg_sampling_ratio=negative_ratio,
        split_labels=False,
    )
    train_data, val_data, test_data = splitter(base_data)

    train_pos, train_neg = _split_labeled_edges(
        train_data.edge_label_index,
        train_data.edge_label,
    )
    val_pos, val_neg = _split_labeled_edges(
        val_data.edge_label_index,
        val_data.edge_label,
    )
    test_pos, test_neg = _split_labeled_edges(
        test_data.edge_label_index,
        test_data.edge_label,
    )

    return LinkPredictionSplit(
        dataset_name=graph.name,
        is_undirected=not graph.is_directed,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        negative_ratio=negative_ratio,
        train_mask=graph.train_mask,
        val_mask=graph.val_mask,
        test_mask=graph.test_mask,
        train_graph_edge_index=train_data.edge_index,
        val_graph_edge_index=val_data.edge_index,
        test_graph_edge_index=test_data.edge_index,
        train_pos_edge_index=train_pos,
        val_pos_edge_index=val_pos,
        test_pos_edge_index=test_pos,
        train_neg_edge_index=train_neg,
        val_neg_edge_index=val_neg,
        test_neg_edge_index=test_neg,
    )


def save_split_artifact(
    split: LinkPredictionSplit,
    *,
    output_dir: str | Path = "artifacts/splits",
) -> Path:
    """Persist one split artifact to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifact_path = output_path / f"{split.dataset_name.lower()}_seed{split.seed}_lp.pt"
    torch.save(asdict(split), artifact_path)
    return artifact_path


def load_split_artifact(path: str | Path) -> LinkPredictionSplit:
    """Load one persisted split artifact from disk."""

    payload = torch.load(Path(path), map_location="cpu")
    return LinkPredictionSplit(**payload)
