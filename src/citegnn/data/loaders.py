"""Dataset loaders for citation graphs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import (
    coalesce,
    remove_self_loops as pyg_remove_self_loops,
    to_undirected,
)


class CitationDataset(StrEnum):
    """Supported citation datasets."""

    CORA = "Cora"
    PUBMED = "PubMed"


@dataclass(slots=True)
class GraphData:
    """Normalized graph container used throughout the pipeline."""

    name: str
    num_nodes: int
    num_features: int
    x: Tensor
    edge_index: Tensor
    original_edge_index: Tensor
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor
    is_undirected: bool = True


def _require_tensor(value: Tensor | None, *, name: str) -> Tensor:
    """Return one required tensor attribute from a PyG data object."""

    if value is None:
        raise ValueError(f"Planetoid sample is missing required tensor attribute '{name}'")
    return value


def _require_int(value: int | None, *, name: str) -> int:
    """Return one required integer attribute from a PyG data object."""

    if value is None:
        raise ValueError(f"Planetoid sample is missing required integer attribute '{name}'")
    return int(value)


def load_citation_graph(
    dataset: str | CitationDataset,
    *,
    root: str | Path = "data",
    normalize_features: bool = True,
    remove_self_loops: bool = True,
) -> GraphData:
    """Load and normalize one undirected citation graph from Planetoid."""

    dataset_name = CitationDataset(dataset).value
    transform = NormalizeFeatures() if normalize_features else None
    pyg_dataset = Planetoid(root=str(Path(root)), name=dataset_name, transform=transform)
    data = cast(Data, pyg_dataset[0])

    x = _require_tensor(data.x, name="x")
    base_edge_index = _require_tensor(data.edge_index, name="edge_index")
    train_mask = _require_tensor(data.train_mask, name="train_mask")
    val_mask = _require_tensor(data.val_mask, name="val_mask")
    test_mask = _require_tensor(data.test_mask, name="test_mask")
    num_nodes = _require_int(data.num_nodes, name="num_nodes")

    original_edge_index = base_edge_index.clone()
    edge_index = base_edge_index.clone()

    if remove_self_loops:
        edge_index, _ = pyg_remove_self_loops(edge_index)

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return GraphData(
        name=dataset_name,
        num_nodes=num_nodes,
        num_features=int(x.size(-1)),
        x=x,
        edge_index=edge_index,
        original_edge_index=original_edge_index,
        train_mask=train_mask.clone(),
        val_mask=val_mask.clone(),
        test_mask=test_mask.clone(),
        is_undirected=True,
    )
