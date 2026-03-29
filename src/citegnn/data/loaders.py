"""Dataset loaders for citation graphs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import (
    coalesce,
    is_undirected,
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
    is_directed: bool


def load_citation_graph(
    dataset: str | CitationDataset,
    *,
    root: str | Path = "data",
    normalize_features: bool = True,
    undirected: bool = True,
    remove_self_loops: bool = True,
) -> GraphData:
    """Load and normalize one citation graph from the Planetoid benchmark suite."""

    dataset_name = CitationDataset(dataset).value
    transform = NormalizeFeatures() if normalize_features else None
    pyg_dataset = Planetoid(root=str(Path(root)), name=dataset_name, transform=transform)
    data = pyg_dataset[0]

    original_edge_index = data.edge_index.clone()
    edge_index = data.edge_index.clone()

    if remove_self_loops:
        edge_index, _ = pyg_remove_self_loops(edge_index)

    directed = not bool(is_undirected(edge_index))

    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

    edge_index = coalesce(edge_index, num_nodes=data.num_nodes)

    return GraphData(
        name=dataset_name,
        num_nodes=int(data.num_nodes),
        num_features=int(data.num_features),
        x=data.x,
        edge_index=edge_index,
        original_edge_index=original_edge_index,
        train_mask=data.train_mask.clone(),
        val_mask=data.val_mask.clone(),
        test_mask=data.test_mask.clone(),
        is_directed=directed and not undirected,
    )
