"""Link prediction split creation and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from .loaders import GraphData
from ..utils.seed import seed_everything


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

    @property
    def graph_variant(self) -> str:
        """Return a stable graph-variant label for artifact naming."""

        return "undirected" if self.is_undirected else "directed"

    @property
    def train_graph_edge_count(self) -> int:
        """Return the number of message-passing edges in the training graph."""

        return int(self.train_graph_edge_index.size(1))

    @property
    def train_supervision_edge_count(self) -> int:
        """Return the number of labeled training edges used for supervision."""

        return int(
            self.train_pos_edge_index.size(1) + self.train_neg_edge_index.size(1)
        )


def _edge_pairs(edge_index: Tensor, *, is_undirected: bool) -> set[tuple[int, int]]:
    """Convert an edge index into hashable edge pairs for overlap checks."""

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")

    pairs: set[tuple[int, int]] = set()
    for source, target in edge_index.t().tolist():
        if is_undirected and source > target:
            source, target = target, source
        pairs.add((int(source), int(target)))
    return pairs


def _validate_ratios(val_ratio: float, test_ratio: float) -> None:
    """Validate the requested validation and test edge ratios."""

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


def _artifact_graph_variant_suffix(*, is_undirected: bool) -> str:
    """Return the suffix needed only for non-default graph variants."""

    return "" if is_undirected else "_directed"


def validate_link_prediction_split(split: LinkPredictionSplit) -> None:
    """Check that held-out positives do not leak into the training graph."""

    train_pairs = _edge_pairs(
        split.train_graph_edge_index,
        is_undirected=split.is_undirected,
    )
    train_supervision_pairs = _edge_pairs(
        split.train_pos_edge_index,
        is_undirected=split.is_undirected,
    )
    val_pairs = _edge_pairs(
        split.val_pos_edge_index,
        is_undirected=split.is_undirected,
    )
    test_pairs = _edge_pairs(
        split.test_pos_edge_index,
        is_undirected=split.is_undirected,
    )

    if train_pairs & val_pairs:
        raise ValueError("validation positive edges leak into the training graph")
    if train_pairs & test_pairs:
        raise ValueError("test positive edges leak into the training graph")
    if train_supervision_pairs & val_pairs:
        raise ValueError("validation positive edges leak into training supervision")
    if train_supervision_pairs & test_pairs:
        raise ValueError("test positive edges leak into training supervision")
    if val_pairs & test_pairs:
        raise ValueError("validation and test positive edges overlap")


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
    seed_everything(seed)

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

    split = LinkPredictionSplit(
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
    validate_link_prediction_split(split)
    return split


def save_split_artifact(
    split: LinkPredictionSplit,
    *,
    output_dir: str | Path = "artifacts/splits",
) -> Path:
    """Persist one split artifact to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    variant_suffix = _artifact_graph_variant_suffix(is_undirected=split.is_undirected)
    artifact_path = (
        output_path
        / f"{split.dataset_name.lower()}{variant_suffix}_seed{split.seed}_lp.pt"
    )
    torch.save(asdict(split), artifact_path)
    return artifact_path


def load_split_artifact(path: str | Path) -> LinkPredictionSplit:
    """Load one persisted split artifact from disk."""

    payload = torch.load(Path(path), map_location="cpu")
    split = LinkPredictionSplit(**payload)
    validate_link_prediction_split(split)
    return split
