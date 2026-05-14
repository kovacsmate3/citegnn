"""Shared test fixtures and helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import importlib.util
import shutil
import sys
from pathlib import Path
import uuid

import torch

from citegnn.data.heart_split import HeartSplit
from citegnn.data.loaders import GraphData
from citegnn.evaluation import ExperimentResult
from citegnn.utils.metrics import ClassificationMetrics
from citegnn.utils.ranking_metrics import RankingMetrics


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def edge_index_from_pairs(
    pairs: list[tuple[int, int]],
    *,
    bidirectional: bool = False,
) -> torch.Tensor:
    all_pairs = list(pairs)
    if bidirectional:
        all_pairs.extend((target, source) for source, target in pairs if source != target)
    return torch.tensor(all_pairs, dtype=torch.long).t().contiguous()


def make_graph_data(*, name: str = "Cora") -> GraphData:
    x = torch.tensor(
        [
            [3.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [1.0, 2.0, 0.0],
            [1.0, 1.0, 3.0],
            [0.0, 1.0, 4.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    base_pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 2),
        (1, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (2, 5),
    ]
    edge_index = edge_index_from_pairs(base_pairs, bidirectional=True)
    return GraphData(
        name=name,
        num_nodes=int(x.size(0)),
        num_features=int(x.size(1)),
        x=x,
        edge_index=edge_index,
        original_edge_index=edge_index.clone(),
        train_mask=torch.tensor([True, True, True, False, False, False]),
        val_mask=torch.tensor([False, False, False, True, False, False]),
        test_mask=torch.tensor([False, False, False, False, True, True]),
        is_undirected=True,
    )


def make_heart_split(*, K: int = 4) -> HeartSplit:
    """Build a tiny deterministic HeartSplit for unit tests.

    The graph has 6 nodes; train uses 5 canonical edges, val holds out
    one, test holds out one. Hard negatives are pre-set so tests can
    assert exact ranking outcomes.
    """

    train_pos = edge_index_from_pairs([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)])
    val_pos = edge_index_from_pairs([(0, 2)])
    test_pos = edge_index_from_pairs([(1, 3)])
    train_edge_index = torch.cat(
        [train_pos, torch.stack([train_pos[1], train_pos[0]], dim=0)],
        dim=1,
    )
    val_neg_nodes = torch.tensor([[5, 4, 3, 5]], dtype=torch.long)
    test_neg_nodes = torch.tensor([[5, 4, 0, 5]], dtype=torch.long)
    return HeartSplit(
        dataset_name="Cora",
        seed=42,
        split_ratio=(0.85, 0.05, 0.10),
        K=K,
        num_nodes=6,
        train_edge_index=train_edge_index,
        train_pos_edge_index=train_pos,
        val_pos_edge_index=val_pos,
        test_pos_edge_index=test_pos,
        val_neg_nodes=val_neg_nodes,
        test_neg_nodes=test_neg_nodes,
        heuristics_used=("resource_allocation", "personalized_pagerank", "feature_cosine"),
    )


# Backwards-compatible alias used by older tests we keep around.
make_link_prediction_split = make_heart_split


def make_metrics(
    *,
    roc_auc: float = 0.9,
    average_precision: float = 0.85,
    precision: float = 0.8,
    recall: float = 0.75,
    f1: float = 0.77,
    threshold: float = 0.5,
    positive_rate: float = 0.5,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=roc_auc,
        average_precision=average_precision,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=threshold,
        positive_rate=positive_rate,
    )


def make_ranking_metrics() -> RankingMetrics:
    return RankingMetrics(
        mrr=0.42,
        hits_at={1: 0.3, 3: 0.5, 10: 0.7, 100: 0.95},
        num_positives=10,
        num_negatives_per_positive=500,
    )


def make_experiment_result(method: str) -> ExperimentResult:
    classification = make_metrics().to_dict()
    ranking = make_ranking_metrics().to_dict()
    payload: dict[str, float] = {**ranking, **classification}
    return ExperimentResult(
        dataset="Cora",
        method=method,
        seed=42,
        params={"source": "test"},
        validation=payload,
        test=payload,
    )


def save_split_payload(path: Path, split: HeartSplit) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(split)
    payload["split_ratio"] = tuple(split.split_ratio)
    payload["heuristics_used"] = tuple(split.heuristics_used)
    torch.save(payload, path)
    return path


def load_script_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def workspace_temp_dir(prefix: str):
    root = REPO_ROOT / ".test-output"
    root.mkdir(exist_ok=True)
    path = root / f"{prefix}-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
