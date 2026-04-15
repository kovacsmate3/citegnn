from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import importlib.util
import shutil
import sys
from pathlib import Path
import uuid

import torch

from citegnn.data.loaders import GraphData
from citegnn.data.splits import LinkPredictionSplit
from citegnn.evaluation import ExperimentResult
from citegnn.utils.metrics import ClassificationMetrics


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


def make_link_prediction_split() -> LinkPredictionSplit:
    train_graph_pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)]
    train_supervision_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    return LinkPredictionSplit(
        dataset_name="Cora",
        seed=42,
        val_ratio=0.1,
        test_ratio=0.2,
        negative_ratio=1.0,
        train_mask=torch.tensor([True, True, True, False, False, False]),
        val_mask=torch.tensor([False, False, False, True, False, False]),
        test_mask=torch.tensor([False, False, False, False, True, True]),
        train_graph_edge_index=edge_index_from_pairs(train_graph_pairs, bidirectional=True),
        val_graph_edge_index=edge_index_from_pairs(train_graph_pairs, bidirectional=True),
        test_graph_edge_index=edge_index_from_pairs(train_graph_pairs, bidirectional=True),
        train_pos_edge_index=edge_index_from_pairs(train_supervision_pairs),
        val_pos_edge_index=edge_index_from_pairs([(0, 2)]),
        test_pos_edge_index=edge_index_from_pairs([(1, 3)]),
        train_neg_edge_index=edge_index_from_pairs([(0, 4), (1, 4), (2, 5), (4, 5)]),
        val_neg_edge_index=edge_index_from_pairs([(1, 5)]),
        test_neg_edge_index=edge_index_from_pairs([(0, 5)]),
    )


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


def make_experiment_result(method: str) -> ExperimentResult:
    metrics = make_metrics()
    return ExperimentResult(
        dataset="Cora",
        method=method,
        seed=42,
        params={"source": "test"},
        validation=metrics.to_dict(),
        test=metrics.to_dict(),
    )


def save_split_payload(path: Path, split: LinkPredictionSplit) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(split), path)
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
