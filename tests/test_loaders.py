from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from citegnn.data import loaders


class _DummyPlanetoid:
    def __init__(self, root: str, name: str, transform=None) -> None:
        self.root = root
        self.name = name
        self.transform = transform

    def __getitem__(self, index: int) -> Data:
        if index != 0:
            raise IndexError(index)
        data = Data(
            x=torch.tensor(
                [
                    [2.0, 0.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [1.0, 1.0, 2.0],
                ],
                dtype=torch.float32,
            ),
            edge_index=torch.tensor(
                [
                    [0, 0, 1],
                    [0, 1, 2],
                ],
                dtype=torch.long,
            ),
        )
        data.train_mask = torch.tensor([True, False, False])
        data.val_mask = torch.tensor([False, True, False])
        data.test_mask = torch.tensor([False, False, True])
        if self.transform is not None:
            data = self.transform(data)
        return data


class _UndirectedPlanetoid:
    def __init__(self, root: str, name: str, transform=None) -> None:
        self.transform = transform

    def __getitem__(self, index: int) -> Data:
        if index != 0:
            raise IndexError(index)
        data = Data(
            x=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        data.train_mask = torch.tensor([True, False])
        data.val_mask = torch.tensor([False, True])
        data.test_mask = torch.tensor([False, False])
        if self.transform is not None:
            data = self.transform(data)
        return data


class LoaderTests(unittest.TestCase):
    def test_require_helpers_raise_for_missing_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "tensor attribute 'x'"):
            loaders._require_tensor(None, name="x")
        with self.assertRaisesRegex(ValueError, "integer attribute 'num_nodes'"):
            loaders._require_int(None, name="num_nodes")

    @patch.object(loaders, "Planetoid", _DummyPlanetoid)
    def test_load_citation_graph_normalizes_and_undirects_edges(self) -> None:
        graph = loaders.load_citation_graph(
            loaders.CitationDataset.CORA,
            root="ignored",
            normalize_features=True,
            undirected=True,
            remove_self_loops=True,
        )

        pairs = {tuple(edge) for edge in graph.edge_index.t().tolist()}
        self.assertEqual(graph.name, "Cora")
        self.assertEqual(graph.num_nodes, 3)
        self.assertEqual(graph.num_features, 3)
        self.assertFalse(graph.is_directed)
        self.assertNotIn((0, 0), pairs)
        self.assertIn((0, 1), pairs)
        self.assertIn((1, 0), pairs)
        self.assertIn((1, 2), pairs)
        self.assertIn((2, 1), pairs)
        self.assertTrue(torch.equal(graph.original_edge_index, torch.tensor([[0, 0, 1], [0, 1, 2]])))
        row_sums = graph.x.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums)))

    @patch.object(loaders, "Planetoid", _DummyPlanetoid)
    def test_load_citation_graph_can_preserve_directionality(self) -> None:
        graph = loaders.load_citation_graph(
            "Cora",
            root="ignored",
            normalize_features=False,
            undirected=False,
            remove_self_loops=True,
        )

        pairs = {tuple(edge) for edge in graph.edge_index.t().tolist()}
        self.assertTrue(graph.is_directed)
        self.assertIn((0, 1), pairs)
        self.assertIn((1, 2), pairs)
        self.assertNotIn((1, 0), pairs)

    @patch.object(loaders, "Planetoid", _DummyPlanetoid)
    def test_load_citation_graph_can_keep_self_loops_when_requested(self) -> None:
        graph = loaders.load_citation_graph(
            "Cora",
            root="ignored",
            normalize_features=False,
            undirected=False,
            remove_self_loops=False,
        )

        pairs = {tuple(edge) for edge in graph.edge_index.t().tolist()}
        self.assertIn((0, 0), pairs)
        self.assertTrue(graph.is_directed)

    @patch.object(loaders, "Planetoid", _UndirectedPlanetoid)
    def test_load_citation_graph_detects_undirected_input_without_forcing_conversion(self) -> None:
        graph = loaders.load_citation_graph(
            "Cora",
            root="ignored",
            normalize_features=False,
            undirected=False,
            remove_self_loops=True,
        )

        self.assertFalse(graph.is_directed)
        self.assertEqual(graph.num_nodes, 2)
        self.assertEqual(graph.num_features, 2)

    def test_invalid_dataset_name_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            loaders.load_citation_graph("UnknownDataset")


if __name__ == "__main__":
    unittest.main()
