"""Field and batching equivalence checks for SS-AGA packed subgraphs."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

SOURCE = Path(__file__).resolve().parents[1] / "sources" / "SS-AGA"
sys.path.insert(0, str(SOURCE))

from src.utils import (
    PackedSubgraphList,
    nodes_to_graph,
    subgrarph_list_from_alignment,
)


def make_graph(center, nodes, edges, relations):
    return Data(
        x=torch.tensor(nodes, dtype=torch.long),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_attr=torch.tensor(relations, dtype=torch.long),
        y=torch.tensor([center], dtype=torch.long),
        num_size=torch.tensor([len(nodes)], dtype=torch.long),
    )


def fields(batch):
    return {
        name: getattr(batch, name)
        for name in ("x", "edge_index", "edge_attr", "y", "num_size", "batch", "ptr")
    }


def assert_same(left, right):
    for name, expected in fields(left).items():
        observed = fields(right)[name]
        assert torch.equal(expected, observed), (name, expected, observed)


def main():
    graphs = [
        make_graph(0, [5], [[0], [0]], [11]),
        make_graph(1, [7, 8, 9], [[0, 2], [1, 1]], [12, 13]),
        make_graph(0, [20, 21], [[], []], []),
        make_graph(2, [30, 31, 32, 33], [[0, 1, 3], [2, 2, 1]], [2, 3, 4]),
    ]
    packed = PackedSubgraphList(graphs)
    order = torch.tensor([3, 0, 2, 1, 3], dtype=torch.long)
    expected = next(iter(DataLoader([graphs[int(i)] for i in order], batch_size=len(order))))
    observed = nodes_to_graph(packed, order)
    assert_same(expected, observed)

    replacement = make_graph(1, [100, 101], [[0, 1], [1, 0]], [8, 9])
    packed[1] = replacement
    expected_mutated = next(iter(DataLoader(
        [graphs[3], replacement, graphs[0]], batch_size=3
    )))
    observed_mutated = nodes_to_graph(packed, torch.tensor([3, 1, 0]))
    assert_same(expected_mutated, observed_mutated)

    # Exercise the repository's actual self-learning mutation function on a
    # normal list and on packed storage, including a second update to an
    # already-overridden graph.
    kg0_graphs = [copy.deepcopy(graphs[0]), copy.deepcopy(graphs[1])]
    kg1_graphs = [copy.deepcopy(graphs[2]), copy.deepcopy(graphs[3])]
    reference0 = SimpleNamespace(
        num_relation=20, entity_id_base=0, relation_id_base=0,
        subgraph_list_kg=copy.deepcopy(kg0_graphs),
        subgraph_list_align=copy.deepcopy(kg0_graphs),
    )
    reference1 = SimpleNamespace(
        num_relation=20, entity_id_base=2, relation_id_base=20,
        subgraph_list_kg=copy.deepcopy(kg1_graphs),
        subgraph_list_align=copy.deepcopy(kg1_graphs),
    )
    packed0 = SimpleNamespace(
        num_relation=20, entity_id_base=0, relation_id_base=0,
        subgraph_list_kg=PackedSubgraphList(copy.deepcopy(kg0_graphs)),
        subgraph_list_align=PackedSubgraphList(copy.deepcopy(kg0_graphs)),
    )
    packed1 = SimpleNamespace(
        num_relation=20, entity_id_base=2, relation_id_base=20,
        subgraph_list_kg=PackedSubgraphList(copy.deepcopy(kg1_graphs)),
        subgraph_list_align=PackedSubgraphList(copy.deepcopy(kg1_graphs)),
    )
    for links in (torch.tensor([[0, 1]]), torch.tensor([[0, 0]])):
        for is_kg in (True, False):
            subgrarph_list_from_alignment(links, reference0, reference1, is_kg)
            subgrarph_list_from_alignment(links, packed0, packed1, is_kg)
    for reference_kg, packed_kg in ((reference0, packed0), (reference1, packed1)):
        for field_name in ("subgraph_list_kg", "subgraph_list_align"):
            reference_list = getattr(reference_kg, field_name)
            packed_list = getattr(packed_kg, field_name)
            for graph_id in range(len(reference_list)):
                for field in ("x", "edge_index", "edge_attr", "y", "num_size"):
                    assert torch.equal(
                        getattr(reference_list[graph_id], field),
                        getattr(packed_list[graph_id], field),
                    ), (field_name, graph_id, field)
    print("ssaga_packed_subgraphs_equivalent=True")


if __name__ == "__main__":
    main()
