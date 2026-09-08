"""Equivalence check for the low-memory DMKGC preprocessing compatibility port."""

import copy
import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphLoader

from src.utils import (
    MyData,
    create_subgraph_list,
    get_k_subgraph_list,
    nodes_to_k_graph,
)


def reference_domain_views(subgraph_list, node_index, kg_index, num_kgs, entity2kgidx, total_num_nodes):
    output = []

    def get_kg_index(index, subset, node2kgidx):
        return node2kgidx[subset[index].item()]

    for center_tensor in node_index:
        center = int(center_tensor)
        subgraph = subgraph_list[center]
        nodes = subgraph.x
        edge_index = subgraph.edge_index
        graphs = []
        for view in range(num_kgs):
            subset = [center]
            node2kgidx = {center: kg_index}
            endpoint_ok = edge_index.new_zeros((2, edge_index.shape[1]), dtype=torch.bool)
            for row in range(2):
                for col in range(edge_index.shape[1]):
                    node = nodes[edge_index[row, col]]
                    if view in entity2kgidx[node.item()] or node == center:
                        subset.append(node.item())
                        endpoint_ok[row, col] = True
                        if node != center:
                            node2kgidx[node.item()] = view
            subset, inv = torch.tensor(subset).unique(return_inverse=True)
            inv = inv[:1]
            edge_mask = endpoint_ok[0] & endpoint_ok[1]
            edge_global = nodes[edge_index[:, edge_mask]]

            # Isolated nodes have no effect on the queried center and are the
            # only nodes deliberately omitted by the compact representation.
            subset = torch.unique(torch.cat((torch.tensor([center]), edge_global.reshape(-1))))
            inv = torch.searchsorted(subset, torch.tensor(center)).view(1)
            edge_index_view = torch.searchsorted(subset, edge_global)
            edge_kg_index = copy.deepcopy(edge_index_view)
            edge_kg_index.apply_(lambda value: get_kg_index(value, subset, node2kgidx))
            graphs.append(MyData(
                x=subset,
                edge_index=edge_index_view,
                edge_kg_index=edge_kg_index,
                edge_attr=subgraph.edge_attr[edge_mask],
                y=inv,
                num_size=torch.tensor([len(subset)]),
            ))
        output.append(graphs)
    return output


def main():
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 1, 4, 6, 2],
        [2, 2, 4, 4, 6, 6, 0, 5, 3, 6],
    ])
    edge_type = torch.arange(edge_index.shape[1])
    base = create_subgraph_list(edge_index, edge_type, total_num_nodes=7, num_hops=2, k=4)
    node_index = torch.arange(7)
    entity2kgidx = {
        0: [0], 1: [0, 1], 2: [0, 1], 3: [1],
        4: [0, 1], 5: [1], 6: [0, 1],
    }
    reference = reference_domain_views(base, node_index, 0, 2, entity2kgidx, 7)

    with tempfile.TemporaryDirectory() as temp_dir:
        packed = get_k_subgraph_list(
            base, node_index, 0, 2, entity2kgidx, 7, temp_dir,
        )
        for row in range(7):
            batches = nodes_to_k_graph(packed, torch.tensor([row]), 'cpu')
            for view, batch in enumerate(batches):
                expected = next(iter(GraphLoader([reference[row][view]], batch_size=1)))
                for field in ('x', 'edge_index', 'edge_kg_index', 'edge_attr', 'y', 'num_size'):
                    assert torch.equal(getattr(batch, field), getattr(expected, field)), (row, view, field)

    print('compact_subgraphs_equivalent=True')


if __name__ == '__main__':
    main()
