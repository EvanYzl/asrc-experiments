"""Packed batching/view fusion checks shared by DMKGC and IMKGC.

Run in a fresh process with one source root argument so each repository's
top-level ``src`` package is tested independently.
"""

import copy
import sys
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GraphLoader


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SOURCE))

from src.gnn import GNN  # noqa: E402
from src.utils import MyData, nodes_to_k_graph  # noqa: E402
from src.validate import Tester  # noqa: E402


CACHE = (
    ROOT.parent / "data" / "raw" / "dmkgc" / "datasetdbp5l"
    / "el_en_es_fr_ja" / "kg0_k_subgraph_list.graph"
)


def reference_batch(packed, rows):
    result = []
    for view in range(packed["x"].shape[1]):
        graphs = []
        for row in rows.tolist():
            n_nodes = int(packed["num_nodes"][row, view])
            n_edges = int(packed["num_edges"][row, view])
            graphs.append(MyData(
                x=packed["x"][row, view, :n_nodes].long(),
                edge_index=packed["edge_index"][row, view, :, :n_edges].long(),
                edge_kg_index=packed["edge_kg_index"][row, view, :, :n_edges].long(),
                edge_attr=packed["edge_attr"][row, view, :n_edges].long(),
                y=packed["y"][row, view].long().reshape(1),
                num_size=torch.tensor([n_nodes], dtype=torch.long),
            ))
        result.append(next(iter(GraphLoader(graphs, batch_size=len(graphs), shuffle=False))))
    return result


class EncoderBundle(nn.Module):
    def __init__(self, num_entities, num_relations, num_kgs, dim=16):
        super().__init__()
        self.entities = nn.Embedding(num_entities, dim)
        self.relations = nn.Embedding(num_relations, dim)
        self.prior = nn.Embedding(num_relations, 1)
        self.encoder = GNN(
            num_kgs=num_kgs,
            in_dim=dim,
            in_edge_dim=dim,
            n_hid=dim,
            out_dim=dim,
            n_heads=1,
            n_layers=2,
            dropout=0.0,
        )

    def encode_one(self, graph):
        return self.encoder(
            self.entities(graph.x), graph.edge_index, graph.edge_kg_index,
            self.prior(graph.edge_attr), self.relations(graph.edge_attr),
            graph.y, graph.num_size,
        )

    def reference(self, graphs):
        return torch.stack([self.encode_one(graph) for graph in graphs])

    def fused(self, graphs):
        batch_size = graphs[0].y.shape[0]
        combined = Batch.from_data_list(graphs)
        return self.encode_one(combined).reshape(len(graphs), batch_size, -1)


def gradients(model):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def main():
    torch.manual_seed(2020)
    packed = torch.load(CACHE, map_location="cpu")
    rows = torch.tensor([7, 0, 7, 3, 19, 2], dtype=torch.long)
    actual = nodes_to_k_graph(packed, rows, "cpu")
    expected = reference_batch(packed, rows)
    for view, (got, want) in enumerate(zip(actual, expected)):
        for field in ("x", "edge_index", "edge_kg_index", "edge_attr", "y", "num_size"):
            assert torch.equal(getattr(got, field), getattr(want, field)), (view, field)

    sizes = torch.randint(1, 12, (257,), dtype=torch.long)
    centres = torch.stack([torch.randint(0, int(size), ()) for size in sizes])
    expected_indices = centres + torch.cat((torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1]))
    probe_gnn = GNN(5, 4, 4, 4, 4, 1, 1, 0.0)
    assert torch.equal(probe_gnn.get_real_index(centres, sizes), expected_indices)

    tester = Tester.__new__(Tester)
    for _ in range(100):
        batch, entities = 9, 73
        rankings = torch.stack([torch.randperm(entities) for _ in range(batch)])
        ground_truth = torch.randint(0, entities, (batch, 1))
        filter_mask = torch.rand(batch, entities) < 0.08
        reference_rankings = rankings.clone()
        for row in range(batch):
            ranked_filter = filter_mask[row].gather(0, reference_rankings[row])
            reference_rankings[row, ranked_filter] = -1
            _, stable_partition = ranked_filter.long().sort(stable=True)
            reference_rankings[row] = reference_rankings[row].gather(0, stable_partition)
        expected_metrics = tester.get_hit_mrr(reference_rankings, ground_truth)
        actual_metrics = tester.get_filtered_hit_mrr(rankings, ground_truth, filter_mask)
        assert expected_metrics[:2] == actual_metrics[:2]
        torch.testing.assert_close(actual_metrics[2], expected_metrics[2], rtol=0, atol=0)

    num_entities = int(packed["x"].max()) + 1
    num_relations = int(packed["edge_attr"].max()) + 1
    reference_model = EncoderBundle(num_entities, num_relations, 5)
    fused_model = copy.deepcopy(reference_model)
    reference_output = reference_model.reference(actual)
    fused_output = fused_model.fused(actual)
    torch.testing.assert_close(fused_output, reference_output, rtol=1e-6, atol=1e-7)
    probe = torch.linspace(-1, 1, reference_output.numel()).reshape_as(reference_output)
    (reference_output * probe).sum().backward()
    (fused_output * probe).sum().backward()
    reference_gradients = gradients(reference_model)
    fused_gradients = gradients(fused_model)
    for name in reference_gradients:
        if reference_gradients[name] is None or fused_gradients[name] is None:
            assert reference_gradients[name] is None and fused_gradients[name] is None, name
        else:
            torch.testing.assert_close(
                fused_gradients[name], reference_gradients[name], rtol=1e-4, atol=1e-6
            )

    print(f"source={SOURCE.name}")
    print("packed_batch_fields_equivalent=True")
    print("vectorized_center_offsets_equivalent=True")
    print("vectorized_filtered_metrics_equivalent=True")
    print("fused_views_forward_equivalent=True")
    print("fused_views_backward_gradients_equivalent=True")


if __name__ == "__main__":
    main()
