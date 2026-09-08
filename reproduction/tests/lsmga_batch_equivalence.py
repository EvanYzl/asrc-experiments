"""Regression checks for LSMGA's semantics-preserving packed batching path."""

import copy
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as GraphLoader


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sources" / "LSMGA"
sys.path.insert(0, str(SOURCE))

from src.lsmga_model import LSMGA  # noqa: E402
from src.utils import MyData, nodes_to_k_graph  # noqa: E402
from src.validate import Tester  # noqa: E402


CACHE = (
    ROOT.parent
    / "data"
    / "raw"
    / "dmkgc"
    / "datasetdbp5l"
    / "el_en_es_fr_ja"
    / "kg0_k_subgraph_list.graph"
)


def reference_batch(packed, node_index):
    """Reproduce the previous per-object PyG batching implementation."""
    rows = [int(index) for index in node_index]
    result = []
    for view in range(packed["x"].shape[1]):
        graphs = []
        for row in rows:
            n_nodes = int(packed["num_nodes"][row, view])
            n_edges = int(packed["num_edges"][row, view])
            graphs.append(MyData(
                x=packed["x"][row, view, :n_nodes].long(),
                edge_index=packed["edge_index"][row, view, :, :n_edges].long(),
                edge_kg_index=packed["edge_kg_index"][row, view, :, :n_edges].long(),
                edge_attr=packed["edge_attr"][row, view, :n_edges].long(),
                y=packed["y"][row, view].long().view(1),
                num_size=torch.tensor([n_nodes], dtype=torch.long),
            ))
        result.append(next(iter(GraphLoader(graphs, batch_size=len(graphs), shuffle=False))))
    return result


def assert_graph_fields_equal(actual, expected):
    for view, (got, want) in enumerate(zip(actual, expected)):
        for field in ("x", "edge_index", "edge_kg_index", "edge_attr", "y", "num_size"):
            assert torch.equal(getattr(got, field), getattr(want, field)), (view, field)


def model_args(packed):
    num_entities = int(packed["x"].max()) + 1
    num_relations = int(packed["edge_attr"].max()) + 1
    return Namespace(
        batch_size=4,
        num_kgs=int(packed["x"].shape[1]),
        num_entities=num_entities,
        num_relations=num_relations,
        entity_dim=16,
        relation_dim=16,
        device=torch.device("cpu"),
        margin=0.5,
        encoder_hdim_gnn=16,
        n_heads=1,
        n_layers_gnn=2,
        dropout=0.0,
    )


def reference_view_forward(model, graph_input_list, kg_index):
    """The released implementation's one-encoder-call-per-view path."""
    view_outputs = []
    for graph_input in graph_input_list:
        x_features = model.entity_embedding_layer(graph_input.x)
        edge_beta = model.relation_prior(graph_input.edge_attr)
        edge_relation = model.rel_embedding_layer(graph_input.edge_attr)
        view_outputs.append(model.encoder_KG(
            x_features,
            graph_input.edge_index,
            graph_input.edge_kg_index,
            edge_beta,
            edge_relation,
            graph_input.y,
            graph_input.num_size,
        ))
    return model.cross_graph_attention(torch.stack(view_outputs), kg_index)


def assert_parameter_gradients_close(reference_model, combined_model):
    for (name_ref, param_ref), (name_new, param_new) in zip(
        reference_model.named_parameters(), combined_model.named_parameters()
    ):
        assert name_ref == name_new
        if param_ref.grad is None or param_new.grad is None:
            assert param_ref.grad is None and param_new.grad is None, name_ref
        else:
            # Combining disconnected components can change floating-point
            # reduction order, but not the represented computation.
            torch.testing.assert_close(param_new.grad, param_ref.grad, rtol=1e-4, atol=1e-6)


def main():
    torch.manual_seed(2020)
    packed = torch.load(CACHE, map_location="cpu", weights_only=False)
    assert packed.get("format") == "packed_subgraphs_v1"

    # Closed-form centre offsets must match the released scalar recurrence,
    # including one-node graphs and non-zero local centre positions.
    sizes = torch.randint(1, 12, (257,), dtype=torch.long)
    local_centres = torch.stack([torch.randint(0, int(size), ()) for size in sizes])
    expected_centres = []
    base = torch.tensor([0], dtype=torch.long)
    for centre, size in zip(local_centres, sizes):
        expected_centres.append((base + centre).reshape(-1))
        base += size
    expected_centres = torch.cat(expected_centres)

    tester = Tester.__new__(Tester)
    for _ in range(100):
        batch, entities = 9, 73
        rankings = torch.stack([torch.randperm(entities) for _ in range(batch)])
        ground_truth = torch.randint(0, entities, (batch, 1))
        filter_mask = torch.rand(batch, entities) < 0.08
        reference_rankings = rankings.clone()
        for row in range(batch):
            row_filter = filter_mask[row].gather(0, reference_rankings[row])
            reference_rankings[row, row_filter] = -1
            _, stable_partition = row_filter.long().sort(stable=True)
            reference_rankings[row] = reference_rankings[row].gather(0, stable_partition)
        expected_metrics = tester.get_hit_mrr(reference_rankings, ground_truth)
        actual_metrics = tester.get_filtered_hit_mrr(rankings, ground_truth, filter_mask)
        assert expected_metrics[:2] == actual_metrics[:2]
        torch.testing.assert_close(actual_metrics[2], expected_metrics[2], rtol=0, atol=0)

    # Include duplicate and non-monotone rows: both are common with negative
    # sampling and catch incorrect offset/mask ordering.
    rows = torch.tensor([7, 0, 7, 3, 19, 2], dtype=torch.long)
    vectorized = nodes_to_k_graph(packed, rows, "cpu")
    reference = reference_batch(packed, rows)
    assert_graph_fields_equal(vectorized, reference)

    h_rows = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    t_rows = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    n_rows = torch.tensor([8, 9, 10, 11], dtype=torch.long)
    h_graph = nodes_to_k_graph(packed, h_rows, "cpu")
    t_graph = nodes_to_k_graph(packed, t_rows, "cpu")
    n_graph = nodes_to_k_graph(packed, n_rows, "cpu")
    combined_graph = nodes_to_k_graph(packed, torch.cat((h_rows, t_rows, n_rows)), "cpu")
    sample = torch.stack((h_rows, torch.zeros_like(h_rows), t_rows), dim=1)

    args = model_args(packed)
    reference_model = LSMGA(args)
    combined_model = copy.deepcopy(reference_model)
    actual_centres = reference_model.encoder_KG.get_real_index(local_centres, sizes)
    assert torch.equal(actual_centres, expected_centres)
    reference_model.train()
    combined_model.train()

    reference_loss = reference_model.forward_kg(h_graph, sample, t_graph, n_graph, kg_index=0)
    combined_loss = combined_model.forward_kg_combined(combined_graph, sample, kg_index=0)
    torch.testing.assert_close(combined_loss, reference_loss, rtol=1e-6, atol=1e-7)

    reference_loss.backward()
    combined_loss.backward()
    assert_parameter_gradients_close(reference_model, combined_model)

    view_reference_model = LSMGA(args)
    view_combined_model = copy.deepcopy(view_reference_model)
    view_reference_model.train()
    view_combined_model.train()
    reference_output = reference_view_forward(view_reference_model, combined_graph, kg_index=0)
    combined_output = view_combined_model.forward_GNN_embedding(combined_graph, kg_index=0)
    torch.testing.assert_close(combined_output, reference_output, rtol=1e-6, atol=1e-7)
    probe = torch.linspace(-1.0, 1.0, combined_output.numel()).reshape_as(combined_output)
    (reference_output * probe).sum().backward()
    (combined_output * probe).sum().backward()
    assert_parameter_gradients_close(view_reference_model, view_combined_model)

    print("packed_batch_fields_equivalent=True")
    print("vectorized_center_offsets_equivalent=True")
    print("vectorized_filtered_metrics_equivalent=True")
    print("combined_forward_loss_equivalent=True")
    print("combined_backward_gradients_equivalent=True")
    print("fused_views_forward_equivalent=True")
    print("fused_views_backward_gradients_equivalent=True")


if __name__ == "__main__":
    main()
