"""CPU forward/backward smoke for the patched DMKGC or IMKGC model."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv[1]).resolve()
METHOD = SOURCE.name
sys.path.insert(0, str(SOURCE))

from run_model import parse_args  # noqa: E402
from src.utils import nodes_to_k_graph  # noqa: E402


CACHE = (
    ROOT.parent / "data" / "raw" / "dmkgc" / "datasetdbp5l"
    / "el_en_es_fr_ja" / "kg0_k_subgraph_list.graph"
)


def main():
    torch.manual_seed(2020)
    packed = torch.load(CACHE, map_location="cpu")
    args = parse_args([])
    args.device = torch.device("cpu")
    args.num_kgs = int(packed["x"].shape[1])
    args.num_entities = int(packed["x"].max()) + 1
    args.num_relations = int(packed["edge_attr"].max()) + 1
    args.dim = 16
    args.entity_dim = 16
    args.relation_dim = 16
    args.encoder_hdim_gnn = 16
    args.n_heads = 1
    args.n_layers_gnn = 2
    args.dropout = 0.0
    args.batch_size = 4
    args.pretrain_dim = 16

    if METHOD == "DMKGC":
        from src.dmkgc import DMKGC
        args.n_steps = 4
        args.n_sampling_step = 4
        model = DMKGC(args)
    elif METHOD == "IMKGC":
        from src.imkgc import IMKGC
        args.reason_step = 2
        args.codebook_ratio = 1.0
        model = IMKGC(args)
    else:
        raise ValueError(METHOD)

    h_rows = torch.tensor([0, 1, 2, 3])
    t_rows = torch.tensor([4, 5, 6, 7])
    n_rows = torch.tensor([8, 9, 10, 11])
    h_graph = nodes_to_k_graph(packed, h_rows, "cpu")
    t_graph = nodes_to_k_graph(packed, t_rows, "cpu")
    n_graph = nodes_to_k_graph(packed, n_rows, "cpu")
    sample = torch.stack((h_rows, torch.zeros_like(h_rows), t_rows), dim=1)

    model.train()
    result = model.forward_kg(h_graph, sample, t_graph, n_graph, kg_index=0)
    if METHOD == "DMKGC":
        loss = result["kg_loss"] + 0.01 * result["loss_recon"] + 0.001 * result["loss_reg"]
    else:
        loss = (
            result["kg_loss"] + 0.0001 * result["kld_loss"]
            + 0.1 * result["kg_cur_loss"] + 0.05 * result["kg_assist_loss"]
            + 0.005 * result["info_contrastive"] + result["vq_loss"]
        )
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    if hasattr(model, "forward_kg_combined"):
        model.zero_grad(set_to_none=True)
        combined_graph = nodes_to_k_graph(
            packed, torch.cat((h_rows, t_rows, n_rows)), "cpu"
        )
        torch.manual_seed(2021)
        combined_result = model.forward_kg_combined(
            combined_graph, sample, kg_index=0
        )
        if METHOD == "DMKGC":
            combined_loss = (
                combined_result["kg_loss"]
                + 0.01 * combined_result["loss_recon"]
                + 0.001 * combined_result["loss_reg"]
            )
        else:
            combined_loss = (
                combined_result["kg_loss"]
                + 0.0001 * combined_result["kld_loss"]
                + 0.1 * combined_result["kg_cur_loss"]
                + 0.05 * combined_result["kg_assist_loss"]
                + 0.005 * combined_result["info_contrastive"]
                + combined_result["vq_loss"]
            )
        assert torch.isfinite(combined_loss).all()
        combined_loss.mean().backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )

    model.eval()
    with torch.no_grad():
        embeddings, _ = model.forward_GNN_embedding(h_graph, kg_index=0)
    assert embeddings.shape == (4, 16)
    assert torch.isfinite(embeddings).all()
    print(f"source={METHOD}")
    print("full_model_forward_backward_finite=True")
    if hasattr(model, "forward_kg_combined"):
        print("combined_model_forward_backward_finite=True")
    print("evaluation_embedding_finite=True")


if __name__ == "__main__":
    main()
