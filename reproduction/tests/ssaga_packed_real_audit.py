"""Load and validate the full DBP-5L SS-AGA packed graph state."""

from __future__ import annotations

import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import psutil
import torch

SOURCE = Path(__file__).resolve().parents[1] / "sources" / "SS-AGA"
sys.path.insert(0, str(SOURCE))

from src.data_loader_new import ParseData
from src.utils import PackedSubgraphList, nodes_to_graph


def main() -> None:
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    args = SimpleNamespace(
        data_path="G:/zhishitupui/data/raw/dmkgc/dataset",
        dataset="dbp5l",
        target_language="ja",
        preserved_ratio=0.1,
        num_hop=2,
        k=10,
        MAX_SAM=10_000_000_000,
        device=torch.device("cpu"),
    )
    dataset = ParseData(args)
    kg_objects, seeds_masked, seeds_all, features = dataset.load_data()
    for language, kg in kg_objects.items():
        assert isinstance(kg.subgraph_list_kg, PackedSubgraphList)
        assert isinstance(kg.subgraph_list_align, PackedSubgraphList)
        assert len(kg.subgraph_list_kg) == kg.num_entity
        assert len(kg.subgraph_list_align) == kg.num_entity
        sample = torch.tensor([0, kg.num_entity // 2, kg.num_entity - 1])
        batch = nodes_to_graph(kg.subgraph_list_kg, sample)
        assert batch.edge_index.shape[1] == batch.edge_attr.shape[0]
        print(
            f"{language}: entities={kg.num_entity} "
            f"kg_nodes={kg.subgraph_list_kg.x.numel()} "
            f"kg_edges={kg.subgraph_list_kg.edge_attr.numel()} "
            f"align_nodes={kg.subgraph_list_align.x.numel()} "
            f"align_edges={kg.subgraph_list_align.edge_attr.numel()}"
        )
    memory = psutil.Process(os.getpid()).memory_info()
    private = getattr(memory, "private", None)
    print(
        f"features_shape={features.shape} seeds={len(seeds_all)} "
        f"rss_gib={memory.rss / 2**30:.3f} "
        f"private_gib={private / 2**30:.3f}" if private is not None else
        f"features_shape={features.shape} seeds={len(seeds_all)} "
        f"rss_gib={memory.rss / 2**30:.3f}"
    )
    print("ssaga_full_packed_audit=True")


if __name__ == "__main__":
    main()
