"""Exercise every full DBP-5L self-learning pair from a trained checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--knn-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2020)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    source = Path(cli.source_root).resolve()
    sys.path.insert(0, str(source))

    from run_model import parse_args as parse_model_args, set_args
    from src.data_loader_new import ParseData
    from src.ssaga_model import SSAGA

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)

    args = parse_model_args([
        "--target_language", "ja",
        "--use_default",
        "--data_path", cli.data_path,
        "--dataset", "dbp5l",
        "--seed", str(cli.seed),
        "--precompute_batch_size", "16",
        "--knn_batch_size", str(cli.knn_batch_size),
        "--self_learning_device", "cpu",
    ])
    set_args(args)
    args.device = torch.device("cuda:0")
    args.entity_dim = args.dim
    args.relation_dim = args.entity_dim

    dataset = ParseData(args)
    kg_objects, _, seeds_all, features = dataset.load_data()
    args.num_relations = dataset.num_relations
    args.num_entities = dataset.num_entities
    args.num_kgs = dataset.num_kgs
    del dataset

    model = SSAGA(
        args, features, args.num_relations, args.num_entities, args.num_kgs
    ).to(args.device)
    del features
    checkpoint = torch.load(
        cli.checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint
    model.eval()

    records = []
    with torch.no_grad():
        for kg0_name, kg1_name in seeds_all:
            started = time.perf_counter()
            found = model.extend_seed_align_links(
                kg_objects[kg0_name],
                kg_objects[kg1_name],
                seeds_all[(kg0_name, kg1_name)],
                args.device,
                args.k_csls,
            )
            count = 0 if found is None else len(found)
            records.append({
                "pair": f"{kg0_name}-{kg1_name}",
                "links": count,
                "seconds": round(time.perf_counter() - started, 3),
            })
            assert all(
                kg.computed_entity_embedidng_align is None
                or kg.computed_entity_embedidng_align.device.type == "cpu"
                for kg in kg_objects.values()
            )
            print(json.dumps(records[-1], sort_keys=True), flush=True)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    print(json.dumps({
        "ssaga_full_self_learning_audit": True,
        "pairs": len(records),
        "records": records,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
