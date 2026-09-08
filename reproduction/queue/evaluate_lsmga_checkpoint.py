#!/usr/bin/env python
"""Evaluate one LSMGA checkpoint under released and paper filter scopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPRODUCTION_ROOT / "sources" / "LSMGA"
sys.path.insert(0, str(SOURCE_ROOT))

from run_model import parse_args  # noqa: E402
from src.data_loader import MKGDataset  # noqa: E402
from src.lsmga_model import LSMGA  # noqa: E402
from src.utils import get_k_subgraph_list  # noqa: E402
from src.validate import Tester  # noqa: E402


def as_plain_metrics(metrics):
    return {
        lang: {
            "hits1": float(values[0]),
            "hits10": float(values[1]),
            "mrr": float(values[2]),
        }
        for lang, values in metrics.items()
    }


def macro(metrics):
    count = len(metrics)
    return {
        metric: sum(row[metric] for row in metrics.values()) / count
        for metric in ("hits1", "hits10", "mrr")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True,
                        help="prefix used by LSMGA, e.g. .../dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    args_cli = parser.parse_args()

    args = parse_args([
        "--data_path", args_cli.data_path,
        "--dataset", "dbp5l",
        "--device", args_cli.device,
        "--test_batch_size", str(args_cli.test_batch_size),
        "--precompute_batch_size", str(args_cli.precompute_batch_size),
    ])
    args.device = torch.device(args.device)
    args.entity_dim = args.dim
    args.relation_dim = args.dim
    args.langs = ["el", "en", "es", "fr", "ja"]
    args.kgname2idx = {
        language: index for index, language in enumerate(args.langs)
    }

    dataset = MKGDataset(args)
    kg_objects, subgraphs, entity_to_kgs, global_indices = dataset.load_data()
    entity_to_kg_indices = {
        entity: {args.kgname2idx[language] for language in languages}
        for entity, languages in entity_to_kgs.items()
    }
    cache_dir = Path(dataset.data_dir) / "_".join(args.langs)
    for language, kg in kg_objects.items():
        kg.k_subgraph_list = get_k_subgraph_list(
            subgraphs,
            kg.entity_global_index,
            args.kgname2idx[language],
            dataset.num_kgs,
            entity_to_kg_indices,
            dataset.num_entities,
            str(cache_dir),
        )

    args.num_entities = dataset.num_entities
    args.num_relations = dataset.num_relations
    args.num_kgs = dataset.num_kgs
    args.lr = 0.005
    args.margin = 0.5
    model = LSMGA(args).to(args.device)
    checkpoint_path = Path(args_cli.checkpoint).resolve()
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=args.device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise ValueError(f"checkpoint has no model weights: {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()

    validator = Tester(args, kg_objects, model, args.device, dataset.data_dir)
    with torch.no_grad():
        released = as_plain_metrics(validator.test(
            is_val=False,
            is_filtered=True,
        ))
        for kg in kg_objects.values():
            kg.computed_entity_embedding_kg = None
        paper = as_plain_metrics(validator.test(
            is_val=False,
            is_filtered=True,
            filter_splits=("train", "val", "test"),
            keep_ground_truth=True,
        ))

    payload = {
        "schema_version": 1,
        "method": "LSMGA",
        "dataset": "DBP-5L",
        "checkpoint": str(checkpoint_path),
        "released_train_val_filter": {
            "macro": macro(released),
            "per_domain": released,
        },
        "paper_train_val_test_filter": {
            "macro": macro(paper),
            "per_domain": paper,
        },
    }
    output_path = Path(args_cli.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
