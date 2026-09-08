#!/usr/bin/env python
"""Run both official KEnS stages as one resumable queue job."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--knowledge-model", required=True,
                        choices=("transe", "rotate"))
    parser.add_argument("--knn-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--use-default", action="store_true")
    parser.add_argument("--dim", type=int)
    args = parser.parse_args()

    run_command = [
        sys.executable,
        str(args.source_root / "run.py"),
        "--target_language", args.target_language,
        "--knowledge_model", args.knowledge_model,
        "--knn_batch_size", str(args.knn_batch_size),
        "--seed", str(args.seed),
    ]
    if args.use_default:
        run_command.append("--use_default")
        dimension = 300 if args.knowledge_model == "transe" else 400
    else:
        if args.dim is None:
            parser.error("--dim is required unless --use-default is set")
        dimension = args.dim
        run_command.extend(("--dim", str(dimension)))

    train = subprocess.run(run_command, check=False)
    if train.returncode != 0:
        return train.returncode

    model_dir = (
        Path.cwd() / "trained_model" /
        f"kens-{args.knowledge_model}-{dimension}" /
        args.target_language
    )
    test_command = [
        sys.executable,
        str(args.source_root / "test.py"),
        "--target_language", args.target_language,
        "--knowledge_model", args.knowledge_model,
        "--model_dir", str(model_dir),
        "--dim", str(dimension),
    ]
    test = subprocess.run(test_command, check=False)
    return test.returncode


if __name__ == "__main__":
    raise SystemExit(main())
