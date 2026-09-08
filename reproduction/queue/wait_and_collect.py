#!/usr/bin/env python
"""Wait for the queue to terminate, then emit formal result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--collector", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    state_path = Path(args.state)
    while True:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            time.sleep(args.interval)
            continue
        if state.get("status") not in {"pending", "running"}:
            break
        time.sleep(args.interval)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.call([
        sys.executable, args.collector,
        "--manifest", args.manifest,
        "--state", args.state,
        "--json-output", str(output_dir / "formal_results.json"),
        "--csv-output", str(output_dir / "formal_results.csv"),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
