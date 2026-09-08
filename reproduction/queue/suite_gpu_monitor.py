#!/usr/bin/env python
"""Record whole-GPU telemetry and label each row with the active queue job.

Windows WDDM commonly reports per-process memory as ``N/A``.  Whole-device
memory is therefore retained as an explicitly labelled measurement rather
than silently reporting a false zero.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import time

import psutil


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def gpu_row():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout!r}")
    return fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    state_path = Path(args.state)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if output_path.stat().st_size == 0:
            writer.writerow([
                "timestamp_utc", "current_job", "gpu_used_mib",
                "gpu_total_mib", "gpu_util_percent", "gpu_temp_c",
                "gpu_power_w", "measurement_scope",
            ])
        while True:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                time.sleep(args.interval)
                continue
            runner_pid = int(state.get("pid", -1))
            status = state.get("status")
            if status not in {"pending", "running"} and not psutil.pid_exists(runner_pid):
                break
            try:
                values = gpu_row()
            except Exception:
                values = ["", "", "", "", ""]
            writer.writerow([
                now(), state.get("current_job") or "", *values,
                "whole_device_WDDM",
            ])
            handle.flush()
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
