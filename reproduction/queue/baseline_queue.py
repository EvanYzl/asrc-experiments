#!/usr/bin/env python
"""Serial, resumable runner for the frozen baseline-reproduction suite.

The runner intentionally permits only one child process at a time.  Each job
gets an independent working directory, raw log, resource trace, exit code and
captured artifact path.  Re-running the same command resumes after jobs whose
state is already ``completed``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

import psutil


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def append_runner_log(path: Path, message: str) -> None:
    line = f"{utc_now()} {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def replace_tokens(value: str, run_dir: Path, state: Dict[str, Any]) -> str:
    answer = value.replace("${RUN_DIR}", str(run_dir))
    marker = "${ARTIFACT:"
    while marker in answer:
        start = answer.index(marker)
        end = answer.index("}", start)
        job_id = answer[start + len(marker):end]
        artifact = state["jobs"].get(job_id, {}).get("artifact")
        if not artifact:
            raise RuntimeError(f"artifact for dependency {job_id!r} is unavailable")
        answer = answer[:start] + artifact + answer[end + 1:]
    return answer


def resolve_list(values: Iterable[str], run_dir: Path,
                 state: Dict[str, Any]) -> List[str]:
    return [replace_tokens(str(value), run_dir, state) for value in values]


def process_tree_rss(pid: int) -> int:
    try:
        root = psutil.Process(pid)
        processes = [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def gpu_memory_for_pids(pids: set[int]) -> int:
    if not pids:
        return 0
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    total_mib = 0
    for raw in result.stdout.splitlines():
        parts = [item.strip() for item in raw.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            used = int(parts[1])
        except ValueError:
            continue
        if pid in pids:
            total_mib += used
    return total_mib


def child_pids(pid: int) -> set[int]:
    try:
        root = psutil.Process(pid)
        return {root.pid, *(child.pid for child in root.children(recursive=True))}
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return set()


def monitor_resources(pid: int, path: Path, stop: threading.Event,
                      peaks: Dict[str, float], interval: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "rss_bytes", "gpu_memory_mib",
                         "system_available_bytes"])
        while not stop.is_set():
            rss = process_tree_rss(pid)
            gpu_mib = gpu_memory_for_pids(child_pids(pid))
            available = psutil.virtual_memory().available
            peaks["rss_bytes"] = max(peaks.get("rss_bytes", 0), rss)
            peaks["gpu_memory_mib"] = max(peaks.get("gpu_memory_mib", 0), gpu_mib)
            writer.writerow([utc_now(), rss, gpu_mib, available])
            handle.flush()
            stop.wait(interval)


def latest_directory(root: Path, pattern: str) -> Optional[Path]:
    candidates = [path for path in root.glob(pattern) if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def initial_state(manifest: Dict[str, Any], manifest_path: Path) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "suite": manifest["suite"],
        "manifest": str(manifest_path),
        "status": "pending",
        "pid": os.getpid(),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "current_job": None,
        "jobs": {
            job["id"]: {"status": "pending", "method": job["method"]}
            for job in manifest["jobs"]
        },
    }


def acquire_lock(lock_path: Path) -> None:
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            existing_pid = int(existing["pid"])
        except (ValueError, KeyError, json.JSONDecodeError, OSError):
            existing_pid = -1
        if existing_pid > 0 and psutil.pid_exists(existing_pid):
            raise RuntimeError(f"queue already running as PID {existing_pid}")
    atomic_json(lock_path, {"pid": os.getpid(), "started_at": utc_now()})


def run_job(job: Dict[str, Any], run_dir: Path, state: Dict[str, Any],
            state_path: Path, runner_log: Path,
            global_env: Dict[str, Optional[str]], interval: float) -> None:
    job_id = job["id"]
    job_state = state["jobs"][job_id]
    dependencies = job.get("depends_on", [])
    failed_dependencies = [
        dep for dep in dependencies
        if state["jobs"].get(dep, {}).get("status") != "completed"
    ]
    if failed_dependencies:
        job_state.update({
            "status": "blocked",
            "blocked_by": failed_dependencies,
            "updated_at": utc_now(),
        })
        atomic_json(state_path, state)
        append_runner_log(runner_log,
                          f"BLOCKED {job_id}: dependencies {failed_dependencies}")
        return

    executable = replace_tokens(job["executable"], run_dir, state)
    args = resolve_list(job.get("args", []), run_dir, state)
    cwd = Path(replace_tokens(job["cwd"], run_dir, state))
    cwd.mkdir(parents=True, exist_ok=True)
    for directory in job.get("ensure_dirs", []):
        Path(replace_tokens(directory, run_dir, state)).mkdir(
            parents=True, exist_ok=True)

    env = os.environ.copy()
    for key, value in global_env.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = replace_tokens(value, run_dir, state)
    for key, value in job.get("env", {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = replace_tokens(str(value), run_dir, state)

    command = [executable, *args]
    log_path = run_dir / "logs" / f"{job_id}.log"
    telemetry_path = run_dir / "telemetry" / f"{job_id}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started_monotonic = time.monotonic()
    # A retry reuses the same state entry.  Remove terminal fields from the
    # previous attempt so an in-progress row cannot misleadingly retain an old
    # exit code, duration, peak, or artifact.
    for stale_key in (
        "finished_at", "duration_seconds", "exit_code", "peak_rss_bytes",
        "peak_gpu_memory_mib", "artifact", "artifact_error", "blocked_by",
    ):
        job_state.pop(stale_key, None)
    job_state.update({
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "cwd": str(cwd),
        "command": subprocess.list2cmdline(command),
        "log": str(log_path),
        "telemetry": str(telemetry_path),
    })
    state["status"] = "running"
    state["current_job"] = job_id
    state["updated_at"] = utc_now()
    atomic_json(state_path, state)
    append_runner_log(runner_log, f"START {job_id} ({job['method']})")

    # Preserve completed-round metrics across an interrupted resumable job.
    # Fresh job IDs have no log; retries append an explicit attempt boundary.
    log_mode = "a" if log_path.exists() else "w"
    with log_path.open(log_mode, encoding="utf-8", errors="replace") as log_handle:
        if log_mode == "a":
            log_handle.write("\n# ---- resumed queue attempt ----\n")
        log_handle.write(f"# started_at={job_state['started_at']}\n")
        log_handle.write(f"# cwd={cwd}\n")
        log_handle.write(f"# command={subprocess.list2cmdline(command)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command, cwd=str(cwd), env=env,
            stdout=log_handle, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        job_state["pid"] = process.pid
        atomic_json(state_path, state)
        stop_event = threading.Event()
        peaks: Dict[str, float] = {"rss_bytes": 0, "gpu_memory_mib": 0}
        monitor = threading.Thread(
            target=monitor_resources,
            args=(process.pid, telemetry_path, stop_event, peaks, interval),
            daemon=True,
        )
        monitor.start()
        try:
            exit_code = process.wait()
        finally:
            stop_event.set()
            monitor.join(timeout=max(2.0, interval + 1.0))

    duration = time.monotonic() - started_monotonic
    job_state.update({
        "status": "completed" if exit_code == 0 else "failed",
        "finished_at": utc_now(),
        "updated_at": utc_now(),
        "duration_seconds": round(duration, 3),
        "exit_code": exit_code,
        "peak_rss_bytes": int(peaks["rss_bytes"]),
        "peak_gpu_memory_mib": int(peaks["gpu_memory_mib"]),
    })

    capture = job.get("artifact_capture")
    if exit_code == 0 and capture:
        if capture["type"] != "latest_directory":
            raise ValueError(f"unsupported artifact capture {capture['type']!r}")
        root = Path(replace_tokens(capture["root"], run_dir, state))
        artifact = latest_directory(root, capture["pattern"])
        if artifact is None:
            job_state["status"] = "failed"
            job_state["artifact_error"] = (
                f"no directory matched {capture['pattern']!r} under {root}"
            )
        else:
            job_state["artifact"] = str(artifact)

    state["current_job"] = None
    state["updated_at"] = utc_now()
    atomic_json(state_path, state)
    append_runner_log(
        runner_log,
        f"END {job_id}: {job_state['status']} exit={exit_code} "
        f"duration={duration:.1f}s peak_rss={peaks['rss_bytes']/2**30:.2f}GiB "
        f"peak_gpu={peaks['gpu_memory_mib']:.0f}MiB",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--telemetry-interval", type=float, default=10.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("max_parallel") != 1:
        raise ValueError("baseline queue requires max_parallel=1")

    lock_path = run_dir / "queue.lock.json"
    state_path = run_dir / "state.json"
    runner_log = run_dir / "runner.log"
    acquire_lock(lock_path)
    try:
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["pid"] = os.getpid()
            state["updated_at"] = utc_now()
            for job in manifest["jobs"]:
                state["jobs"].setdefault(
                    job["id"], {"status": "pending", "method": job["method"]})
        else:
            state = initial_state(manifest, manifest_path)
        atomic_json(state_path, state)
        append_runner_log(runner_log,
                          f"QUEUE START pid={os.getpid()} suite={manifest['suite']}")

        for job in manifest["jobs"]:
            previous = state["jobs"][job["id"]].get("status")
            if previous == "completed":
                append_runner_log(runner_log, f"SKIP completed {job['id']}")
                continue
            run_job(job, run_dir, state, state_path, runner_log,
                    manifest.get("global_env", {}), args.telemetry_interval)

        statuses = [entry["status"] for entry in state["jobs"].values()]
        state["status"] = (
            "complete" if all(status == "completed" for status in statuses)
            else "finished_with_failures"
        )
        state["finished_at"] = utc_now()
        state["updated_at"] = utc_now()
        state["current_job"] = None
        atomic_json(state_path, state)
        append_runner_log(runner_log, f"QUEUE END status={state['status']}")
        return 0 if state["status"] == "complete" else 1
    except BaseException as error:
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["status"] = "runner_failed"
                state["runner_error"] = repr(error)
                state["updated_at"] = utc_now()
                atomic_json(state_path, state)
            except Exception:
                pass
        append_runner_log(runner_log, f"QUEUE ERROR {error!r}")
        raise
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
