#!/usr/bin/env python
"""Fail-closed verification for the ten-baseline formal reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def require(condition: bool, message: str, failures: List[str]) -> None:
    if not condition:
        failures.append(message)


def nonempty_file(raw_path: Any) -> bool:
    if not raw_path:
        return False
    path = Path(raw_path)
    return path.is_file() and path.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    state_path = Path(args.state).resolve()
    results_path = Path(args.results).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    results_payload = json.loads(results_path.read_text(encoding="utf-8"))
    result_by_id = {
        row["job_id"]: row for row in results_payload.get("results", [])
    }
    failures: List[str] = []
    checks: List[Dict[str, Any]] = []

    require(state.get("status") == "complete",
            "queue state is not complete", failures)
    baseline_jobs = [job for job in manifest["jobs"]
                     if not job.get("support_job")]
    require(len(baseline_jobs) == 10,
            "manifest does not contain exactly ten baselines", failures)

    for job in manifest["jobs"]:
        job_id = job["id"]
        entry = state.get("jobs", {}).get(job_id, {})
        local_failures: List[str] = []
        require(entry.get("status") == "completed",
                f"{job_id}: status is not completed", local_failures)
        require(entry.get("exit_code") == 0,
                f"{job_id}: exit_code is not zero", local_failures)
        require(nonempty_file(entry.get("log")),
                f"{job_id}: formal log is absent or empty", local_failures)
        if not job.get("support_job"):
            row = result_by_id.get(job_id)
            require(row is not None,
                    f"{job_id}: missing from collected results", local_failures)
            if row:
                require(row.get("status") == "completed",
                        f"{job_id}: collected status is not completed",
                        local_failures)
                if job["method"] == "KEnS (TransE)":
                    for metric in ("hits1", "hits3", "hits10"):
                        require(row.get(metric) is not None,
                                f"{job_id}: missing {metric}", local_failures)
                else:
                    for metric in ("mrr", "hits1", "hits10"):
                        require(row.get(metric) is not None,
                                f"{job_id}: missing {metric}", local_failures)
        if job.get("artifact_capture"):
            artifact = entry.get("artifact")
            require(bool(artifact) and Path(artifact).is_dir(),
                    f"{job_id}: captured artifact directory is absent",
                    local_failures)
        checks.append({
            "job_id": job_id,
            "method": job["method"],
            "support_job": bool(job.get("support_job")),
            "passed": not local_failures,
            "failures": local_failures,
        })
        failures.extend(local_failures)

    run_dir = state_path.parent
    checkpoint_options = {
        "LSMGA": [run_dir / "work/lsmga/resume_latest.ckpt"],
        "DMKGC": [run_dir / "work/dmkgc/resume_latest.ckpt"],
        "IMKGC": [run_dir / "work/imkgc/resume_latest.ckpt"],
        "AlignKGC": [
            run_dir / "work/alignkgc/resume_latest.ckpt",
            run_dir / "artifacts/alignkgc/alignkgc_seed41_resume.ckpt",
        ],
    }
    for method, options in checkpoint_options.items():
        require(any(path.is_file() and path.stat().st_size > 0
                    for path in options),
                f"{method}: no non-empty formal resume checkpoint", failures)

    align_entry = state.get("jobs", {}).get(
        "alignkgc_dbp5l_20_20_s41", {})
    align_artifact = align_entry.get("artifact")
    if align_artifact:
        require(nonempty_file(Path(align_artifact) / "commit.json"),
                "AlignKGC: commit.json absent", failures)
        require(nonempty_file(Path(align_artifact) / "best_valid_model.pt"),
                "AlignKGC: best_valid_model.pt absent", failures)

    report = {
        "schema_version": 1,
        "passed": not failures,
        "baseline_count": len(baseline_jobs),
        "queue_status": state.get("status"),
        "checks": checks,
        "failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
