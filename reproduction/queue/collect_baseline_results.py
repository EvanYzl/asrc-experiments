#!/usr/bin/env python
"""Collect completed baseline metrics without treating smoke values as results."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any, Dict, Iterable, Optional


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def last_float(pattern: str, text: str) -> Optional[float]:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def read_log(entry: Dict[str, Any]) -> str:
    path = entry.get("log")
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_kge(text: str) -> Dict[str, Optional[float]]:
    return {
        "mrr": last_float(r"test-best MRR at step \d+:\s+([0-9.eE+-]+)", text),
        "hits1": last_float(r"test-best HITS@1 at step \d+:\s+([0-9.eE+-]+)", text),
        "hits3": last_float(r"test-best HITS@3 at step \d+:\s+([0-9.eE+-]+)", text),
        "hits10": last_float(r"test-best HITS@10 at step \d+:\s+([0-9.eE+-]+)", text),
    }


def parse_best_multidomain(text: str) -> Dict[str, Any]:
    metrics: Dict[str, Dict[str, float]] = {}
    pattern = re.compile(
        # Logger prefixes every repository summary with a timestamp and level.
        # Match the summary token after that prefix, while deliberately excluding
        # the bracketed per-domain evaluation rows (``[el] Test: ...``).
        r"^[^\r\n]*?\b([a-z]{2}) filterd:\s*([0-9.eE+-]+),\s*"
        r"([0-9.eE+-]+),\s*([0-9.eE+-]+)$",
        flags=re.MULTILINE,
    )
    for lang, hits1, hits10, mrr in pattern.findall(text):
        metrics[lang] = {
            "hits1": float(hits1), "hits10": float(hits10), "mrr": float(mrr)
        }
    if not metrics:
        return {"mrr": None, "hits1": None, "hits10": None, "per_domain": {}}
    return {
        "mrr": mean(item["mrr"] for item in metrics.values()),
        "hits1": mean(item["hits1"] for item in metrics.values()),
        "hits10": mean(item["hits10"] for item in metrics.values()),
        "per_domain": metrics,
        "selected_round": int(last_float(r"best epoch:\s*(\d+)", text) or -1),
        "selection": "repository-best-mean-test-mrr",
    }


def parse_lsmga(text: str) -> Dict[str, Any]:
    val_pattern = re.compile(
        r"\[([a-z]{2})\]Val:\s*([0-9.eE+-]+),\s*"
        r"([0-9.eE+-]+),\s*([0-9.eE+-]+)",
    )
    test_pattern = re.compile(
        r"\[([a-z]{2})\]Test:\s+filterd:\s*([0-9.eE+-]+),\s*"
        r"([0-9.eE+-]+),\s*([0-9.eE+-]+)",
    )
    round_markers = list(re.finditer(
        r"(?m)^.*?\bINFO\s+Round:\s*(\d+)\s*$", text,
    ))
    candidates = []
    for index, marker in enumerate(round_markers):
        block_end = round_markers[index + 1].start() if index + 1 < len(round_markers) else len(text)
        block = text[marker.end():block_end]
        validation = {
            lang: {"hits1": float(h1), "hits10": float(h10), "mrr": float(mrr)}
            for lang, h1, h10, mrr in val_pattern.findall(block)
        }
        test = {
            lang: {"hits1": float(h1), "hits10": float(h10), "mrr": float(mrr)}
            for lang, h1, h10, mrr in test_pattern.findall(block)
        }
        if validation and test and validation.keys() == test.keys():
            candidates.append({
                "round": int(marker.group(1)),
                "validation_mrr": mean(item["mrr"] for item in validation.values()),
                "validation": validation,
                "test": test,
            })
    if not candidates:
        return {"mrr": None, "hits1": None, "hits10": None, "per_domain": {}}
    selected = max(candidates, key=lambda item: (item["validation_mrr"], -item["round"]))
    metrics = selected["test"]
    return {
        "mrr": mean(item["mrr"] for item in metrics.values()),
        "hits1": mean(item["hits1"] for item in metrics.values()),
        "hits10": mean(item["hits10"] for item in metrics.values()),
        "per_domain": metrics,
        "selected_round": selected["round"],
        "validation_mrr": selected["validation_mrr"],
        "validation_per_domain": selected["validation"],
        "selection": "best-mean-validation-mrr",
    }


def parse_ssaga(text: str) -> Dict[str, Optional[float]]:
    matches = re.findall(
        r"BestVal! Epoch\s+(\d+).*?Best mrr\s+([0-9.eE+-]+)\|\s*hits1\s+"
        r"([0-9.eE+-]+)\|\s*hits10\s+([0-9.eE+-]+)",
        text,
    )
    if not matches:
        return {"mrr": None, "hits1": None, "hits10": None}
    epoch, mrr, hits1, hits10 = matches[-1]
    return {
        "mrr": float(mrr), "hits1": float(hits1), "hits10": float(hits10),
        "selected_round": int(epoch), "selection": "best-validation-checkpoint",
    }


def parse_alignkgc(entry: Dict[str, Any]) -> Dict[str, Any]:
    artifact = entry.get("artifact")
    if not artifact:
        return {"mrr": None, "hits1": None, "hits10": None, "per_domain": {}}
    commit_path = Path(artifact) / "commit.json"
    try:
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"mrr": None, "hits1": None, "hits10": None, "per_domain": {}}
    per_domain: Dict[str, Dict[str, float]] = {}
    for lang in ("el", "en", "es", "fr", "ja"):
        score = commit.get("test_scores", {}).get(f"test_{lang}.txt", {}).get("e2")
        if score:
            per_domain[lang] = {
                "mrr": float(score["mrr"]),
                "hits1": float(score["hits1"]),
                "hits10": float(score["hits10"]),
            }
    if not per_domain:
        return {"mrr": None, "hits1": None, "hits10": None, "per_domain": {}}
    validation = commit.get("valid_score", {}).get("e2", {})
    return {
        "mrr": mean(item["mrr"] for item in per_domain.values()),
        "hits1": mean(item["hits1"] for item in per_domain.values()),
        "hits10": mean(item["hits10"] for item in per_domain.values()),
        "per_domain": per_domain,
        "validation_mrr": (
            float(validation["mrr"]) if validation.get("mrr") is not None
            else None
        ),
        "selection": "best-validation-checkpoint",
    }


def metrics_for(job: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    method = job["method"]
    text = read_log(entry)
    if method in {"TransE", "DistMult", "RotatE", "ATransN"}:
        return parse_kge(text)
    if method == "SS-AGA":
        return parse_ssaga(text)
    if method == "LSMGA":
        return parse_lsmga(text)
    if method in {"DMKGC", "IMKGC"}:
        return parse_best_multidomain(text)
    if method == "AlignKGC":
        return parse_alignkgc(entry)
    return {"mrr": None, "hits1": None, "hits10": None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    rows = []
    for job in manifest["jobs"]:
        if job.get("support_job"):
            continue
        entry = state["jobs"].get(job["id"], {"status": "missing"})
        parsed = metrics_for(job, entry) if entry.get("status") == "completed" else {
            "mrr": None, "hits1": None, "hits10": None
        }
        rows.append({
            "baseline_index": job["baseline_index"],
            "job_id": job["id"],
            "method": job["method"],
            "dataset": job["dataset"],
            "protocol": job["protocol"],
            "status": entry.get("status", "missing"),
            "exit_code": entry.get("exit_code"),
            "duration_seconds": entry.get("duration_seconds"),
            "peak_rss_bytes": entry.get("peak_rss_bytes"),
            "mrr": parsed.get("mrr"),
            "hits1": parsed.get("hits1"),
            "hits3": parsed.get("hits3"),
            "hits10": parsed.get("hits10"),
            "per_domain": parsed.get("per_domain"),
            "selected_round": parsed.get("selected_round"),
            "validation_mrr": parsed.get("validation_mrr"),
            "selection": parsed.get("selection"),
            "log": entry.get("log"),
            "artifact": entry.get("artifact"),
        })
    rows.sort(key=lambda item: item["baseline_index"])
    payload = {
        "schema_version": 1,
        "generated_at": now(),
        "queue_status": state.get("status"),
        "warning": "Only status=completed rows are formal runs; do not backfill from smoke results.",
        "results": rows,
    }
    json_path = Path(args.json_output)
    csv_path = Path(args.csv_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "baseline_index", "job_id", "method", "dataset", "protocol",
            "status", "exit_code", "duration_seconds", "peak_rss_bytes",
            "mrr", "hits1", "hits3", "hits10", "selected_round",
            "validation_mrr", "selection", "log", "artifact",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
