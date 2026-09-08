from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PROJECT = Path(r"G:\zhishitupui")
SOURCE = PROJECT / "work" / "DMKGC-review"
DEST = PROJECT / "data" / "raw" / "dmkgc"
MANIFESTS = PROJECT / "data" / "manifests"
DATASETS = ("datasetdbp5l", "datasetdepkg", "datasetdwy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line:
                yield line


def build_hash_manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for source_path in sorted((SOURCE / dataset).rglob("*")):
            if not source_path.is_file():
                continue
            relative = source_path.relative_to(SOURCE)
            dest_path = DEST / relative
            source_hash = sha256(source_path)
            dest_hash = sha256(dest_path)
            rows.append(
                {
                    "relative_path": relative.as_posix(),
                    "bytes": dest_path.stat().st_size,
                    "sha256": dest_hash,
                    "source_sha256": source_hash,
                    "source_copy_match": source_hash == dest_hash,
                }
            )
    return rows


def build_dataset_stats() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        dataset_path = DEST / dataset
        for train_path in sorted((dataset_path / "kg").glob("*-train.tsv")):
            kg = train_path.name.removesuffix("-train.tsv")
            entity_count = sum(1 for _ in nonempty_lines(dataset_path / "entity" / f"{kg}.tsv"))
            relation_ids: set[int] = set()
            split_counts: dict[str, int] = {}
            invalid_rows = 0
            max_entity_id = -1
            for split in ("train", "val", "test"):
                count = 0
                for line in nonempty_lines(dataset_path / "kg" / f"{kg}-{split}.tsv"):
                    count += 1
                    parts = line.split("\t")
                    if len(parts) != 3:
                        invalid_rows += 1
                        continue
                    try:
                        head, relation, tail = map(int, parts)
                    except ValueError:
                        invalid_rows += 1
                        continue
                    relation_ids.add(relation)
                    max_entity_id = max(max_entity_id, head, tail)
                split_counts[split] = count
            rows.append(
                {
                    "dataset": dataset,
                    "kg": kg,
                    "entities": entity_count,
                    "relation_ids_observed": len(relation_ids),
                    "train": split_counts["train"],
                    "validation": split_counts["val"],
                    "test": split_counts["test"],
                    "invalid_triple_rows": invalid_rows,
                    "max_entity_id": max_entity_id,
                    "entity_id_bound_ok": max_entity_id < entity_count,
                }
            )
    return rows


def build_alignment_stats() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for path in sorted((DEST / dataset / "seed_alignlinks").glob("*.tsv")):
            count = 0
            invalid_rows = 0
            for line in nonempty_lines(path):
                count += 1
                parts = line.split("\t")
                if len(parts) != 2:
                    invalid_rows += 1
                    continue
                try:
                    float(parts[0])
                    float(parts[1])
                except ValueError:
                    invalid_rows += 1
            rows.append(
                {
                    "dataset": dataset,
                    "pair": path.stem,
                    "rows": count,
                    "invalid_rows": invalid_rows,
                }
            )
    return rows


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    path = MANIFESTS / name
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    hash_rows = build_hash_manifest()
    stat_rows = build_dataset_stats()
    alignment_rows = build_alignment_stats()
    write_csv("SHA256SUMS.csv", hash_rows)
    write_csv("DATASET_STATS.csv", stat_rows)
    write_csv("ALIGNMENT_STATS.csv", alignment_rows)
    mismatches = sum(not bool(row["source_copy_match"]) for row in hash_rows)
    bad_triples = sum(int(row["invalid_triple_rows"]) for row in stat_rows)
    bad_bounds = sum(not bool(row["entity_id_bound_ok"]) for row in stat_rows)
    bad_alignments = sum(int(row["invalid_rows"]) for row in alignment_rows)
    print(f"files={len(hash_rows)} hash_mismatches={mismatches}")
    print(f"kg_splits={len(stat_rows)} invalid_triples={bad_triples} bad_entity_bounds={bad_bounds}")
    print(f"alignment_files={len(alignment_rows)} invalid_alignments={bad_alignments}")


if __name__ == "__main__":
    main()
