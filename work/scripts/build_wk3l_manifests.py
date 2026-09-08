from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PROJECT = Path(r"G:\zhishitupui")
SOURCE = PROJECT / "work" / "ATransN-review" / "data"
DEST = PROJECT / "data" / "raw" / "atransn"
MANIFESTS = PROJECT / "data" / "manifests"
KG_DIRS = {
    "WK3l-15k_EN_F": "teacher_en",
    "WK3l-15k_FR": "target_fr",
}
ALIGNMENT = Path("SHARED/wk3l-15k_en_f_fr_aligned_entity_id.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line:
                yield line


def read_dictionary(path: Path) -> tuple[set[int], int]:
    ids: set[int] = set()
    invalid = 0
    for line in nonempty_lines(path):
        try:
            _, raw_id = line.rsplit("\t", 1)
            ids.add(int(raw_id))
        except (ValueError, IndexError):
            invalid += 1
    return ids, invalid


def read_triples(path: Path) -> tuple[list[tuple[int, int, int]], int]:
    triples: list[tuple[int, int, int]] = []
    invalid = 0
    for line in nonempty_lines(path):
        parts = line.split("\t")
        if len(parts) != 3:
            invalid += 1
            continue
        try:
            triples.append(tuple(map(int, parts)))
        except ValueError:
            invalid += 1
    return triples, invalid


def build_hash_manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    required = []
    for kg_dir in KG_DIRS:
        required.extend(
            Path(kg_dir) / name
            for name in (
                "entity_dict.txt",
                "relation_dict.txt",
                "train_triple_id.txt",
                "valid_triple_id.txt",
                "test_triple_id.txt",
                "triple_id.txt",
            )
        )
    required.append(ALIGNMENT)
    for relative in required:
        source_path = SOURCE / relative
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
    for kg_dir, role in KG_DIRS.items():
        root = DEST / kg_dir
        entity_ids, bad_entities = read_dictionary(root / "entity_dict.txt")
        relation_ids, bad_relations = read_dictionary(root / "relation_dict.txt")
        split_triples: dict[str, list[tuple[int, int, int]]] = {}
        invalid_triples = 0
        for split, filename in (
            ("train", "train_triple_id.txt"),
            ("validation", "valid_triple_id.txt"),
            ("test", "test_triple_id.txt"),
            ("all", "triple_id.txt"),
        ):
            values, invalid = read_triples(root / filename)
            split_triples[split] = values
            invalid_triples += invalid
        all_split = split_triples["train"] + split_triples["validation"] + split_triples["test"]
        observed = all_split
        entity_bounds_ok = all(h in entity_ids and t in entity_ids for h, _, t in observed)
        relation_bounds_ok = all(r in relation_ids for _, r, _ in observed)
        rows.append(
            {
                "dataset": "WK3l-15k",
                "kg_directory": kg_dir,
                "role": role,
                "entities": len(entity_ids),
                "relations": len(relation_ids),
                "train": len(split_triples["train"]),
                "validation": len(split_triples["validation"]),
                "test": len(split_triples["test"]),
                "all_triples": len(split_triples["all"]),
                "invalid_dictionary_rows": bad_entities + bad_relations,
                "invalid_triple_rows": invalid_triples,
                "entity_id_bounds_ok": entity_bounds_ok,
                "relation_id_bounds_ok": relation_bounds_ok,
                "split_multiset_matches_full": sorted(all_split) == sorted(split_triples["all"]),
            }
        )
    return rows


def build_alignment_stats() -> list[dict[str, object]]:
    source_ids, _ = read_dictionary(DEST / "WK3l-15k_EN_F" / "entity_dict.txt")
    target_ids, _ = read_dictionary(DEST / "WK3l-15k_FR" / "entity_dict.txt")
    pairs: list[tuple[int, int]] = []
    invalid = 0
    for line in nonempty_lines(DEST / ALIGNMENT):
        parts = line.split("\t")
        if len(parts) != 2:
            invalid += 1
            continue
        try:
            pairs.append((int(parts[0]), int(parts[1])))
        except ValueError:
            invalid += 1
    source_unique = {left for left, _ in pairs}
    target_unique = {right for _, right in pairs}
    return [
        {
            "dataset": "WK3l-15k",
            "pair": "EN_F-FR",
            "rows": len(pairs),
            "unique_source_entities": len(source_unique),
            "unique_target_entities": len(target_unique),
            "source_alignment_pct": 100 * len(source_unique) / len(source_ids),
            "target_alignment_pct": 100 * len(target_unique) / len(target_ids),
            "invalid_rows": invalid,
            "source_id_bounds_ok": source_unique <= source_ids,
            "target_id_bounds_ok": target_unique <= target_ids,
        }
    ]


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    path = MANIFESTS / name
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    hashes = build_hash_manifest()
    stats = build_dataset_stats()
    alignments = build_alignment_stats()
    write_csv("WK3L15K_SHA256SUMS.csv", hashes)
    write_csv("WK3L15K_STATS.csv", stats)
    write_csv("WK3L15K_ALIGNMENT_STATS.csv", alignments)
    print(f"files={len(hashes)} hash_mismatches={sum(not row['source_copy_match'] for row in hashes)}")
    print(
        "kgs={} invalid_dictionary_rows={} invalid_triple_rows={} bad_entity_bounds={} "
        "bad_relation_bounds={} split_full_mismatches={}".format(
            len(stats),
            sum(int(row["invalid_dictionary_rows"]) for row in stats),
            sum(int(row["invalid_triple_rows"]) for row in stats),
            sum(not bool(row["entity_id_bounds_ok"]) for row in stats),
            sum(not bool(row["relation_id_bounds_ok"]) for row in stats),
            sum(not bool(row["split_multiset_matches_full"]) for row in stats),
        )
    )
    print(
        f"alignment_rows={alignments[0]['rows']} invalid_alignments={alignments[0]['invalid_rows']} "
        f"source_pct={alignments[0]['source_alignment_pct']:.2f} "
        f"target_pct={alignments[0]['target_alignment_pct']:.2f}"
    )


if __name__ == "__main__":
    main()
