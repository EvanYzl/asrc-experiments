"""Package only the frozen Table 2 datasets; never include generated graph caches."""
from pathlib import Path
import csv
import hashlib
import json
import tarfile
from datetime import datetime, timezone

ROOT = Path('G:/zhishitupui')
OUT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

files = {}
raw_count = 0
raw_bytes = 0
for manifest, prefix in [('SHA256SUMS.csv', 'data/raw/dmkgc'),
                         ('WK3L15K_SHA256SUMS.csv', 'data/raw/atransn')]:
    with (ROOT / 'data/manifests' / manifest).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            relative = prefix + '/' + row['relative_path'].replace('\\', '/')
            path = ROOT / relative
            assert path.stat().st_size == int(row['bytes']), relative
            assert digest(path) == row['sha256'] == row['source_sha256'], relative
            files[relative] = path
            raw_count += 1
            raw_bytes += path.stat().st_size
assert raw_count == 109 and raw_bytes == 102133997
for path in (ROOT / 'data/manifests').glob('*.csv'):
    files[path.relative_to(ROOT).as_posix()] = path
for relative in ['data/DATASETS.md', 'data/SOURCE_PROVENANCE.json']:
    files[relative] = ROOT / relative
for path in (ROOT / 'reproduction/strict_baselines/data_manifests').glob('*.json'):
    files[path.relative_to(ROOT).as_posix()] = path
for name in ['install_environment.sh', 'verify_data.py', 'verify_environment.py']:
    files['deployment/' + name] = OUT / name
inventory = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'raw_files': raw_count, 'raw_bytes': raw_bytes,
    'files': [{'path': rel, 'bytes': p.stat().st_size, 'sha256': digest(p)}
              for rel, p in sorted(files.items())],
    'datasets': ['DBP-5L', 'E-PKG', 'DWY (shared DMKGC split)', 'WK3l-15k EN_F -> FR'],
    'frozen_split_manifests': 16,
    'note': 'Original Windows manifest bytes and split indices are preserved. Normalize backslashes only when resolving their relative file paths on Linux.'
}
inventory_path = OUT / 'transfer_manifest.json'
inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
archive = OUT / 'dataset_bundle.tar.gz'
with tarfile.open(archive, 'w:gz', compresslevel=1) as tf:
    for relative, path in sorted(files.items()):
        tf.add(path, arcname=relative, recursive=False)
    tf.add(inventory_path, arcname='deployment/transfer_manifest.json', recursive=False)
report = {'archive': str(archive), 'archive_bytes': archive.stat().st_size,
          'archive_sha256': digest(archive), 'raw_files': raw_count,
          'raw_bytes': raw_bytes, 'inventory_files': len(files), 'frozen_split_manifests': 16}
(OUT / 'bundle_receipt.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
