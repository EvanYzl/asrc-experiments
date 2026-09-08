"""Verify transfer, all source files and the preserved train/validation/test protocol."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def partition(triples):
    groups = {}
    for i, triple in enumerate(triples):
        groups.setdefault(tuple(map(int, triple)), []).append(i)
    by_relation = {}
    for triple in groups:
        by_relation.setdefault(triple[1], []).append(triple)
    select, cert = [], []
    total_select = total_cert = 0
    for relation, keys in sorted(by_relation.items()):
        keys.sort(key=lambda k: hashlib.sha256(f'20260905:{k}'.encode()).digest())
        a = b = 0
        for key in keys:
            indices = groups[key]
            choose_select = a < b or (a == b and total_select <= total_cert)
            if choose_select:
                select.extend(indices); a += len(indices); total_select += len(indices)
            else:
                cert.extend(indices); b += len(indices); total_cert += len(indices)
    return sorted(select), sorted(cert)

inventory = json.loads((ROOT / 'deployment/transfer_manifest.json').read_text())
for row in inventory['files']:
    path = ROOT / row['path']
    assert path.stat().st_size == row['bytes'], row['path']
    assert digest(path) == row['sha256'], row['path']
domains = []
for manifest in sorted((ROOT / 'reproduction/strict_baselines/data_manifests').glob('*.json')):
    obj = json.loads(manifest.read_text())
    for relative, expected in obj['files'].items():
        assert digest(ROOT / relative.replace('\\', '/')) == expected, relative
    ds, kg = obj['dataset'], obj['kg']
    if ds == 'wk3l':
        folder = ROOT / 'data/raw/atransn' / ('WK3l-15k_EN_F' if kg == 'en' else 'WK3l-15k_FR')
        paths = {s: folder / f'{s}_triple_id.txt' for s in ['train', 'valid', 'test']}
    else:
        folder = ROOT / f'data/raw/dmkgc/dataset{ds}/kg'
        paths = {s: folder / f'{kg}-{v}.tsv' for s, v in [('train','train'),('valid','val'),('test','test')]}
    arrays = {}
    for split, path in paths.items():
        a = np.loadtxt(path, dtype=np.int64, delimiter='\t', ndmin=2)
        assert a.shape == (obj['counts'][split], 3) and a.min() >= 0
        assert a[:,[0,2]].max() < obj['entities']
        assert a[:,1].max() < obj['relations_dictionary']
        arrays[split] = a
    select, cert = partition(arrays['valid'])
    assert select == obj['val_select_indices'] and cert == obj['val_cert_indices']
    assert not set(select).intersection(cert)
    assert len(select) + len(cert) == len(arrays['valid'])
    domains.append({'dataset':ds, 'kg':kg, 'counts':obj['counts'],
                    'original_dataset_hash':obj['dataset_hash'], 'split_manifest_sha256':digest(manifest)})
assert len(domains) == 16
report = {'passed':True, 'raw_files':inventory['raw_files'], 'raw_bytes':inventory['raw_bytes'],
          'inventory_files_verified':len(inventory['files']), 'frozen_splits_verified':len(domains),
          'python':sys.version, 'numpy':np.__version__, 'domains':domains}
(ROOT / 'deployment/data_verification.json').write_text(json.dumps(report,indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k != 'domains'}))
