"""Record the pre-training Windows graph-manifest portability fix."""
from pathlib import Path
import datetime
import json
import shutil
import tarfile
from prepare_graph_repeats import ROOT, PHASE, sha, write

stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
archive = PHASE / 'preflight_revisions' / '01_manifest_keys'
assert not (archive / 'REVISION.json').exists()
archive.mkdir(parents=True, exist_ok=True)
runtime = PHASE / 'runtime'
target = runtime / 'reproduction/strict_baselines/run_graph_baseline.py'
for src in (target, PHASE / 'SOURCE_INPUT_FREEZE.json', PHASE / 'DEPLOYMENT.json', ROOT / 'deployment/graph_three_seed/runtime_cache.tar.gz'):
    assert not (archive / src.name).exists()
    shutil.copy2(src, archive / src.name)
content = (ROOT / 'reproduction/strict_baselines/run_graph_baseline.py').read_text(encoding='utf-8')
old = "    signature={'input_facts':'train-only','k':10,'num_hops':2,'dataset':dataset,"
assert content.count(old) == 1
new = "    cache_signature = dest/'strict_graph_manifest.json'\n    cache_keys = {}\n    if cache_signature.exists():\n        cache_keys = {k.replace('\\\\', '/'): k for k in json.loads(cache_signature.read_text())['source_files']}\n" + old
content = content.replace(old, new)
content = content.replace("str(p.relative_to(source)):sha256(p)", "cache_keys.get(p.relative_to(source).as_posix(), p.relative_to(source).as_posix()):sha256(p)")
target.write_text(content, encoding='utf-8', newline='\n')
shutil.copy2(target, PHASE / 'code_objects' / (sha(target) + '.py'))
freeze = json.loads((PHASE / 'SOURCE_INPUT_FREEZE.json').read_text(encoding='utf-8'))
freeze['sources']['reproduction/strict_baselines/run_graph_baseline.py']['server_sha256'] = sha(target)
freeze['adaptations'].append({'path': 'reproduction/strict_baselines/run_graph_baseline.py',
    'reason': 'Preserve original Windows frozen graph-manifest keys on Linux; file hashes, graph bytes, training and evaluation unchanged',
    'recorded_at': stamp, 'preflight_failure': 'Linux kg/path key versus frozen Windows kg\\path key; no training started'})
write(PHASE / 'SOURCE_INPUT_FREEZE.json', freeze)
package = ROOT / 'deployment/graph_three_seed/runtime_cache.tar.gz'
with tarfile.open(package, 'w:gz', compresslevel=3) as tar:
    for path in sorted(runtime.rglob('*')):
        if path.is_file():
            tar.add(path, arcname=path.relative_to(runtime).as_posix())
    for name in freeze['frozen_graph_cache']:
        tar.add(ROOT / name, arcname=name)
write(PHASE / 'DEPLOYMENT.json', {'package': package.relative_to(ROOT).as_posix(), 'sha256': sha(package), 'bytes': package.stat().st_size})
write(archive / 'REVISION.json', {'recorded_at': stamp, 'purpose': 'Fix configuration preflight portability only',
    'changes': freeze['adaptations'][-1], 'training_started': False, 'next': 'Repeat all nine configuration checks'})
print(json.dumps({'revision': str(archive), 'source_sha256': sha(target), 'package_sha256': sha(package)}))
