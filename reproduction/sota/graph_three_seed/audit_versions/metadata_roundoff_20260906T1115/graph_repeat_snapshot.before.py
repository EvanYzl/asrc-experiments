"""Incremental SSH/SFTP artifact bundle; credentials are never handled here."""
from pathlib import Path
import argparse
import datetime
import hashlib
import io
import json
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / 'reproduction/sota/graph_three_seed'
DEPLOY = ROOT / 'deployment/graph_three_seed'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def pack():
    assert str(ROOT) == '/root/zhishitupui'
    acknowledged = read(PHASE / 'RETURNED_JOBS.json')['jobs'] if (PHASE / 'RETURNED_JOBS.json').exists() else []
    state = read(PHASE / 'state.json')
    new_jobs = [k for k, v in state['jobs'].items() if v['status'] == 'completed' and k not in acknowledged]
    paths = [ROOT / 'SOTA_LOG.md', ROOT / 'reproduction/sota/TASK_SCOPE.json']
    paths += sorted(PHASE.glob('*.json')) + sorted(PHASE.glob('*.md'))
    paths += sorted((PHASE / 'configuration_checks').glob('*/config.json'))
    paths += sorted((PHASE / 'configuration_checks').glob('*/data_adaptation.json'))
    paths += sorted((PHASE / 'code_objects').glob('*.py'))
    paths += sorted((PHASE / 'raw_acceptance').glob('*.json'))
    for jid, s in state['jobs'].items():
        out = ROOT / s['output']
        if jid in new_jobs:
            paths += [p for p in out.rglob('*') if p.is_file()]
        elif s['status'] != 'completed':
            paths += [out / name for name in ('config.json', 'learning_curve.jsonl', 'stdout.log', 'data_adaptation.json') if (out / name).exists()]
    paths += list((ROOT / 'reproduction/sota').glob('*graph_repeat*.py'))
    files = []
    DEPLOY.mkdir(parents=True, exist_ok=True)
    temp = DEPLOY / 'latest.tar.gz.tmp'
    with tarfile.open(temp, 'w:gz', compresslevel=3) as tar:
        for path in sorted(set(paths)):
            if not path.is_file():
                continue
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mtime = int(path.stat().st_mtime)
            tar.addfile(info, io.BytesIO(data))
            files.append({'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    temp.replace(DEPLOY / 'latest.tar.gz')
    receipt = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'new_completed_jobs': new_jobs, 'previously_returned_jobs': acknowledged,
               'files': files, 'snapshot_semantics': 'Exact captured bytes; live logs may continue to grow after capture',
               'package_sha256': hashlib.sha256((DEPLOY / 'latest.tar.gz').read_bytes()).hexdigest()}
    write(DEPLOY / 'latest_receipt.json', receipt)
    print(json.dumps({'files': len(files), 'new_completed_jobs': new_jobs,
                      'package_bytes': (DEPLOY / 'latest.tar.gz').stat().st_size}))


def apply():
    assert str(ROOT) != '/root/zhishitupui', 'Apply only in the receiving local workspace'
    receipt = read(DEPLOY / 'latest_receipt.json')
    package = DEPLOY / 'latest.tar.gz'
    assert hashlib.sha256(package.read_bytes()).hexdigest() == receipt['package_sha256']
    expected = {f['path']: f for f in receipt['files']}
    with tarfile.open(package, 'r:gz') as tar:
        members = tar.getmembers()
        assert {m.name for m in members} == set(expected)
        for member in members:
            target = ROOT / member.name
            assert target.resolve().is_relative_to(ROOT.resolve()) and member.isfile()
            assert member.name in ('SOTA_LOG.md', 'reproduction/sota/TASK_SCOPE.json') or member.name.startswith('reproduction/sota/graph_three_seed/') or (member.name.startswith('reproduction/sota/') and member.name.endswith('.py'))
            data = tar.extractfile(member).read()
            entry = expected[member.name]
            assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + '.transfer.tmp')
            temp.write_bytes(data)
            temp.replace(target)
    for name, item in expected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == item['sha256']
    acknowledged = sorted(set(receipt['previously_returned_jobs'] + receipt['new_completed_jobs']))
    report = {'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'snapshot_time': receipt['created_at'], 'verified_files': len(expected), 'jobs': acknowledged,
              'package_sha256': receipt['package_sha256']}
    write(PHASE / 'LOCAL_SYNC_VERIFICATION.json', report)
    write(PHASE / 'RETURNED_JOBS.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['pack', 'apply'])
    args = parser.parse_args()
    pack() if args.mode == 'pack' else apply()
