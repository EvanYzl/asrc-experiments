"""Server-only deployment and recipe checks for the 18 fixed baseline repeats."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / 'reproduction/sota/graph_three_seed'
RUNTIME = PHASE / 'runtime'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def log(message):
    with (ROOT / 'SOTA_LOG.md').open('a', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write('\n' + message + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def normalized(args):
    return {k: v for k, v in args.items() if k not in ('seed', 'data_path', 'resume_checkpoint', 'v', 'device')}


def main():
    assert str(ROOT) == '/root/zhishitupui'
    assert not (PHASE / 'state.json').exists(), 'Inspect the already-registered scheduler before any restart'
    phase_plan = read(PHASE / 'PLAN.json')
    spec = read(PHASE / 'manifest.json')
    freeze = read(PHASE / 'SOURCE_INPUT_FREEZE.json')
    deploy = read(PHASE / 'DEPLOYMENT.json')
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log(f'## BEFORE graph-repeat preflight — {stamp}\n'
        'Purpose: user-requested LSMGA/DMKGC/IMKGC seed29/43 completion on three core datasets, 18 jobs. '
        'Local IMKGC seed17 DBP-5L/E-PKG remains assigned to the local machine and is not duplicated.\n'
        'Changes/config: graph_three_seed/PLAN.json, SOURCE_INPUT_FREEZE.json and manifest.json; identical seed17 training and evaluation recipe. '
        'Accepted path-key portability plus scheduler GPU visibility only; isolated runtime preserves all historical source files.\n'
        f'Command: `{sys.executable} reproduction/sota/preflight_graph_repeats.py`; PID {os.getpid()}; GPU: configuration checks only, no training.\n'
        'Output: reproduction/sota/graph_three_seed/PREFLIGHT.json. Next: verify inputs, graph cache and all nine parsed recipes before launching 18 fixed jobs.')
    package = ROOT / deploy['package']
    assert sha(package) == deploy['sha256']
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package, 'r:gz') as tar:
        for member in tar.getmembers():
            target = (RUNTIME / member.name).resolve()
            assert target.is_relative_to(RUNTIME.resolve()) and member.isfile()
        tar.extractall(RUNTIME)
    for name, digest in freeze['inputs'].items():
        assert sha(ROOT / name) == digest, ('input mismatch', name)
    for name, record in freeze['sources'].items():
        assert sha(RUNTIME / name) == record['server_sha256'], ('source mismatch', name)
    for name, digest in freeze['frozen_graph_cache'].items():
        assert sha(RUNTIME / name) == digest, ('cache mismatch', name)
    for link, dest in ((RUNTIME / 'data', ROOT / 'data'),
                       (RUNTIME / 'reproduction/strict_baselines/data_manifests', ROOT / 'reproduction/strict_baselines/data_manifests'),
                       (RUNTIME / 'reproduction/runs/strict_baselines_20260905/code_objects', PHASE / 'code_objects')):
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            assert link.resolve() == dest.resolve()
        else:
            assert not link.exists()
            link.symlink_to(dest, target_is_directory=True)
    recipes = []
    for method in phase_plan['methods']:
        for ds in phase_plan['datasets']:
            jid = f'{method.lower()}_{ds}_s29'
            job = next(j for j in spec['jobs'] if j['id'] == jid)
            out = PHASE / 'configuration_checks' / jid
            command = list(job['command'])
            command[command.index('--output') + 1] = str(out)
            command.append('--config-only')
            out.mkdir(parents=True, exist_ok=True)
            if (out / 'stdout.log').exists():
                saved = PHASE / 'configuration_check_history' / f'{jid}_{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")}'
                saved.mkdir(parents=True)
                for name in ('stdout.log', 'config.json', 'data_adaptation.json'):
                    if (out / name).exists():
                        shutil.copy2(out / name, saved / name)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
            with (out / 'stdout.log').open('w') as stream:
                check = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            assert check.returncode == 0, ('configuration check failed', jid)
            cfg = read(out / 'config.json')
            assert cfg['purpose'] == 'configuration_check' and cfg['seed'] == 29
            assert not (out / 'result.json').exists() and not (out / 'best.pt').exists()
            assert cfg['input_facts'] == 'train-only'
            previous = PHASE / 'seed17_reference' / f'{method.lower()}_{ds}_s17/config.json'
            if previous.exists():
                reference = read(previous)
                assert normalized(cfg['parsed_arguments_before_author_main']) == normalized(reference['parsed_arguments_before_author_main']), jid
                assert cfg['rounds'] == reference['rounds'] and cfg['selection'] == reference['selection']
                origin = 'existing local formal seed17 configuration; exact semantic match'
            else:
                assert (method, ds) == ('IMKGC', 'depkg')
                reference = read(PHASE / 'seed17_reference/imkgc_depkg_smoke/config.json')
                expected = normalized(reference['parsed_arguments_before_author_main'])
                expected.update(round=50, MAX_SAM=10**10, epoch_each=2)
                assert normalized(cfg['parsed_arguments_before_author_main']) == expected
                origin = 'existing local accepted smoke recipe, restored fixed formal budget; compare local formal config when it starts'
            for name, digest in cfg['source_hashes'].items():
                name = name.replace('\\', '/')
                assert digest == freeze['sources'][name]['server_sha256']
            recipes.append({'method': method, 'dataset': ds, 'rounds': cfg['rounds'],
                            'arguments': cfg['parsed_arguments_before_author_main'],
                            'config_sha256': sha(out / 'config.json'), 'reference': origin})
            print(json.dumps({'recipe_passed': jid, 'rounds': cfg['rounds']}), flush=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader'], text=True)
    assert not apps.strip(), 'Unexpected existing server GPU workers; do not overlap'
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.free', '--format=csv,noheader,nounits'], text=True)
    assert len(gpu.strip().splitlines()) == 6
    for line in gpu.strip().splitlines():
        assert int(line.rsplit(',', 1)[1]) >= 9000
    receipt = {'status': 'passed', 'finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'input_files_verified': len(freeze['inputs']), 'graph_cache_files_verified': len(freeze['frozen_graph_cache']),
               'source_files_verified': len(freeze['sources']), 'recipes': recipes, 'gpu_inventory': gpu,
               'training_performed': False, 'new_jobs_authorized': len(spec['jobs']),
               'manifest_sha256': sha(PHASE / 'manifest.json'), 'source_input_freeze_sha256': sha(PHASE / 'SOURCE_INPUT_FREEZE.json')}
    write(PHASE / 'PREFLIGHT.json', receipt)
    scope = read(ROOT / 'reproduction/sota/TASK_SCOPE.json')
    scope.update(status='ready_to_launch', preflight='reproduction/sota/graph_three_seed/PREFLIGHT.json')
    write(ROOT / 'reproduction/sota/TASK_SCOPE.json', scope)
    log(f'## AFTER graph-repeat preflight — {receipt["finished_at"]}\n'
        f'Purpose/config/seed: as BEFORE; PID {os.getpid()}; GPU: no training. '
        f'All {len(freeze["inputs"])} input files, {len(freeze["frozen_graph_cache"])} cache files, {len(freeze["sources"])} sources and nine parsed recipes passed.\n'
        'Result: reproduction/sota/graph_three_seed/PREFLIGHT.json. Conclusion: seed17 recipe parity verified; six GPUs available. '
        'Next: launch the registered 18 runs via run_sota_queue.py; keep local seed17 queue untouched.')
    print(json.dumps({'preflight': 'passed', 'jobs': len(spec['jobs'])}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        log(f'## AFTER graph-repeat preflight failed — {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n'
            f'PID {os.getpid()}; no training started; error: {error!r}. '
            'Preserve configuration logs. Next: fix only the failed deployment prerequisite, record the revision and rerun preflight.')
        raise
