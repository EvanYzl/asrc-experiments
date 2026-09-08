"""Launch only the audited fixed repeat manifest; reject duplicate schedulers."""
import datetime
import os
import subprocess
import sys
import psutil
from preflight_graph_repeats import ROOT, PHASE, read, write, sha, log

assert str(ROOT) == '/root/zhishitupui'
preflight = read(PHASE / 'PREFLIGHT.json')
assert preflight['status'] == 'passed' and preflight['new_jobs_authorized'] == 18
assert sha(PHASE / 'manifest.json') == preflight['manifest_sha256']
assert sha(PHASE / 'SOURCE_INPUT_FREEZE.json') == preflight['source_input_freeze_sha256']
assert not (PHASE / 'state.json').exists() and not (PHASE / 'LAUNCH.json').exists()
for proc in psutil.process_iter(['pid', 'cmdline']):
    args = proc.info['cmdline'] or []
    assert not any(a.endswith('/run_sota_queue.py') or a == 'reproduction/sota/run_sota_queue.py' for a in args), proc.info
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
assert not apps.strip()
command = [str(ROOT / '.envs/kgc/bin/python'), str(ROOT / 'reproduction/sota/run_sota_queue.py'),
           '--manifest', str(PHASE / 'manifest.json')]
with (PHASE / 'scheduler.stdout.log').open('a') as stream:
    worker = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                              stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {'launched_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'scheduler_pid': worker.pid, 'command': command, 'jobs': 18,
           'manifest_sha256': sha(PHASE / 'manifest.json'),
           'queue_source_sha256': sha(ROOT / 'reproduction/sota/run_sota_queue.py'),
           'worker_allocation': 'six independent GPU lanes; jobs recorded individually in SOTA_LOG.md and state.json'}
write(PHASE / 'LAUNCH.json', receipt)
scope = read(ROOT / 'reproduction/sota/TASK_SCOPE.json')
scope.update(status='running', scheduler_pid=worker.pid, manifest='reproduction/sota/graph_three_seed/manifest.json')
write(ROOT / 'reproduction/sota/TASK_SCOPE.json', scope)
log(f'## Graph-repeat scheduler launched — {receipt["launched_at"]}\n'
    f'Purpose: complete only the authorized 18 seed29/43 runs. Command: {command}; scheduler PID {worker.pid}.\n'
    'Configuration/input freeze: graph_three_seed/SOURCE_INPUT_FREEZE.json; per-job commands, seeds, process and GPU receipts in state.json and individual BEFORE/AFTER entries. '
    'Local seed17 queue continues unchanged. Next: monitor validation progress and completion; audit all raw results before the three-seed Table2 update.')
print(__import__('json').dumps(receipt))
