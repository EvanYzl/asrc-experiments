"""Observe already-running candidates after the user removed all baseline jobs."""
import datetime
import json
import time
from pathlib import Path
import psutil
from frozen_data import ROOT,atomic_json,sha256
from run_sota_queue import append_log

base=ROOT/'reproduction/sota/formal_queue'
state=json.loads((base/'state.json').read_text())
manifest=json.loads((base/'manifest.json').read_text())
jobs={j['id']:j for j in manifest['jobs']}
assert all(k.startswith('asrc_') for k in jobs)
while True:
    for jid,s in state['jobs'].items():
        if s['status']!='running':continue
        assert jid in jobs
        alive=psutil.pid_exists(s['pid']) and psutil.Process(s['pid']).status()!=psutil.STATUS_ZOMBIE
        if alive:continue
        out=ROOT/jobs[jid]['output'];rp=out/'result.json';result=json.loads(rp.read_text()) if rp.exists() else None
        ok=bool(result and result['status']=='completed' and result['test_access'] is False and sha256(out/'best.pt')==result['checkpoint_sha256'])
        s.update(status='completed' if ok else 'failed',result=result,finished=time.time(),returncode=None,
                 completion_evidence='Adopted process exited; OS exit code unavailable. Result status, validation-only flag and checkpoint SHA-256 verified.' if ok else 'Adopted process exited without a valid completed result.')
        append_log(f"## AFTER {jid} — {datetime.datetime.now(datetime.timezone.utc).isoformat()}\nPurpose: frozen candidate three-seed confirmation.\nChanges/config/seed/command: {json.dumps(jobs[jid])}\nPID {s['pid']}, GPU {s['gpu']}; result {jobs[jid]['output']}/result.json.\nConclusion: {s['completion_evidence']}\nValidation: {json.dumps(result.get('validation') if result else None)}\nNext: compare frozen candidate means once all three seeds finish; reuse local baseline results, preserve unknown/provisional status.")
    atomic_json(base/'state.json',state)
    if not any(s['status']=='running' for s in state['jobs'].values()):break
    time.sleep(10)
state['finished']=time.time();state['scope']='candidate only; baseline queue cancelled by user steering'
atomic_json(base/'state.json',state)
