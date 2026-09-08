"""Six-lane durable queue. Name deliberately avoids shadowing Python's queue."""
import argparse
import datetime
import fcntl
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from frozen_data import ROOT,atomic_json

def append_log(text):
    with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+text+'\n');f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args()
    manifest=Path(a.manifest).resolve();spec=json.loads(manifest.read_text());base=manifest.parent
    lock=(base/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=base/'state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else {'jobs':{},'scheduler_pid':os.getpid()}
    running={}
    for job in spec['jobs']:
        old=state['jobs'].get(job['id'],{})
        if old.get('status')=='running':
            try:os.kill(old['pid'],0);raise RuntimeError('Previous job still alive; inspect before queue restart')
            except ProcessLookupError:old['status']='interrupted';state['jobs'][job['id']]=old
    while True:
        for jid,(proc,logfile,job) in list(running.items()):
            ret=proc.poll()
            if ret is None:continue
            logfile.close();s=state['jobs'][jid];output=ROOT/job['output'];ok=ret==0 and (output/'result.json').exists()
            result=json.loads((output/'result.json').read_text()) if ok else None
            if ok:ok=result.get('status')=='completed'
            s.update(status='completed' if ok else 'failed',returncode=ret,finished=time.time(),result=result)
            append_log(f"## AFTER {jid} — {datetime.datetime.now(datetime.timezone.utc).isoformat()}\nPurpose: {job['purpose']}\nChanges/config/seed/command: see BEFORE and {job['output']}/config.json.\nPID: {proc.pid}; GPU: {s['gpu']}; return code: {ret}; result: {job['output']}/result.json.\nConclusion: {'completed; validation only, no test' if ok and result.get('test_access') is False else ('completed; purpose='+str(result.get('purpose')) if ok else 'FAILED; keep logs and artifacts')}.\nMetrics: {json.dumps(result.get('validation',result.get('macro')) if result else None)}\nNext: {job.get('next','review validation and select further budget without accessing test')}")
            del running[jid]
        atomic_json(statepath,state)
        pending=[j for j in spec['jobs'] if j['id'] not in state['jobs']]
        if not pending and not running:break
        gpurows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        busy={state['jobs'][jid]['gpu'] for jid in running}
        free=[(int(row.split(',')[0]),int(row.split(',')[1])) for row in gpurows.strip().splitlines()]
        for job in pending:
            usable=[(g,m) for g,m in free if g not in busy and m>=job.get('min_free_mib',9500)]
            if not usable:break
            gpu,mem=max(usable,key=lambda x:(x[1],-x[0]));busy.add(gpu)
            out=ROOT/job['output'];out.mkdir(parents=True,exist_ok=True)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            command=job['command'];stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
            append_log(f"## BEFORE {job['id']} — {stamp}\nPurpose: {job['purpose']}\nChanges: {job.get('changes','see frozen command and source hashes in config')}\nSeed: {job['seed']}; GPU: {gpu}, free {mem} MiB; command: `{shlex.join(command)}`\nOutput: {job['output']}; PID: pending launch.\nConclusion: preconditions met; next: run bounded job, inspect exit and actual result.")
            log=(out/'stdout.log').open('a');proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['jobs'][job['id']]={'status':'running','pid':proc.pid,'gpu':gpu,'started':time.time(),'command':command,'output':job['output']}
            running[job['id']]=(proc,log,job)
            append_log(f"Launch receipt {job['id']}: PID {proc.pid}, GPU {gpu}, seed {job['seed']}, log {job['output']}/stdout.log.")
            atomic_json(statepath,state)
        time.sleep(10)
    state['finished']=time.time();atomic_json(statepath,state)

if __name__=='__main__':main()
