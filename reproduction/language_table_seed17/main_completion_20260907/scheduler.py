"""Keep six independent GPU slots occupied by the frozen, finite main-table plan."""
from pathlib import Path
import collections
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

import psutil
import numpy as np

BASE=Path(__file__).resolve().parent;ROOT=Path('/root/zhishitupui')
CHILDREN={}
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def read(path):return json.loads(path.read_text())
def log(data):
    item={'time':now(),**data}
    for path in [BASE/'TRAINING_LOG.md',ROOT/'SOTA_LOG.md']:
        with path.open('a') as f:f.write('\n```json\n'+json.dumps(item,indent=2)+'\n```\n')
def alive(pid):
    if not pid:return False
    try:return psutil.Process(pid).is_running() and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
    except psutil.Error:return False
def gpu_status():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in raw.strip().splitlines():
        a=[x.strip() for x in line.split(',')]
        rows.append(dict(index=int(a[0]),uuid=a[1],free_mib=int(a[2]),used_mib=int(a[3]),utilization=int(a[4])))
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    occupied=set()
    for line in procs.splitlines():
        if ',' in line:occupied.add(line.split(',')[0].strip())
    for row in rows:row['has_compute_process']=row['uuid'] in occupied
    return rows
def check_result(job):
    out=BASE/job['output'];result=read(out/'result.json');config=read(out/'config.json')
    assert result['status']=='completed' and result['full_data'] is True and result['seed']==job['seed']
    assert config['seed']==job['seed'] and config['method']==job['method']
    freeze=read(out/'FINAL_EVALUATION_FREEZE.json')
    assert freeze['seed']==job['seed'] and freeze['checkpoint_sha256']==sha(out/'best.pt')
    if job['kind']=='ssaga':
        assert freeze['completed_rounds']==25
        groups=[(out,result,job['kg'],'train')]
    else:
        assert freeze['completed_rounds']==(50 if job['method']=='LSMGA' else 30)
        groups=[(out/kg,v,kg,'all') for kg,v in result['per_kg'].items()]
        assert set(result['per_kg'])=={'en','fr'}
    checks=[]
    for folder,group,kg,primary in groups:
        arrays=np.load(folder/'test_queries.npz',allow_pickle=False)
        ranks=arrays['rank_'+primary].astype(np.float64)
        calculated={'mrr':float((1/ranks).mean()),'h1':float((ranks<=1).mean()),'h10':float((ranks<=10).mean())}
        assert len(ranks)>64 and np.isfinite(ranks).all() and (ranks>=1).all()
        assert len(ranks)==group['metrics'][primary]['n']
        for key,value in calculated.items():assert abs(value-group['metrics'][primary][key])<1e-12,(job['id'],key)
        checks.append({'kg':kg,'filter':primary,'n':len(ranks),'metrics':calculated,'rank_sha256':sha(folder/'test_queries.npz')})
    return {'status':'verified','checked_at':now(),'checks':checks,'freeze':freeze,'config_sha256':sha(out/'config.json')}
def archive(job,state):
    out=BASE/job['output'];verification=check_result(job)
    save(out/'QUEUE_VERIFICATION.json',verification)
    files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='RETURN_HASHES.json'}
    save(out/'RETURN_HASHES.json',files)
    returns=BASE/'returns';returns.mkdir(exist_ok=True)
    final=returns/(job['id']+'.tar.gz');temporary=final.with_suffix('.partial')
    with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:
        for p in sorted(out.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(out).as_posix(),recursive=False)
    os.replace(temporary,final)
    state.update(return_archive=str(final),return_sha256=sha(final),return_bytes=final.stat().st_size,verification=verification)
    save(BASE/'states'/(job['id']+'.json'),state)
    log({'event':'after_main_table_job_verified',**job,'gpu':state.get('gpu'),'pid':state.get('pid'),'result':str(out/'result.json'),
         'conclusion':'Completed fixed budget; raw full-candidate ranks agree with result','next':'Return verified raw artifacts and aggregate frozen seeds',
         'return_archive':str(final),'return_sha256':state['return_sha256']})
def start(job,gpu):
    out=BASE/job['output'];assert not out.exists(),('Unregistered output exists',out)
    command=list(job['command']);command[0]=sys.executable
    if job['kind']=='graph':
        runtime=BASE/'contexts'/job['id']/'runtime'
        assert not runtime.exists()
        shutil.copytree(BASE/'runtime_graph',runtime)
        command[1]=str(runtime/'run_graph_sota.py')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4' if job['kind']=='ssaga' else '2',MKL_NUM_THREADS='4' if job['kind']=='ssaga' else '2',LANG_TABLE_GPU_FRACTION='0.85')
    state={**job,'status':'starting','gpu':gpu,'command':command,'started_at':now(),'controller_pid':os.getpid()}
    save(BASE/'states'/(job['id']+'.json'),state)
    log({'event':'before_main_table_job',**state,'purpose':'Fill the remaining main-PDF cells using fixed three-seed recipes',
        'change':'Host/path and independent GPU scheduling only; no model or hyperparameter changes',
        'result':str(out/'result.json'),'next':'Finish fixed budget, verify raw ranks, return and fill the original table'})
    with (BASE/'logs'/(job['id']+'.log')).open('wb') as f:
        child=subprocess.Popen(command,cwd=BASE,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    CHILDREN[job['id']]=child;state.update(status='running',pid=child.pid);save(BASE/'states'/(job['id']+'.json'),state)
def main():
    lock=BASE/'SCHEDULER_LOCK.json'
    if lock.exists():
        previous=read(lock)
        assert not alive(previous['pid']),('Scheduler already running',previous)
        shutil.copy2(lock,BASE/f"SCHEDULER_LOCK.previous_{int(time.time())}.json")
    save(lock,{'pid':os.getpid(),'started_at':now()})
    jobs=read(BASE/'PLAN.json')['jobs'];assert len(jobs)==39
    for folder in ['logs','states','returns']:(BASE/folder).mkdir(exist_ok=True)
    log({'event':'start_six_gpu_queue','pid':os.getpid(),'jobs':39,'already_running_jobs':'Adopt early states without restarting','stop':'Finite registered queue only; six independent slots'})
    while True:
        statuses={};reserved=set()
        for job in jobs:
            statefile=BASE/'states'/(job['id']+'.json')
            if not statefile.exists():statuses[job['id']]={'status':'pending'};continue
            state=read(statefile)
            child=CHILDREN.get(job['id'])
            if child is not None and child.poll() is not None:
                state.update(exit_code=child.returncode,finished_at=now(),status='completed' if child.returncode==0 else 'failed')
                save(statefile,state);CHILDREN.pop(job['id'])
            if state['status'] in ['starting','running']:
                if alive(state.get('pid')) or alive(state.get('controller_pid')):
                    reserved.add(state['gpu'])
                else:
                    state.update(status='completed' if (BASE/job['output']/'result.json').exists() else 'failed',finished_at=now(),reason='Recovered terminal process state')
                    save(statefile,state)
            if state['status']=='completed' and not state.get('return_archive'):
                try:archive(job,state)
                except Exception:
                    state.update(status='verification_failed',error=traceback.format_exc());save(statefile,state)
                    log({'event':'result_verification_failed','job':job['id'],'error':state['error'],'next':'Preserve result and inspect; do not fill unsupported values'})
            if state['status']=='failed' and not state.get('failure_recorded'):
                state['failure_recorded']=now();save(statefile,state)
                log({'event':'after_main_table_job_failed',**state,'result':str(BASE/job['output']),
                    'conclusion':'Execution failed; logs preserved','next':'Continue other registered tasks and inspect failure before retry'})
            statuses[job['id']]=state
        rows=gpu_status()
        ready=(BASE/'PREFLIGHT.json').exists()
        pending=[j for j in jobs if statuses[j['id']]['status']=='pending' and (j['kind']=='graph' or ready)]
        free=[g['index'] for g in rows if not g['has_compute_process'] and g['index'] not in reserved and g['free_mib']>7000]
        for gpu,job in zip(free,pending):
            start(job,gpu);statuses[job['id']]=read(BASE/'states'/(job['id']+'.json'))
        counts=dict(collections.Counter(s['status'] for s in statuses.values()))
        snapshot={'updated_at':now(),'pid':os.getpid(),'counts':counts,'gpu':rows,'preflight_ready':ready,'jobs':statuses}
        save(BASE/'QUEUE_STATE.json',snapshot)
        if all(s['status'] in ['completed','failed','verification_failed'] for s in statuses.values()):
            snapshot['status']='completed' if all(s['status']=='completed' for s in statuses.values()) else 'needs_attention'
            save(BASE/'QUEUE_STATE.json',snapshot);log({'event':'queue_finished','status':snapshot['status'],'counts':counts,'next':'Verify returned artifacts and compile both PDFs; no additional runs'});return
        time.sleep(15)
if __name__=='__main__':
    try:main()
    except Exception:
        save(BASE/'SCHEDULER_ERROR.json',{'time':now(),'error':traceback.format_exc()});raise
