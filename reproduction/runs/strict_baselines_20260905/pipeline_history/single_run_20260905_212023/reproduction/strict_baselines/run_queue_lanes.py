"""One durable queue owner, one GPU job and one CPU job; adopt live children safely."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil


def save(path,obj):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for attempt in range(40):
        try:
            os.replace(temp,path);return
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05) # Windows readers may briefly hold the destination open.


def lane(job):
    if 'benchmark' in job['id']:return 'exclusive'
    return job.get('resource','cpu' if job.get('env',{}).get('CUDA_VISIBLE_DEVICES')=='-1' else 'gpu')


def valid_result(job):
    try:
        result=json.loads(Path(job['result']).read_text(encoding='utf-8'))
        if result.get('status')!='completed':return False
        return True
    except (OSError,ValueError):return False


def process_matches(job,state,process):
    # Never adopt an unrelated process after PID reuse, even if it has the same name.
    normal=lambda value:str(value).replace('\\','/').lower()
    try:
        if not process.is_running() or process.status()==psutil.STATUS_ZOMBIE:return False
        if abs(process.create_time()-state.get('process_created_at',state['start']))>30:return False
        return list(map(normal,process.cmdline()))==list(map(normal,job['command']))
    except (psutil.Error,KeyError):return False


def has_capacity(job,live,jobs,min_memory_bytes):
    wanted=lane(job);occupied={lane(jobs[name]) for name in live}
    if wanted=='exclusive':return not live
    if 'exclusive' in occupied or wanted in occupied:return False
    return not live or psutil.virtual_memory().available>=min_memory_bytes


def run(path,poll_seconds,min_memory_bytes):
    root=path.parent;state_path=root/'queue_state.json';lock=root/'queue.lock.json'
    if lock.exists():
        old=json.loads(lock.read_text(encoding='utf-8'))
        if psutil.pid_exists(old['pid']):raise RuntimeError('Queue already active')
    save(lock,{'pid':os.getpid(),'started':time.time(),'scheduler':'cpu-gpu-lanes-v1'})
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'jobs':{}}
    state.update(pid=os.getpid(),status='running',scheduler='cpu-gpu-lanes-v1')
    live={};handles={}
    try:
        manifest=json.loads(path.read_text(encoding='utf-8'))
        jobs={job['id']:job for job in manifest['jobs']}
        for name,s in state['jobs'].items():
            if s.get('status')!='running':continue
            if name not in jobs:raise RuntimeError(f'Running job is outside active scope: {name}')
            if s.get('pid') and psutil.pid_exists(s['pid']):
                process=psutil.Process(s['pid'])
                if not process_matches(jobs[name],s,process):
                    raise RuntimeError(f'Live process identity does not match {name}: {s["pid"]}')
                live[name]=process;s['adopted_by']=os.getpid();s['process_created_at']=process.create_time()
                print(f'ADOPT {name}: {process.pid}',flush=True)
            elif valid_result(jobs[name]):s.update(status='completed',recovered=True)
            else:s.update(status='pending',interrupted=True)
        while True:
            manifest=json.loads(path.read_text(encoding='utf-8'))
            jobs={job['id']:job for job in manifest['jobs']}
            assert set(live)<=set(jobs),'Cannot defer a running job without preserving it'
            for name in jobs:state['jobs'].setdefault(name,{'status':'pending'})
            for name,process in list(live.items()):
                s=state['jobs'][name];child=handles.get(name)
                alive=child.poll() is None if child is not None else process.is_running()
                if alive:
                    rss=0
                    try:
                        for member in [process,*process.children(recursive=True)]:
                            try:rss+=member.memory_info().rss
                            except psutil.Error:pass
                    except psutil.Error:pass
                    s['rss_tree_peak_bytes']=max(s.get('rss_tree_peak_bytes',0),rss)
                    s['elapsed_s']=time.time()-s['start']
                    continue
                code=child.returncode if child is not None else None
                success=(code is None or code==0) and valid_result(jobs[name])
                s.update(status='completed' if success else 'failed',exit_code=code,
                         elapsed_s=time.time()-s['start'],finished=time.time())
                if child is None:s['completion_evidence']='result artifact after adopted process exit'
                del live[name];handles.pop(name,None)
                if not success and s.get('attempts',1)<3:
                    log=Path(s['log'])
                    # Only reuse the exact recipe on an OOM; other failures need repair.
                    with log.open('rb') as f:
                        f.seek(max(0,log.stat().st_size-20000));tail=f.read().decode('utf-8',errors='replace').lower()
                    if 'out of memory' in tail:s['status']='retry'
                print(f'END {name}: {s["status"]}',flush=True)
            ready=[]
            for job in manifest['jobs']:
                s=state['jobs'][job['id']]
                if s['status'] not in ['pending','retry']:continue
                if valid_result(job):
                    s.update(status='completed',recovered=True);continue
                if all(state['jobs'].get(dep,{}).get('status')=='completed' for dep in job.get('depends_on',[])):
                    ready.append(job)
            # Results recovered above can make a dependency ready on the next poll.
            for job in ready:
                if not has_capacity(job,live,jobs,min_memory_bytes):continue
                name=job['id'];s=state['jobs'][name]
                env=os.environ.copy();env.update(manifest.get('env',{}));env.update(job.get('env',{}))
                log=root/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
                Path(job['cwd']).mkdir(parents=True,exist_ok=True)
                start=time.time();attempt=s.get('attempts',0)+1
                with log.open('a',encoding='utf-8') as f:
                    f.write('\n'+json.dumps({'attempt':attempt,'start':start,'command':job['command'],'resource':lane(job)})+'\n');f.flush()
                    child=subprocess.Popen(job['command'],cwd=job['cwd'],env=env,stdout=f,stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                process=psutil.Process(child.pid)
                s.update(status='running',pid=child.pid,process_created_at=process.create_time(),start=start,
                         attempts=attempt,log=str(log),resource=lane(job))
                live[name]=process;handles[name]=child
                print(f'START {name}: {lane(job)}',flush=True)
                save(state_path,state)
            state['current_jobs']=sorted(live,key=lambda name:(lane(jobs[name])!='gpu',name))
            state['current_job']=next(iter(state['current_jobs']),None)
            state['updated_at']=time.time()
            if not live:
                # Recheck dependencies after every recovered result, avoiding false completion.
                runnable=any(state['jobs'][job['id']]['status'] in ['pending','retry'] and
                    all(state['jobs'].get(dep,{}).get('status')=='completed' for dep in job.get('depends_on',[])) for job in jobs.values())
                if not runnable:
                    replaced={j['supersedes'] for j in jobs.values() if j.get('supersedes')}
                    complete=all(state['jobs'][name]['status']=='completed' for name in jobs if name not in replaced)
                    state['status']='complete' if complete else 'needs_attention'
                    save(state_path,state);break
            save(state_path,state)
            time.sleep(poll_seconds)
    finally:
        if lock.exists() and json.loads(lock.read_text())['pid']==os.getpid():lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    parser.add_argument('--poll-seconds',type=float,default=10)
    parser.add_argument('--min-free-memory-gb',type=float,default=8)
    args=parser.parse_args();run(Path(args.manifest),args.poll_seconds,args.min_free_memory_gb*1024**3)
