"""Single-GPU Windows scheduler. Resume at durable checkpoints; validate completion."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);args=p.parse_args()
    path=Path(args.manifest);root=path.parent;state_path=root/'queue_state.json'
    lock=root/'queue.lock.json'
    if lock.exists():
        old=json.loads(lock.read_text(encoding='utf-8'))
        if psutil.pid_exists(old['pid']):raise RuntimeError('Queue already active')
    save(lock,{'pid':os.getpid(),'started':time.time()})
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'jobs':{}}
    state.update(pid=os.getpid(),status='running')
    try:
        while True:
            manifest=json.loads(path.read_text(encoding='utf-8'))
            for j in manifest['jobs']:state['jobs'].setdefault(j['id'],{'status':'pending'})
            ready=[]
            for j in manifest['jobs']:
                s=state['jobs'][j['id']]
                if s['status'] not in ['pending','running','retry']:continue
                if all(state['jobs'][d]['status']=='completed' for d in j.get('depends_on',[])):ready.append(j)
            if not ready:
                state['status']='complete' if all(s['status']=='completed' for s in state['jobs'].values()) else 'needs_attention'
                state['current_job']=None;save(state_path,state);break
            j=ready[0];s=state['jobs'][j['id']]
            if s.get('pid') and psutil.pid_exists(s['pid']):raise RuntimeError(f'Existing job is still running: {s["pid"]}')
            result_path=Path(j['result'])
            if result_path.exists():
                result=json.loads(result_path.read_text(encoding='utf-8'))
                if result.get('status')=='completed':
                    s.update(status='completed',recovered=True);save(state_path,state);continue
            env=os.environ.copy();env.update(manifest.get('env',{}));env.update(j.get('env',{}))
            Path(j['cwd']).mkdir(parents=True,exist_ok=True)
            log=root/'logs'/f'{j["id"]}.log';log.parent.mkdir(exist_ok=True)
            attempt=s.get('attempts',0)+1;start=time.time()
            print(f'START {j["id"]}',flush=True)
            with log.open('a',encoding='utf-8') as f:
                f.write('\n'+json.dumps({'attempt':attempt,'start':start,'command':j['command']})+'\n');f.flush()
                child=subprocess.Popen(j['command'],cwd=j['cwd'],env=env,stdout=f,stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                s.update(status='running',pid=child.pid,start=start,attempts=attempt,log=str(log));state['current_job']=j['id'];save(state_path,state)
                peak_rss=0
                while child.poll() is None:
                    try:peak_rss=max(peak_rss,psutil.Process(child.pid).memory_info().rss)
                    except psutil.Error:pass
                    state['updated_at']=time.time();s['elapsed_s']=time.time()-start;s['rss_peak_bytes']=peak_rss;save(state_path,state)
                    time.sleep(10)
                code=child.returncode
            success=False
            if code==0 and result_path.exists():
                result=json.loads(result_path.read_text(encoding='utf-8'));success=result.get('status')=='completed'
            s.update(status='completed' if success else 'failed',exit_code=code,elapsed_s=time.time()-start,finished=time.time())
            save(state_path,state);print(f'END {j["id"]}: {s["status"]}',flush=True)
            # An OOM retry uses the same recipe; numerical/training changes require an explicit new config.
            if not success and attempt<3 and 'out of memory' in log.read_text(encoding='utf-8',errors='replace').lower()[-20000:]:
                s['status']='retry';save(state_path,state);time.sleep(30)
    finally:
        if lock.exists():lock.unlink()


if __name__=='__main__':main()
