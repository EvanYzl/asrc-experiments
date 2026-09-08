"""One isolated main-table graph repeat, with durable state for later queue adoption."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent;ROOT=Path('/root/zhishitupui')
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def log(d):
    for p in [BASE/'TRAINING_LOG.md',ROOT/'SOTA_LOG.md']:
        with p.open('a') as f:f.write('\n\n```json\n'+json.dumps({'time':now(),**d},indent=2)+'\n```\n')
def main():
    name=sys.argv[1];job=next(j for j in json.loads((BASE/'EARLY_PLAN.json').read_text())['jobs'] if j['id']==name)
    statepath=BASE/'states'/f'{name}.json';assert not statepath.exists()
    hashes=json.loads((BASE/'EARLY_HASHES.json').read_text())
    for rel,digest in hashes.items():assert hashlib.sha256((BASE/rel).read_bytes()).hexdigest()==digest,rel
    info=subprocess.check_output(['nvidia-smi','-i',str(job['gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    assert int(info.strip())>7000
    runtime=BASE/'contexts'/name/'runtime';shutil.copytree(BASE/'runtime_graph',runtime)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(job['gpu']),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LANG_TABLE_GPU_FRACTION='0.85')
    command=[sys.executable,str(runtime/'run_graph_sota.py'),'--method',job['method'],'--dataset','wk3l','--seed',str(job['seed']),'--output',str(BASE/job['output'])]
    state={**job,'status':'starting','controller_pid':os.getpid(),'started_at':now(),'command':command};save(statepath,state)
    log({'event':'before_main_table_repeat',**state,'purpose':'Fill remaining WK3l Table2 means/SD while using all available GPUs','change':'Host and paths only; seed17 frozen recipe reused','next':'Complete fixed budget and preserve final raw results'})
    with (BASE/'logs'/f'{name}.log').open('wb') as f:
        p=subprocess.Popen(command,cwd=BASE,env=env,stdout=f,stderr=subprocess.STDOUT)
        state.update(status='running',pid=p.pid);save(statepath,state)
        while p.poll() is None:state['updated_at']=now();save(statepath,state);time.sleep(15)
    result=BASE/job['output']/'result.json';okay=p.returncode==0 and result.exists()
    if okay:
        d=json.loads(result.read_text());okay=d['status']=='completed' and d['full_data'] is True and d['seed']==job['seed']
    state.update(status='completed' if okay else 'failed',exit_code=p.returncode,finished_at=now());save(statepath,state)
    log({'event':'after_main_table_repeat',**state,'result':str(result),'conclusion':state['status'],'next':'Aggregate seeds17/29/43 and return raw artifacts; do not tune on test'})
    return 0 if okay else 1

if __name__=='__main__':raise SystemExit(main())
