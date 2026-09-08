"""Run one registered migrated experiment, then archive its exact output files."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time

BASE=Path(__file__).resolve().parent
ROOT=Path('/root/zhishitupui')
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def log(d):
    for p in [BASE/'MIGRATION_LOG.md',ROOT/'SOTA_LOG.md']:
        with p.open('a') as f:f.write('\n\n```json\n'+json.dumps({'time':now(),**d},indent=2)+'\n```\n')
def main():
    name=sys.argv[1];deployment=json.loads((BASE/'DEPLOYMENT.json').read_text());job=deployment['jobs'][name]
    statepath=BASE/'states'/f'{name}.json'
    assert not statepath.exists(),'This registered job already has a controller state'
    if name=='imkgc_depkg_s17':assert (BASE/job['output']/'MIGRATION_CHECKPOINT.json').exists()
    check=subprocess.run([sys.executable,str(BASE/'server_preflight.py')]+(['--checkpoint'] if name=='imkgc_depkg_s17' else []),capture_output=True,text=True)
    assert check.returncode==0,check.stdout+check.stderr
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(job['gpu']),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4' if name=='imkgc_depkg_s17' else '2',MKL_NUM_THREADS='4' if name=='imkgc_depkg_s17' else '2',LANG_TABLE_GPU_FRACTION='0.85')
    command=job['command'].copy();command[0]=sys.executable
    logfile=BASE/'logs'/f'{name}.log';logfile.parent.mkdir(exist_ok=True)
    state={'status':'starting','controller_pid':os.getpid(),'started_at':now(),**job,'command':command,'log':str(logfile),'preflight':json.loads(check.stdout)}
    save(statepath,state);log({'event':'before_migrated_training',**state,'change':'Host and filesystem relocation only; model/loss/splits/budget unchanged','next':'Complete fixed budget, evaluate the validation-selected checkpoint, return all raw artifacts'})
    with logfile.open('wb') as f:
        proc=subprocess.Popen(command,cwd=BASE,env=env,stdout=f,stderr=subprocess.STDOUT)
        state.update(status='running',pid=proc.pid);save(statepath,state)
        log({'event':'migrated_process_started','job':name,'pid':proc.pid,'gpu':job['gpu'],'seed':17})
        while proc.poll() is None:
            state['updated_at']=now();save(statepath,state);time.sleep(15)
    result=BASE/job['output']/'result.json'
    okay=proc.returncode==0 and result.exists()
    if okay:
        r=json.loads(result.read_text());okay=r.get('status')=='completed' and r.get('full_data') is True and r.get('seed')==17
    state.update(status='completed' if okay else 'failed',exit_code=proc.returncode,finished_at=now());save(statepath,state)
    log({'event':'after_migrated_training','job':name,'pid':proc.pid,'gpu':job['gpu'],'seed':17,'exit_code':proc.returncode,'result':str(result),'conclusion':state['status'],'next':'Archive exact raw results for verified return' if okay else 'Preserve failure log; inspect before retry'})
    if not okay:return 1
    out=result.parent
    hashes={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix!='.tmp'}
    save(out/'RETURN_HASHES.json',hashes)
    (BASE/'returns').mkdir(exist_ok=True)
    archive=BASE/'returns'/f'{name}.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix!='.tmp':tar.add(p,arcname=p.relative_to(out).as_posix(),recursive=False)
        tar.add(logfile,arcname='server_training.log',recursive=False)
    state.update(status='ready_to_return',archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,files=len(hashes));save(statepath,state)
    log({'event':'migrated_results_archived','job':name,'archive':str(archive),'sha256':state['archive_sha256'],'next':'Return and verify locally; no further experiment for this job'})
    return 0

if __name__=='__main__':raise SystemExit(main())
