"""Pause the single remaining original job and preserve its complete resumable state."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shutil
import tarfile
import psutil
import torch

ROOT=Path('G:/zhishitupui')
HERE=Path(__file__).resolve().parent
QUEUE=ROOT/'reproduction/runs/strict_baselines_20260905'
JOB=QUEUE/'jobs/imkgc_depkg_s17'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)

def main():
    assert not (HERE/'checkpoint_bundle.READY.json').exists()
    state=read(QUEUE/'queue_state.json');manifest=read(QUEUE/'manifest.json')
    assert [j['id'] for j in manifest['jobs'] if state['jobs'][j['id']]['status']!='completed']==['imkgc_depkg_s17']
    scheduler=psutil.Process(75848);training=psutil.Process(106428)
    assert any(x.replace('\\','/').endswith('/strict_baselines/run_queue_lanes.py') for x in scheduler.cmdline())
    assert '--dataset' in training.cmdline() and training.cmdline()[training.cmdline().index('--dataset')+1]=='depkg'
    assert str(JOB).replace('\\','/').lower() in [x.replace('\\','/').lower() for x in training.cmdline()]
    scheduler.suspend();training.suspend()
    record={'timestamp':dt.datetime.now(dt.timezone.utc).isoformat(),'local_scheduler_pid':scheduler.pid,'local_training_pid':training.pid,
            'scheduler_created_at':scheduler.create_time(),'training_created_at':training.create_time(),'status':'suspended_for_checkpoint_transfer'}
    save(HERE/'LOCAL_HANDOFF.json',record)
    snapshot=HERE/'local_epkg_checkpoint'
    snapshot.mkdir(exist_ok=False)
    for p in JOB.rglob('*'):
        if p.is_file() and p.suffix!='.tmp':
            dest=snapshot/p.relative_to(JOB);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name in ['queue_state.json','queue.lock.json','manifest.json']:
        if (QUEUE/name).exists():shutil.copy2(QUEUE/name,HERE/('parent_before_'+name))
    last=torch.load(snapshot/'last.pt',map_location='cpu',weights_only=False)
    best=torch.load(snapshot/'best.pt',map_location='cpu',weights_only=False)
    assert last['format']=='multidomain_full_state_v1'
    assert 0<last['completed_round']+1<50 and best['evaluation_index']<=last['completed_round']+2
    required=['model_state_dict','optimizer_state_dict','scheduler_state_dict','python_rng_state','numpy_rng_state','torch_rng_state','cuda_rng_state_all']
    assert all(k in last and last[k] is not None for k in required)
    assert len(last['cuda_rng_state_all'])==1
    files={p.relative_to(snapshot).as_posix():sha(p) for p in snapshot.rglob('*') if p.is_file()}
    provenance={**record,'completed_rounds':last['completed_round']+1,'selected_evaluation':best['evaluation_index'],'sha256':files,
                'training_budget':50,'seed':17,'resume_fields_verified':required,'local_device':'RTX 5080','remote_device':'RTX 2080 Ti',
                'selection':'Fixed validation-only selection continues; no final test has been run or used to choose the host'}
    save(snapshot/'MIGRATION_CHECKPOINT.json',provenance)
    archive=HERE/'checkpoint_bundle.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in sorted(snapshot.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(snapshot).as_posix(),recursive=False)
    ready={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'completed_rounds':provenance['completed_rounds']}
    save(HERE/'checkpoint_bundle.READY.json',ready)
    state=read(QUEUE/'queue_state.json')
    state['jobs']['imkgc_depkg_s17'].update(status='migrating',migration=str(HERE/'LOCAL_HANDOFF.json'),completed_rounds_at_migration=provenance['completed_rounds'])
    state.update(status='paused_for_migration',migration_owner=str(HERE/'TRANSFER_STATE.json'))
    save(QUEUE/'queue_state.json',state)
    record.update(completed_rounds=provenance['completed_rounds'],checkpoint_sha256=ready['sha256']);save(HERE/'LOCAL_HANDOFF.json',record)
    log={'time':record['timestamp'],'event':'before_server_continuation','job':'imkgc_depkg_s17','seed':17,'purpose':'User-requested migration to free server GPU',
         'local_processes_suspended':[scheduler.pid,training.pid],'completed_rounds':provenance['completed_rounds'],'checkpoint_archive':str(archive),
         'changes':'Host/path only; optimizer/scheduler/RNG and model state retained','next':'Verify server checkpoint, start continuation, then retire these suspended local processes'}
    for p in [ROOT/'SOTA_LOG.md',ROOT/'reproduction/language_table_seed17/TRAINING_LOG.md']:
        with p.open('a',encoding='utf-8') as f:f.write('\n'+json.dumps(log,ensure_ascii=False)+'\n')
    print(json.dumps(ready))

if __name__=='__main__':main()
