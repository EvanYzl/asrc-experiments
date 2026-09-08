"""Build the bounded validation-only smoke and training queues from PLAN.json."""
import json
from frozen_data import ROOT,atomic_json,sha256
base=ROOT/'reproduction/sota';phase=base/'paper_support';plan=json.loads((phase/'PLAN.json').read_text())
smoke=[];full=[]
for ident,e in plan['models'].items():
    if e['trainer']!='support':continue
    cmd=['/root/zhishitupui/.envs/kgc/bin/python','reproduction/sota/train_support.py','--id',ident]
    job={'id':ident,'purpose':'Registered ASRC component/perturbation experiment for supporting tables','changes':e['recipe'],
         'seed':e['seed'],'command':cmd,'output':e['training_path'],'min_free_mib':9500,
         'next':'Freeze val_select-best checkpoint; retain all outcomes; do not use test for selection.'}
    full.append(job)
    if e['seed']==17:smoke.append(dict(job,id='smoke_'+ident,purpose='One-epoch validation-only implementation check; excluded from paper results',
        command=cmd+['--smoke'],output='reproduction/sota/paper_support/smoke/'+ident,next='Check finite objectives, validation and derived-input audits before full queue.'))
assert len(full)==24 and len(smoke)==8
atomic_json(phase/'smoke_queue/manifest.json',{'jobs':smoke});atomic_json(phase/'wave2/manifest.json',{'jobs':full})
atomic_json(phase/'TRAINING_SOURCE_FREEZE.json',{'plan_sha256':sha256(phase/'PLAN.json'),'source_hashes':{
    p.relative_to(ROOT).as_posix():sha256(p) for p in [base/'train_complex.py',base/'frozen_data.py',base/'train_support.py',base/'support_data.py']}})
print(json.dumps({'smoke':8,'full_training':24}))
