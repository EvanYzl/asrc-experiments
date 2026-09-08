"""Freeze one dataset using reused seed-17 validation evidence and locked predecessors."""
import argparse
import datetime
import json
from pathlib import Path
from frozen_data import ROOT,atomic_json,sha256
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args()
base=ROOT/'reproduction/sota';scope=json.loads((base/'TASK_SCOPE.json').read_text());order=scope['order']
assert scope['phase']=='single_seed_sequential' and a.dataset in order
locks=[]
for ds in order[:order.index(a.dataset)]:
    path=base/'single_seed'/ds/'LOCKED_RESULT.json';assert path.exists(),f'Prior dataset not locked: {ds}'
    lock=json.loads(path.read_text());assert lock['can_still_reach_overall_threshold']
    locks.append({'dataset':ds,'lock_sha256':sha256(path)})
folder=base/'single_seed'/a.dataset;folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'freeze.json').exists(),'Dataset already frozen; do not select again after test'
sharing='independent' if a.dataset=='depkg' else 'shared'
run=base/'pilot02'/f'{a.dataset}_{sharing}_r256_n3p01_s17'
result=json.loads((run/'result.json').read_text());config=json.loads((run/'config.json').read_text())
assert result['status']=='completed' and result['seed']==17 and result['test_access'] is False
assert sha256(run/'best.pt')==result['checkpoint_sha256']
for name,digest in config['source_hashes'].items():assert sha256(base/name)==digest
references=json.loads((base/'SINGLE_SEED_REFERENCES.json').read_text())
freeze={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':a.dataset,'seed':17,
        'method':'ASRC (Alignment-selective Reciprocal ComplEx)','algorithm_definition':'Reciprocal ComplEx, full-softmax training, N3 regularization, optional deterministic alignment entity sharing selected only by val_select; no text/external data',
        'recipe':config['recipe'],'checkpoint':(run/'best.pt').relative_to(ROOT).as_posix(),'checkpoint_sha256':result['checkpoint_sha256'],
        'best_epoch':result['best_epoch'],'validation':result['validation'],'validation_result_sha256':sha256(run/'result.json'),
        'selection_basis':'Reused complete seed17 pilot; choose maximum val_select macro MRR; earlier epoch on exact tie. Other seeds not used for selection or this phase result.',
        'source_hashes':{f'reproduction/sota/{name}':sha256(base/name) for name in ['train_complex.py','frozen_data.py','evaluate_single.py']},
        'common_sha256':sha256(ROOT/'reproduction/strict_baselines/common.py'),
        'reference_registry_sha256':sha256(base/'SINGLE_SEED_REFERENCES.json'),'references':references['datasets'][a.dataset],
        'predecessor_locks':locks,'protocol':'kbs-baselines-v1-20260905','test_policy':'One test; no retraining/reselection after test; lock before transfer.'}
atomic_json(folder/'freeze.json',freeze)
scope['current_dataset']=a.dataset;scope['locked_datasets']=[x['dataset'] for x in locks];atomic_json(base/'TASK_SCOPE.json',scope)
job={'id':f'single_{a.dataset}_s17','purpose':f'Single-seed once-only acceptance of {a.dataset}; freeze and lock before transfer',
     'changes':'Reuse already completed seed17 pilot; no training, new seeds or baselines','seed':17,'output':f'reproduction/sota/single_seed/{a.dataset}/evaluation',
     'min_free_mib':10500,'command':['/root/zhishitupui/.envs/kgc/bin/python','reproduction/sota/evaluate_single.py','--dataset',a.dataset],
     'next':'Lock test result, check remaining overall threshold feasibility, then migrate to the next dataset without changing the tested dataset.'}
atomic_json(folder/'queue/manifest.json',{'jobs':[job]})
with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## Freeze single seed '+a.dataset+' — '+freeze['timestamp']+'\nPurpose: validate this dataset once after sufficient validation evidence.\n')
    f.write('Changes/config/seed: seed17, existing pilot checkpoint at epoch '+str(result['best_epoch'])+', sharing='+sharing+'; other hyperparameters unchanged; see freeze.json.\n')
    f.write('Validation evidence: '+json.dumps(result['validation']['macro'])+'; not compared to baseline test to claim achievement.\n')
    f.write('Command/PID/GPU: queue/manifest.json has one evaluation-only command; scheduler records launch. Outputs: '+str(folder.relative_to(ROOT))+'/evaluation.\n')
    f.write('Conclusion: validation-selected checkpoint and comparison references frozen; predecessor tests locked. Next: one test, lock, then migrate only if threshold remains feasible.\n')
print(json.dumps({'dataset':a.dataset,'seed':17,'checkpoint_frozen':True,'prior_locked':[x['dataset'] for x in locks]}))
