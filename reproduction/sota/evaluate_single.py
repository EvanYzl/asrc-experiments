"""Exactly one test for a validation-frozen seed17 candidate, then irrevocably lock."""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from frozen_data import ROOT,DOMAINS,atomic_json,sha256,load_multikg
from train_complex import Complex
sys.path.insert(0,str(ROOT/'reproduction/strict_baselines'))
from common import load_kg,evaluate,macro_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DOMAINS,required=True);a=p.parse_args()
    base=ROOT/'reproduction/sota';folder=base/'single_seed'/a.dataset;frozen_path=folder/'freeze.json'
    frozen=json.loads(frozen_path.read_text());scope=json.loads((base/'TASK_SCOPE.json').read_text())
    assert scope['phase']=='single_seed_sequential' and scope['current_dataset']==a.dataset and frozen['seed']==17
    assert sha256(base/'SINGLE_SEED_REFERENCES.json')==frozen['reference_registry_sha256']
    for name,digest in frozen['source_hashes'].items():assert sha256(ROOT/name)==digest
    assert sha256(ROOT/'reproduction/strict_baselines/common.py')==frozen['common_sha256']
    for item in frozen['predecessor_locks']:assert sha256(base/'single_seed'/item['dataset']/'LOCKED_RESULT.json')==item['lock_sha256']
    cp=ROOT/frozen['checkpoint'];assert sha256(cp)==frozen['checkpoint_sha256']
    out=folder/'evaluation';out.mkdir(parents=True,exist_ok=True);marker=out/'TEST_OPENED.json'
    assert not marker.exists(),'Already tested; no test-driven retries'
    atomic_json(out/'config.json',frozen)
    atomic_json(marker,{'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha256(frozen_path),'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(cp,map_location='cuda',weights_only=False);recipe=saved['config']['recipe']
    _,maps,roff,n,nr,audit=load_multikg(a.dataset,recipe['sharing']);maps={k:torch.as_tensor(v,device='cuda') for k,v in maps.items()}
    model=Complex(n,nr,recipe['rank']).cuda();model.load_state_dict(saved['model']);model.eval();per={};selection={};t0=time.monotonic()
    for kg in DOMAINS[a.dataset]:
        d=load_kg(a.dataset,kg);mapping=maps[kg]
        def score(batch):
            b=torch.as_tensor(batch.copy(),device='cuda')
            return model.scores(mapping[b[:,0]],b[:,1]+roff[kg],mapping)
        kgout=out/kg;kgout.mkdir(exist_ok=True)
        val=evaluate(d,score,split='val_select',batch_size=256,output=kgout/'selection_queries.npz')
        for key in ['mrr','h1','h10']:assert abs(val['metrics']['select'][key]-frozen['validation']['per_kg'][kg][key])<1e-14
        selection[kg]=val;per[kg]=evaluate(d,score,split='test',batch_size=256,output=kgout/'test_queries.npz')
    macro=macro_metrics({kg:r['metrics'][r['primary_filter']] for kg,r in per.items()})
    result={'status':'completed','purpose':'single_seed_frozen_test','dataset':a.dataset,'method':frozen['method'],'seed':17,
            'macro':macro,'per_kg':per,'selection_replay':selection,'checkpoint':frozen['checkpoint'],'checkpoint_sha256':frozen['checkpoint_sha256'],
            'freeze_sha256':sha256(frozen_path),'references_sha256':frozen['reference_registry_sha256'],'test_used_for_selection':False,
            'wall_seconds':time.monotonic()-t0,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'protocol':frozen['protocol']}
    atomic_json(out/'result.json',result)
    diffs={k:macro[k]-frozen['references'][k]['value'] for k in ['mrr','h1','h10']}
    prior=[]
    for item in frozen['predecessor_locks']:prior.append(json.loads((base/'single_seed'/item['dataset']/'LOCKED_RESULT.json').read_text()))
    wins=sum(v>0 for v in diffs.values())+sum(x['strict_wins'] for x in prior)
    deficits_ok=all(v>=-.005 for v in diffs.values()) and all(x['all_deficits_within_half_point'] for x in prior)
    remaining=12-3*(len(prior)+1);feasible=deficits_ok and wins+remaining>=10
    locked={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':a.dataset,'seed':17,'macro':macro,'differences':diffs,
            'strict_wins':sum(v>0 for v in diffs.values()),'all_deficits_within_half_point':all(v>=-.005 for v in diffs.values()),
            'cumulative_strict_wins':wins,'remaining_metrics':remaining,'can_still_reach_overall_threshold':feasible,
            'overall_threshold_met':remaining==0 and wins>=10 and deficits_ok,'result_sha256':sha256(out/'result.json'),
            'source_result':(out/'result.json').relative_to(ROOT).as_posix(),'comparison_status':'single_seed_provisional','no_further_test_or_tuning_for_this_dataset':True}
    atomic_json(folder/'LOCKED_RESULT.json',locked)
    with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
        f.write('\n## Dataset locked '+a.dataset+' — '+locked['timestamp']+'\nPurpose: lock once-only seed17 test before migration.\n')
        f.write('Config/seed/command/PID/GPU: frozen in freeze.json; seed17; evaluate_single.py --dataset '+a.dataset+'; PID '+str(os.getpid())+'; GPU '+str(os.environ.get('CUDA_VISIBLE_DEVICES'))+'.\n')
        f.write('Results: '+json.dumps(macro)+'; differences (0..1 units): '+json.dumps(diffs)+'; '+str(locked['strict_wins'])+'/3 strict wins.\n')
        f.write('Artifacts: '+str(folder.relative_to(ROOT))+'/evaluation and LOCKED_RESULT.json; raw query ranks retained.\n')
        f.write('Conclusion: result locked; no retraining or reselection for this dataset. Overall feasibility='+str(feasible)+'. Next: '+('stop and report final provisional verdict' if remaining==0 else ('migrate to next dataset with same algorithm, selecting only on its validation' if feasible else 'stop; locked tests make overall threshold unattainable; await user'))+'.\n')
    print(json.dumps(locked),flush=True)

if __name__=='__main__':main()
