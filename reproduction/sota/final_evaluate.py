"""One final acceptance evaluation per frozen dataset/seed. Never trains or selects."""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from frozen_data import ROOT,DOMAINS,atomic_json,sha256,load_multikg
from train_complex import Complex
sys.path.insert(0,str(ROOT/'reproduction/strict_baselines'))
from common import load_kg,evaluate,macro_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DOMAINS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    frozen_path=ROOT/'reproduction/sota/FINAL_EVALUATION_FREEZE.json'
    frozen=json.loads(frozen_path.read_text());entry=frozen['checkpoints'][f'{a.dataset}_s{a.seed}']
    assert sha256(ROOT/'reproduction/sota/PROVISIONAL_COMPARATORS_FREEZE.json')==frozen['comparators_sha256']
    for name,digest in frozen['source_hashes'].items():assert sha256(ROOT/name)==digest,(name,'Source changed after final freeze')
    checkpoint=ROOT/entry['checkpoint'];assert sha256(checkpoint)==entry['sha256']
    atomic_json(out/'config.json',{'arguments':vars(a),'frozen_checkpoint':entry,'evaluation_freeze_sha256':sha256(frozen_path),
        'comparators_sha256':frozen['comparators_sha256'],'protocol':frozen['protocol'],'purpose':'one_final_evaluation_no_selection'})
    marker=out/'TEST_OPENED.json'
    assert not marker.exists(),'This final evaluation was already opened; inspect retained artifacts, do not retest for selection'
    atomic_json(marker,{'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_sha256':entry['sha256'],'freeze_sha256':sha256(frozen_path),'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
    state=torch.load(checkpoint,map_location='cuda',weights_only=False);recipe=state['config']['recipe']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data,maps,roff,n,nr,audit=load_multikg(a.dataset,recipe['sharing'])
    model=Complex(n,nr,recipe['rank']).cuda();model.load_state_dict(state['model']);model.eval()
    maps={kg:torch.as_tensor(ids,device='cuda') for kg,ids in maps.items()}
    per={};selection={}
    for kg in DOMAINS[a.dataset]:
        # Test triples are first loaded only after the global configuration/checkpoint/reference freeze.
        d=load_kg(a.dataset,kg);mapping=maps[kg]
        def score(batch):
            b=torch.as_tensor(batch.copy(),device='cuda')
            return model.scores(mapping[b[:,0]],b[:,1]+roff[kg],mapping)
        kgout=out/kg;kgout.mkdir(exist_ok=True)
        # Validation replay audits the common evaluator without selecting another checkpoint.
        val=evaluate(d,score,split='val_select',batch_size=256,output=kgout/'selection_queries.npz')
        expected=entry['validation']['per_kg'][kg]
        for key in ['mrr','h1','h10']:assert abs(val['metrics']['select'][key]-expected[key])<1e-14,(kg,key,'Validation replay mismatch')
        selection[kg]=val
        per[kg]=evaluate(d,score,split='test',batch_size=256,output=kgout/'test_queries.npz')
    mm=macro_metrics({kg:r['metrics'][r['primary_filter']] for kg,r in per.items()})
    atomic_json(out/'result.json',{'status':'completed','purpose':'frozen_final_candidate_evaluation','method':frozen['method'],
        'dataset':a.dataset,'seed':a.seed,'macro':mm,'per_kg':per,'selection_replay':selection,'checkpoint_sha256':entry['sha256'],
        'checkpoint':entry['checkpoint'],'candidate_freeze_sha256':frozen['candidate_freeze_sha256'],'evaluation_freeze_sha256':sha256(frozen_path),
        'comparators_sha256':frozen['comparators_sha256'],'comparison_status':'provisional: unknown baselines, user reference placeholders or unmatched repeats remain',
        'protocol':'kbs-baselines-v1-20260905','test_used_for_selection':False})
    print(json.dumps({'dataset':a.dataset,'seed':a.seed,'macro':mm}),flush=True)

if __name__=='__main__':main()
