"""One test of an existing validation-selected checkpoint for a missing seed."""
import argparse
import datetime
import json
import os
import sys
import time
import torch
from frozen_data import ROOT,DOMAINS,atomic_json,sha256,load_multikg
from train_complex import Complex
sys.path.insert(0,str(ROOT/'reproduction/strict_baselines'))
from common import load_kg,evaluate,macro_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DOMAINS,required=True);p.add_argument('--seed',type=int,choices=[29,43],required=True);args=p.parse_args()
    base=ROOT/'reproduction/sota';fp=base/'three_seed/freeze.json';freeze=json.loads(fp.read_text());e=freeze['entries'][f'{args.dataset}_s{args.seed}']
    assert json.loads((base/'TASK_SCOPE.json').read_text())['phase']=='three_seed_confirmation'
    for path,digest in freeze['source_hashes'].items():assert sha256(ROOT/path)==digest
    assert sha256(ROOT/'reproduction/strict_baselines/common.py')==freeze['common_sha256']
    assert sha256(base/'SINGLE_SEED_REFERENCES.json')==freeze['reference_registry_sha256']
    cp=ROOT/e['checkpoint'];assert sha256(cp)==e['checkpoint_sha256']
    out=(ROOT/e['evaluation_result']).parent;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'result.json').exists()
    with (out/'TEST_OPENED.json').open('x',encoding='utf-8') as f:
        json.dump({'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha256(fp),'checkpoint_sha256':e['checkpoint_sha256'],
            'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')},f,indent=2)
    atomic_json(out/'config.json',{'freeze_sha256':sha256(fp),'entry':e,'protocol':freeze['protocol']})
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(cp,map_location='cuda',weights_only=False);recipe=saved['config']['recipe']
    _,maps,roff,n,nr,_=load_multikg(args.dataset,recipe['sharing']);maps={k:torch.as_tensor(v,device='cuda') for k,v in maps.items()}
    model=Complex(n,nr,recipe['rank']).cuda();model.load_state_dict(saved['model']);model.eval();per={};selection={};start=time.monotonic()
    for kg in DOMAINS[args.dataset]:
        data=load_kg(args.dataset,kg);mapping=maps[kg]
        def score(batch):
            b=torch.as_tensor(batch.copy(),device='cuda');return model.scores(mapping[b[:,0]],b[:,1]+roff[kg],mapping)
        kgout=out/kg;kgout.mkdir(exist_ok=True)
        val=evaluate(data,score,split='val_select',batch_size=256,output=kgout/'selection_queries.npz')
        for k in ['mrr','h1','h10']:assert abs(val['metrics']['select'][k]-e['validation']['per_kg'][kg][k])<1e-14
        selection[kg]=val;per[kg]=evaluate(data,score,split='test',batch_size=256,output=kgout/'test_queries.npz')
    macro=macro_metrics({kg:v['metrics'][v['primary_filter']] for kg,v in per.items()})
    atomic_json(out/'result.json',{'status':'completed','purpose':'three_seed_frozen_test','method':'ASRC','dataset':args.dataset,'seed':args.seed,
        'macro':macro,'per_kg':per,'selection_replay':selection,'checkpoint':e['checkpoint'],'checkpoint_sha256':e['checkpoint_sha256'],
        'freeze_sha256':sha256(fp),'references_sha256':freeze['reference_registry_sha256'],'protocol':freeze['protocol'],
        'test_used_for_selection':False,'wall_seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
    print(json.dumps({'dataset':args.dataset,'seed':args.seed,'macro':macro}),flush=True)

if __name__=='__main__':main()
