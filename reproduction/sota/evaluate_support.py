"""Once-only evaluation of every registered control, with exact validation replay."""
import argparse
import datetime
import json
import os
import sys
import time
import torch
from frozen_data import ROOT,DOMAINS,atomic_json,sha256
from support_data import build_inputs
from train_complex import Complex
sys.path.insert(0,str(ROOT/'reproduction/strict_baselines'))
from common import load_kg,evaluate,macro_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);ident=p.parse_args().id
    phase=ROOT/'reproduction/sota/paper_support';fp=phase/'EVALUATION_FREEZE.json';freeze=json.loads(fp.read_text());e=freeze['models'][ident]
    assert not e['reuse_locked_test']
    for path,digest in freeze['source_hashes'].items():assert sha256(ROOT/path)==digest,path
    assert sha256(ROOT/'reproduction/sota/three_seed/RESULTS.json')==freeze['table2_results_sha256']
    cp=ROOT/e['training_path']/'best.pt';assert sha256(cp)==e['checkpoint_sha256']
    out=ROOT/e['evaluation_path'];out.mkdir(parents=True,exist_ok=True)
    assert not (out/'result.json').exists()
    with (out/'TEST_OPENED.json').open('x',encoding='utf-8') as f:
        json.dump({'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha256(fp),
            'checkpoint_sha256':e['checkpoint_sha256'],'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')},f,indent=2)
    atomic_json(out/'config.json',{'freeze_sha256':sha256(fp),'entry':e,'protocol':freeze['protocol']})
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    recipe=e['recipe'];_,maps,roff,n,nr,audit,_,_=build_inputs(recipe)
    assert audit['entity_map_sha256']==e['entity_map_sha256']
    maps={k:torch.as_tensor(v,device='cuda') for k,v in maps.items()}
    saved=torch.load(cp,map_location='cpu',weights_only=False);model=Complex(n,nr,recipe['rank']).cuda()
    model.load_state_dict(saved['model']);del saved;model.eval();per={};selection={};start=time.monotonic()
    for kg in DOMAINS[e['dataset']]:
        data=load_kg(e['dataset'],kg);mapping=maps[kg]
        def score(batch):
            b=torch.as_tensor(batch.copy(),device='cuda');return model.scores(mapping[b[:,0]],b[:,1]+roff[kg],mapping)
        kgout=out/kg;kgout.mkdir(exist_ok=True)
        val=evaluate(data,score,split='val_select',batch_size=256,output=kgout/'selection_queries.npz')
        for k in ['mrr','h1','h10']:assert abs(val['metrics']['select'][k]-e['validation']['per_kg'][kg][k])<1e-14,(ident,kg,k)
        selection[kg]=val;per[kg]=evaluate(data,score,split='test',batch_size=256,output=kgout/'test_queries.npz')
    macro=macro_metrics({kg:v['metrics'][v['primary_filter']] for kg,v in per.items()})
    atomic_json(out/'result.json',{'status':'completed','purpose':'registered_support_frozen_test','method':'ASRC control','id':ident,
        'dataset':e['dataset'],'seed':e['seed'],'macro':macro,'per_kg':per,'selection_replay':selection,'checkpoint_sha256':e['checkpoint_sha256'],
        'freeze_sha256':sha256(fp),'protocol':freeze['protocol'],'test_used_for_selection':False,
        'wall_seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
    print(json.dumps({'id':ident,'macro':macro}),flush=True)

if __name__=='__main__':main()
