"""Official ATransN-released KGE equations with controlled KBS selection/evaluation.

Training retains alternating head/tail corruptions and the released losses.
Only held-out tail queries are used for model selection and final reporting.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import *

MODEL_PATH=ROOT/'reproduction/sources/ATransN/src/kge_model.py'
spec=importlib.util.spec_from_file_location('released_kge_model',MODEL_PATH)
released=importlib.util.module_from_spec(spec); spec.loader.exec_module(released)


def score_all(model, triples, chunk=2048):
    device=model.entity_embedding.device
    pos=torch.as_tensor(triples.copy(),device=device)
    scores=[]
    for start in range(0,model.num_entities,chunk):
        ids=torch.arange(start,min(start+chunk,model.num_entities),device=device)
        neg=ids[None,:].expand(len(triples),-1).contiguous()
        scores.append(model((pos,neg),mode='tail-batch'))
    return torch.cat(scores,dim=1)


def negative_batch(train, nentities, nrelations, sorted_keys, positive, mode, size):
    device=positive.device
    negative=torch.randint(nentities,(len(positive),size),device=device)
    def invalid(neg):
        h=neg if mode=='head-batch' else positive[:,0,None]
        t=neg if mode=='tail-batch' else positive[:,2,None]
        packed=(h*nrelations+positive[:,1,None])*nentities+t
        idx=torch.searchsorted(sorted_keys,packed)
        return (idx<sorted_keys.numel()) & (sorted_keys[idx.clamp_max(sorted_keys.numel()-1)]==packed)
    bad=invalid(negative)
    attempts=0
    while bad.any():
        negative[bad]=torch.randint(nentities,(int(bad.sum()),),device=device)
        bad=invalid(negative); attempts+=1
        if attempts>1000: raise RuntimeError('negative pool possibly empty')
    return negative


def run_domain(args,kg):
    out=Path(args.output)/kg; out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        return json.loads((out/'result.json').read_text(encoding='utf-8'))
    seed_all(args.seed)
    data=load_kg(args.dataset,kg)
    dim=100 if args.method=='RotatE' else 200
    margin=1. if args.method=='DistMult' else 8.
    if args.teacher: dim,margin=200,4.
    model=released.KGEModel(args.method,data['entities'],data['relations'],dim,margin,
                            double_entity_embedding=args.method=='RotatE').cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    train=torch.tensor(data['arrays']['train'],device='cuda')
    if args.smoke: train=train[:1024]
    sorted_keys=torch.unique((train[:,0]*data['relations']+train[:,1])*data['entities']+train[:,2],sorted=True)
    config={**vars(args),'kg':kg,'embedding_dim':dim,'margin':margin,'negative_samples':128,
            'learning_rate':.001,'training':'alternating head/tail; released loss; shuffled full-data epochs',
            'selection':'val_select filtered tail MRR; strictly greater; earliest tie',
            'selection_filter':'train+val_select','test_filter':'all' if args.dataset=='wk3l' else 'train+valid',
            'dataset_hash':data['manifest']['dataset_hash'],'environment':version_info(),
            'source_hashes':source_hashes([MODEL_PATH,Path(__file__),SUITE/'common.py'])}
    atomic_json(out/'config.json',config)
    best=-1.; best_step=0; start_step=1; orders={}; positions={'head-batch':0,'tail-batch':0}
    for mode in positions: orders[mode]=torch.randperm(len(train),device='cuda')
    last=out/'last.pt'
    if last.exists():
        state=torch.load(last,map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        best,best_step,start_step=state['best'],state['best_step'],state['step']+1
        orders,positions=state['orders'],state['positions']
        torch.set_rng_state(state['cpu_rng'].cpu()); torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    limit=64 if args.smoke else None
    log_path=out/'learning_curve.jsonl'
    with ResourceTrace(out/'resources.csv') as trace:
        model.train(); loss_sum=0.; interval_start=time.perf_counter()
        for step in range(start_step,args.steps+1):
            mode='tail-batch' if step%2 else 'head-batch'
            begin=positions[mode]
            if begin>=len(train):
                orders[mode]=torch.randperm(len(train),device='cuda'); begin=0
            indices=orders[mode][begin:begin+args.batch_size]
            positions[mode]=begin+len(indices); positive=train[indices]
            negative=negative_batch(train,data['entities'],data['relations'],sorted_keys,positive,mode,128)
            optimizer.param_groups[0]['lr']=.001*min(step/max(1,args.steps//100),1.)
            optimizer.zero_grad(set_to_none=True)
            pos_score=model(positive); neg_score=model((positive,negative),mode=mode)
            if args.method=='RotatE': loss=-F.logsigmoid(pos_score).mean()-F.logsigmoid(-neg_score).mean()
            else: loss=F.relu(-pos_score+neg_score+margin).mean()
            if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss at {step}')
            loss.backward(); optimizer.step(); loss_sum+=float(loss.detach())
            if step%100==0 or step==args.steps:
                with log_path.open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'step':step,'loss_mean':loss_sum/(100 if step%100==0 else step%100),
                                        'interval_seconds':time.perf_counter()-interval_start})+'\n')
                loss_sum=0.; interval_start=time.perf_counter()
            if step%args.eval_steps==0 or step==args.steps:
                model.eval(); torch.cuda.empty_cache()
                result=evaluate(data,lambda b:score_all(model,b),split='val_select',batch_size=64,limit=limit)
                val=result['metrics']['select']['mrr']
                with log_path.open('a',encoding='utf-8') as f: f.write(json.dumps({'step':step,'val_select':result})+'\n')
                if val>best:
                    best,best_step=val,step
                    atomic_checkpoint(out/'best.pt',{'model':model.state_dict(),'step':step,'validation_mrr':val,'config':config})
                atomic_checkpoint(last,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                    'best':best,'best_step':best_step,'orders':orders,'positions':positions,
                    'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
                print(f'{args.method} {args.dataset}/{kg} seed={args.seed} step={step} val_mrr={val:.6f} best={best:.6f}',flush=True)
                model.train(); interval_start=time.perf_counter()
        state=torch.load(out/'best.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']); model.eval()
        final=evaluate(data,lambda b:score_all(model,b),output=out/'test_queries.npz',limit=limit)
        val_final=evaluate(data,lambda b:score_all(model,b),split='val_select',output=out/'selection_queries.npz',limit=limit)
        if args.teacher:
            # Match the original ATransN teacher loader's checkpoint schema.
            torch.save({'learner':model.state_dict()},out/'checkpoint_valid.pt')
    result={'status':'smoke' if args.smoke else 'completed','method':args.method,'dataset':args.dataset,'kg':kg,
            'seed':args.seed,'best_step':best_step,'validation_mrr':best,'test':final,
            'checkpoint':str(out/'best.pt'),'checkpoint_sha256':sha256(out/'best.pt'),
            'resources':trace.summary,'parameter_count':sum(p.numel() for p in model.parameters()),
            'config':str(out/'config.json'),'dataset_hash':data['manifest']['dataset_hash']}
    atomic_json(out/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['TransE','DistMult','RotatE'],required=True)
    p.add_argument('--dataset',choices=DOMAINS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--eval-steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--smoke',action='store_true');p.add_argument('--teacher',action='store_true');p.add_argument('--kg')
    args=p.parse_args()
    domains=[args.kg] if args.kg else (['en'] if args.teacher else DOMAINS[args.dataset])
    results={kg:run_domain(args,kg) for kg in domains}
    aggregate={k:v['test']['metrics'][v['test']['primary_filter']] for k,v in results.items()}
    atomic_json(Path(args.output)/'result.json',{'status':'smoke' if args.smoke else 'completed',
        'method':args.method,'dataset':args.dataset,'seed':args.seed,'domains':results,'macro':macro_metrics(aggregate),
        'protocol':PROTOCOL_VERSION,'full_data':not args.smoke,'teacher':args.teacher})


if __name__=='__main__':main()
