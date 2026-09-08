"""Train shared Target-only and frozen Uniform-All controls, without a QURA gate."""
import argparse
from pathlib import Path

from common import *
from internal_data import INTERNAL_VERSION,NegativeSampler,load_setting,source_kgs
from internal_models import TargetModel,FusionModel,sampled_loss


def train_one(opts,kg):
    out=Path(opts.output)/kg;out.mkdir(parents=True,exist_ok=True);seed_all(opts.seed)
    data=load_setting(opts.dataset,kg,opts.condition)
    if opts.stage=='base':model=TargetModel(data['entities'],data['relations']).cuda()
    else:model=FusionModel(opts.dataset,kg,opts.base_root,opts.target_base_root,opts.condition).cuda()
    config={'version':INTERNAL_VERSION,'method':'Target-only' if opts.stage=='base' else 'Uniform-All','dataset':opts.dataset,
        'kg':kg,'seed':opts.seed,'condition':opts.condition,'recipe':vars(opts),'data_manifest':data['manifest'],
        'dimension':128,'score':'negative Euclidean distance of normalized entity head + relation + additive source message',
        'loss':'mean hinge(margin=.5), 256 filtered negatives; without replacement unless pool<256',
        'teacher':'frozen Target-only/source bases; train directional edge encoder and per-source projections; gamma=1 during training',
        'environment':version_info(),'source_hashes':source_hashes([Path(__file__),SUITE/'internal_data.py',SUITE/'internal_models.py',SUITE/'common.py'])}
    atomic_json(out/'config.json',config)
    if (out/'training_complete.json').exists():
        model.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=False)['model']);model.eval();return model,data
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    sampler=NegativeSampler(data,opts.seed+5000);order_rng=np.random.default_rng(opts.seed+6000)
    train=data['arrays']['train'];order=order_rng.permutation(len(train));position=0;start=0;best=-1.
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step'];best=state['best_validation_mrr']
        sampler.rng.bit_generator.state=state['negative_rng'];order_rng.bit_generator.state=state['order_rng'];order=state['order'];position=state['position']
        torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    steps=32 if opts.smoke else (10000 if opts.stage=='base' else 5000)
    interval=32 if opts.smoke else (2000 if opts.stage=='base' else 1000)
    def score(batch):
        if opts.stage=='base':return model.all_scores(torch.as_tensor(batch.copy(),device='cuda'))
        return model.all_scores_numpy(batch)
    with ResourceTrace(out/'training_resources.csv') as trace:
        model.train()
        for step in range(start,steps):
            if position>=len(order):order=order_rng.permutation(len(train));position=0
            index=order[position:position+128];position+=len(index);batch=train[index]
            negatives,valid=sampler.draw(batch);batch=batch[valid];negatives=negatives[valid]
            if not len(batch):continue
            triples=torch.as_tensor(batch.copy(),device='cuda');negative=torch.as_tensor(negatives,device='cuda')
            optimizer.zero_grad(set_to_none=True)
            if opts.stage=='base':loss=sampled_loss(model,triples,negative)
            else:
                offset,*_=model.offset(triples);loss=sampled_loss(model.base,triples,negative,offset)
            if loss.requires_grad:loss.backward();optimizer.step()
            if (step+1)%100==0:
                with (out/'learning_curve.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'step':step+1,'loss':float(loss.detach()),'learning_rate':optimizer.param_groups[0]['lr']})+'\n')
            if (step+1)%interval==0 or step+1==steps:
                model.eval()
                with torch.random.fork_rng(devices=[0]):val=evaluate(data,score,split='val_select',limit=64 if opts.smoke else None)
                mrr=val['metrics']['select']['mrr'];improved=mrr>best
                if improved:best=mrr
                state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'best_validation_mrr':best,'validation_mrr':mrr,
                    'negative_rng':sampler.rng.bit_generator.state,'order_rng':order_rng.bit_generator.state,'order':order,'position':position,
                    'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
                atomic_checkpoint(out/'last.pt',state)
                if improved:atomic_checkpoint(out/'best.pt',state)
                with (out/'learning_curve.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'step':step+1,'val_select':val,'best_mrr':best})+'\n')
                print(f'{opts.stage} {opts.dataset}/{kg} {opts.condition} s{opts.seed} step={step+1} val={mrr:.6f} best={best:.6f}',flush=True)
                model.train()
    atomic_json(out/'training_complete.json',{'status':'completed','steps':steps,'resources':trace.summary,'smoke':opts.smoke})
    model.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=False)['model']);model.eval()
    return model,data


def source_trace(model,data,out):
    triples=data['arrays']['test'];aligned,mask,counts=model.availability(triples[:,0])
    weights=mask.astype(np.float32)/np.maximum(mask.sum(1,keepdims=True),1)
    np.savez_compressed(out/'source_trace.npz',query_index=np.arange(len(triples)),aligned_entities=aligned,
        available=mask,weights=weights,edges=counts,sources=np.array(model.sources))
    return {'coverage':float(mask.any(1).mean()),'edges':float(counts.sum(1).mean()),'sources':float(mask.sum(1).mean())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['base','teacher'],required=True)
    p.add_argument('--dataset',choices=DOMAINS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--condition',choices=['full','target20','align20'],default='full');p.add_argument('--output',required=True)
    p.add_argument('--base-root');p.add_argument('--target-base-root');p.add_argument('--smoke',action='store_true')
    opts=p.parse_args();out=Path(opts.output).resolve();out.mkdir(parents=True,exist_ok=True)
    domains=source_kgs(opts.dataset) if opts.stage=='base' and opts.condition=='full' else DOMAINS[opts.dataset]
    for kg in domains:
        model,data=train_one(opts,kg);del model,data;torch.cuda.empty_cache()
    gamma=1.
    if opts.stage=='teacher':
        grid={str(g):{} for g in [.25,.5,1.]}
        for kg in domains:
            model,data=train_one(opts,kg)
            for g in [.25,.5,1.]:
                model.gamma=g
                grid[str(g)][kg]=evaluate(data,model.all_scores_numpy,split='val_select',limit=64 if opts.smoke else None)['metrics']['select']
            del model,data;torch.cuda.empty_cache()
        scores={g:macro_metrics(per)['mrr'] for g,per in grid.items()}
        gamma=min([.25,.5,1.],key=lambda g:(-scores[str(g)],g))
        atomic_json(out/'gamma_selection.json',{'candidates':grid,'macro_mrr':scores,'selected_gamma':gamma,'rule':'maximum val_select macro MRR; smaller gamma on exact tie'})
    results={};paired={}
    for kg in domains:
        model,data=train_one(opts,kg);kgout=out/kg
        if opts.stage=='teacher':model.gamma=gamma;score=model.all_scores_numpy
        else:score=lambda batch:model.all_scores(torch.as_tensor(batch.copy(),device='cuda'))
        with ResourceTrace(kgout/'evaluation_resources.csv') as trace:
            final=evaluate(data,score,output=kgout/'test_queries.npz',limit=64 if opts.smoke else None)
            selection=evaluate(data,score,split='val_select',output=kgout/'selection_queries.npz',limit=64 if opts.smoke else None)
        results[kg]={'test':final,'selection':selection,'checkpoint':str(kgout/'best.pt'),'checkpoint_sha256':sha256(kgout/'best.pt'),
            'config':str(kgout/'config.json'),'dataset_hash':data['manifest']['dataset_hash'],'resources':trace.summary,
            'parameter_count':sum(p.numel() for p in model.parameters())}
        if opts.stage=='teacher':
            baseline=evaluate(data,lambda b:model.all_scores_numpy(b,mode='target'),output=kgout/'target_reference_queries.npz',limit=64 if opts.smoke else None)
            with np.load(kgout/'test_queries.npz') as f,np.load(kgout/'target_reference_queries.npz') as b:
                key='rank_'+final['primary_filter'];rank=f[key];r0=b[key]
                access=source_trace(model,data,kgout)
                paired[kg]={'mrr':float((1/rank).mean()),'ntr_all':float((rank>r0).mean()),'ptr':float((rank<r0).mean()),
                    'mean_harm':float(np.maximum(1/r0-1/rank,0).mean()),**access}
        atomic_json(kgout/'result.json',results[kg]);del model,data;torch.cuda.empty_cache()
    public_domains={kg:results[kg] for kg in DOMAINS[opts.dataset]}
    atomic_json(out/'result.json',{'status':'completed','purpose':'smoke' if opts.smoke else 'formal','full_data':not opts.smoke,
        'method':'Target-only' if opts.stage=='base' else 'Uniform-All','dataset':opts.dataset,'condition':opts.condition,
        'seed':opts.seed,'domains':public_domains,'macro':macro_metrics({kg:r['test']['metrics'][r['test']['primary_filter']] for kg,r in public_domains.items()}),
        'paired':paired,'paired_macro':{k:float(np.mean([v[k] for v in paired.values()])) for k in next(iter(paired.values()),{})},
        'protocol':INTERNAL_VERSION,'gamma':gamma})


if __name__=='__main__':main()
