"""Query Attention plus preserved Global/Random budget replay data.

Global and Random will match the frozen QURA validation access budget. Complete
prefix ranks are preserved for that later match without selecting on test
accuracy. No QURA predictor, threshold or certificate is trained here.
"""
import argparse
from pathlib import Path

import torch.nn.functional as F
from common import *
from internal_data import INTERNAL_VERSION,NegativeSampler,load_setting,hash64
from internal_models import FusionModel,QueryAttention,sampled_loss


def load_teacher(opts,kg):
    root=Path(opts.teacher_root);config=json.loads((root/kg/'config.json').read_text(encoding='utf-8'))['recipe']
    model=FusionModel(opts.dataset,kg,config['base_root'],config.get('target_base_root'),config['condition']).cuda()
    model.load_state_dict(torch.load(root/kg/'best.pt',map_location='cuda',weights_only=False)['model'])
    model.gamma=json.loads((root/'gamma_selection.json').read_text())['selected_gamma']
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


@torch.no_grad()
def feature_statistics(model,data):
    total=None;squares=None;count=0
    for start in range(0,len(data['arrays']['train']),1024):
        batch=torch.as_tensor(data['arrays']['train'][start:start+1024].copy(),device='cuda')
        values=model.raw_features(batch).reshape(-1,1281).double()
        if total is None:total=values.sum(0);squares=(values**2).sum(0)
        else:total+=values.sum(0);squares+=(values**2).sum(0)
        count+=len(values)
    mean=total/count;std=(squares/count-mean**2).clamp_min(0).sqrt().clamp_min(1e-6)
    model.feature_mean=mean.float();model.feature_std=std.float()


def train_attention(opts,model,data,out):
    out.mkdir(parents=True,exist_ok=True);gate=QueryAttention().cuda()
    if (out/'best.pt').exists() and (out/'training_complete.json').exists():
        state=torch.load(out/'best.pt',map_location='cuda',weights_only=False);gate.load_state_dict(state['gate'])
        model.feature_mean=state['feature_mean'];model.feature_std=state['feature_std'];gate.eval();return gate
    feature_statistics(model,data);optimizer=torch.optim.Adam(gate.parameters(),lr=.001)
    sampler=NegativeSampler(data,opts.seed+7100);rng=np.random.default_rng(opts.seed+7200)
    train=data['arrays']['train'];order=rng.permutation(len(train));position=0;best=-1.
    steps=32 if opts.smoke else 5000;interval=32 if opts.smoke else 1000
    with ResourceTrace(out/'training_resources.csv') as trace:
        for step in range(steps):
            if position>=len(order):order=rng.permutation(len(train));position=0
            ids=order[position:position+128];position+=len(ids);batch=train[ids]
            negative,valid=sampler.draw(batch);batch=batch[valid];negative=negative[valid]
            if not len(batch):continue
            triples=torch.as_tensor(batch.copy(),device='cuda');negative=torch.as_tensor(negative,device='cuda')
            optimizer.zero_grad(set_to_none=True)
            offset,*_=model.offset(triples,mode='attention',attention=gate)
            loss=sampled_loss(model.base,triples,negative,offset)
            if loss.requires_grad:loss.backward();optimizer.step()
            if (step+1)%100==0:
                with (out/'learning_curve.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'step':step+1,'loss':float(loss.detach())})+'\n')
            if (step+1)%interval==0 or step+1==steps:
                gate.eval();val=evaluate(data,lambda b:model.all_scores_numpy(b,mode='attention',attention=gate),split='val_select',limit=64 if opts.smoke else None)
                state={'gate':gate.state_dict(),'feature_mean':model.feature_mean,'feature_std':model.feature_std,
                    'optimizer':optimizer.state_dict(),'step':step+1,'val_select':val}
                atomic_checkpoint(out/'last.pt',state)
                if val['metrics']['select']['mrr']>best:best=val['metrics']['select']['mrr'];atomic_checkpoint(out/'best.pt',state)
                with (out/'learning_curve.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'step':step+1,'val_select':val})+'\n')
                gate.train()
    atomic_json(out/'training_complete.json',{'status':'completed','steps':steps,'resources':trace.summary,'resume':'restart incomplete gate training with fixed seed'})
    state=torch.load(out/'best.pt',map_location='cuda',weights_only=False);gate.load_state_dict(state['gate']);gate.eval()
    return gate


@torch.no_grad()
def global_utilities(opts,model,data,out):
    path=out/'training_loso_labels.npz'
    if path.exists():
        with np.load(path) as a:
            return np.array([a['delta'][:,i][a['available'][:,i]].mean() if a['available'][:,i].any() else -1e30 for i in range(len(model.sources))])
    train=data['arrays']['train'][:64] if opts.smoke else data['arrays']['train']
    sampler=NegativeSampler(data,opts.seed+8100);delta=[];full=[];removed=[];availability=[];negatives=[];valid_rows=[]
    for start in range(0,len(train),128):
        batch=train[start:start+128];negative,valid=sampler.draw(batch)
        # The datasets have many valid negatives; do not silently fabricate labels for empty pools.
        assert valid.all(),'Global utility label query has an empty negative pool'
        triples=torch.as_tensor(batch.copy(),device='cuda');negative_gpu=torch.as_tensor(negative,device='cuda')
        messages,mask,counts,_=model.messages(batch[:,0]);weights=model.weights(mask)
        def margin(weights):
            offset=model.gamma*(messages*weights[:,:,None]).sum(1)
            pos,neg=model.base.sampled_scores(triples,negative_gpu,offset)
            return pos-torch.logsumexp(neg,dim=1)
        all_margin=margin(weights);margins=[]
        for s in range(len(model.sources)):
            subset=mask.clone();subset[:,s]=False;margins.append(margin(model.weights(subset)))
        margins=torch.stack(margins,dim=1);effects=(all_margin[:,None]-margins)*mask
        delta.append(effects.cpu().numpy());full.append(all_margin.cpu().numpy());removed.append(margins.cpu().numpy())
        availability.append(mask.cpu().numpy());negatives.append(negative.astype(np.int32))
    values=np.concatenate(delta);available=np.concatenate(availability)
    np.savez_compressed(path,triples=train.astype(np.int32),query_index=np.arange(len(train)),sources=np.array(model.sources),
        delta=values,full_margin=np.concatenate(full),removed_margin=np.concatenate(removed),available=available,negative_ids=np.concatenate(negatives))
    means=np.array([values[:,i][available[:,i]].mean() if available[:,i].any() else -1e30 for i in range(len(model.sources))])
    atomic_json(out/'global_utility.json',{'source_order':model.sources,'mean_signed_loso':means.tolist(),
        'label_query_split':'train only','negative_scope':'fixed 256 valid sampled tails without replacement when possible',
        'gamma':model.gamma,'labels_sha256':sha256(path),'priority':'descending mean signed utility, source ID ties; equal fusion weights inside admitted prefix'})
    return means


def export_source_edges(model,data,out):
    """Compact exact K=32 records keyed by unique query head, not duplicated per query."""
    path=out/'source_edge_records.npz'
    if path.exists():return
    heads,inverse=np.unique(data['arrays']['test'][:,0],return_inverse=True)
    aligned,available,counts=model.availability(heads)
    records={'head_entity_id':heads,'query_to_head_row':inverse,'sources':np.array(model.sources),'aligned_entities':aligned}
    for i,s in enumerate(model.sources):
        edges=model.stores[s].fetch(np.where(available[:,i],aligned[:,i],-1))
        records.update({f'{s}_{key}':value for key,value in edges.items()})
    np.savez_compressed(path,**records)
    atomic_json(out/'source_edge_manifest.json',{'artifact_sha256':sha256(path),'query_count':len(inverse),'unique_heads':len(heads),
        'record_keys':'query index -> head row -> source -> direction/CSR offset/relation/neighbor/mask',
        'condition':model.condition,'k':32,'source_csr_manifests':{s:str(store.path/'manifest.json') for s,store in model.stores.items()}})


@torch.no_grad()
def prefix_replay(opts,model,data,out,global_means,split):
    path=out/f'{split}_budget_prefixes.npz'
    if path.exists():return
    triples=data['arrays'][split][:64] if opts.smoke else data['arrays'][split];ns=len(model.sources);repeats=10
    a=data['arrays']
    filters={'select':tail_map([a['train'],a['val_select']])} if split=='val_select' else {
        'train':tail_map([a['train']]),'train_valid':tail_map([a['train'],a['valid']]),'all':tail_map([a['train'],a['valid'],a['test']])}
    # Global is repetition 0; ten independently keyed random permutations follow.
    rank={k:np.zeros((repeats+1,len(triples),ns+1),dtype=np.int32) for k in filters}
    order_out=np.full((repeats+1,len(triples),ns),-1,dtype=np.int8)
    prefix_edges=np.zeros((repeats+1,len(triples),ns+1),dtype=np.int32)
    available_n=np.zeros(len(triples),dtype=np.int8);jitter=np.zeros((repeats+1,len(triples)),dtype=np.float64)
    priorities=np.lexsort((np.arange(ns),-global_means))
    for start in range(0,len(triples),64):
        batch=triples[start:start+64];ids=torch.as_tensor(batch.copy(),device='cuda')
        messages,mask,counts,_=model.messages(batch[:,0]);mask_np=mask.cpu().numpy();available_n[start:start+len(batch)]=mask_np.sum(1)
        base=model.base.all_scores(ids)
        for protocol,known in filters.items():rank[protocol][:,start:start+len(batch),0]=filtered_ranks(base,batch,known)
        for repeat in range(repeats+1):
            order=np.empty((len(batch),ns),dtype=np.int64)
            for row in range(len(batch)):
                qid=start+row
                keys=np.array([int(hash64(np.array([qid*101+s],dtype=np.uint64),seed=opts.seed+repeat*1009+9000)[0]) for s in range(ns)],dtype=np.uint64)
                local=priorities if repeat==0 else np.argsort(keys,kind='stable')
                local=np.r_[local[mask_np[row,local]],local[~mask_np[row,local]]];order[row]=local
                order_out[repeat,qid,:]=local
                jitter[repeat,qid]=int(hash64(np.array([qid],dtype=np.uint64),seed=opts.seed+repeat*1103+9100)[0])/(2**64)
            for length in range(1,ns+1):
                admitted=np.zeros_like(mask_np);np.put_along_axis(admitted,order[:,:length],True,axis=1);admitted&=mask_np
                prefix_edges[repeat,start:start+len(batch),length]=(counts*admitted).sum(1)
                weights=model.weights(torch.as_tensor(admitted,device='cuda'));offset=model.gamma*(messages*weights[:,:,None]).sum(1)
                scores=model.base.all_scores(ids,offset)
                for protocol,known in filters.items():rank[protocol][repeat,start:start+len(batch),length]=filtered_ranks(scores,batch,known)
    np.savez_compressed(path,triples=triples.astype(np.int32),query_index=np.arange(len(triples)),sources=np.array(model.sources),
        source_priority=order_out,prefix_edges=prefix_edges,available_sources=available_n,budget_rounding_jitter=jitter,
        **{f'rank_{k}':v for k,v in rank.items()})
    primary='select' if split=='val_select' else ('all' if opts.dataset=='wk3l' else 'train_valid')
    grid=[]
    for name,repeat_ids in [('Global-Utility',[0]),('Random',list(range(1,11)))]:
        for fraction in [0,.1,.2,.25,.4,.5,.6,.75,.8,.9,1.]:
            values=[]
            for repeat in repeat_ids:
                length=np.minimum(np.floor(available_n*fraction+jitter[0]).astype(int),available_n)
                r=rank[primary][repeat,np.arange(len(triples)),length];r0=rank[primary][repeat,:,0]
                values.append({'mrr':float((1/r).mean()),'ntr':float((r>r0).mean()),'coverage':float((length>0).mean()),
                    'edges':float(prefix_edges[repeat,np.arange(len(triples)),length].mean())})
            grid.append({'method':name,'budget_fraction':fraction,**{k:float(np.mean([v[k] for v in values])) for k in values[0]}})
    atomic_json(out/f'{split}_budget_grid.json',{'split':split,'entries':grid,'primary_filter':primary,
        'status':'budget scan only; final budget must match frozen QURA validation access',
        'cost':'edge counts of replayed admission policy; this offline scan prefetches all messages and is not a latency measurement'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DOMAINS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--teacher-root',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--skip-robustness',action='store_true',help='Run Table 4 controls only; defer Table 7 perturbations')
    opts=p.parse_args();out=Path(opts.output).resolve();out.mkdir(parents=True,exist_ok=True);seed_all(opts.seed)
    teacher_result=json.loads((Path(opts.teacher_root)/'result.json').read_text());condition=teacher_result.get('condition','full')
    atomic_json(out/'config.json',{'version':INTERNAL_VERSION,'dataset':opts.dataset,'seed':opts.seed,'condition':condition,'teacher_root':opts.teacher_root,
        'teacher_result_sha256':sha256(Path(opts.teacher_root)/'result.json'),'robustness_enabled':not opts.skip_robustness,'attention_input':'frozen target entity 128 + relation 128 + current source log1p sketch 512 + mean other log1p sketches 512 + other-empty bit',
        'attention_network':'1281 -> Linear(256) -> GELU -> Linear(1); train-only feature normalization; masked softmax across all available sources',
        'attention_objective':'same sampled KG hinge loss; no LOSO supervision; teacher frozen; val_select MRR selection',
        'global':'mean signed frozen-teacher train LOSO priority; equal weights for the admitted prefix',
        'random':'10 deterministic independent source-permutation repeats nested within each model seed',
        'budget_matching':'pending frozen QURA val_select access budget; all prefix ranks and rounding jitters retained',
        'environment':version_info(),'source_hashes':source_hashes([Path(__file__),SUITE/'internal_models.py',SUITE/'internal_data.py',SUITE/'common.py'])})
    results={};paired={};dependencies=[];robust={}
    for kg in DOMAINS[opts.dataset]:
        seed_all(opts.seed);model=load_teacher(opts,kg);data=load_setting(opts.dataset,kg,condition)
        kgout=out/'attention'/kg;kgout.mkdir(parents=True,exist_ok=True)
        if condition=='full':
            gate=train_attention(opts,model,data,kgout)
            final=evaluate(data,lambda b:model.all_scores_numpy(b,mode='attention',attention=gate),output=kgout/'test_queries.npz',limit=64 if opts.smoke else None)
            baseline=evaluate(data,lambda b:model.all_scores_numpy(b,mode='target'),output=kgout/'target_reference_queries.npz',limit=64 if opts.smoke else None)
            evaluate(data,lambda b:model.all_scores_numpy(b,mode='attention',attention=gate),split='val_select',output=kgout/'selection_queries.npz',limit=64 if opts.smoke else None)
            weights=[]
            with torch.no_grad():
                for start in range(0,final['count'],128):
                    batch=data['arrays']['test'][start:min(start+128,final['count'])];t=torch.as_tensor(batch.copy(),device='cuda')
                    aligned,available,counts=model.availability(batch[:,0]);mask=torch.as_tensor(available,device='cuda')
                    weights.append(model.weights(mask,'attention',gate,model.features(t)).cpu().numpy())
            aligned,available,counts=model.availability(data['arrays']['test'][:final['count'],0])
            np.savez_compressed(kgout/'source_trace.npz',query_index=np.arange(final['count']),aligned_entities=aligned,
                available=available,weights=np.concatenate(weights),edges=counts,sources=np.array(model.sources))
            with np.load(kgout/'test_queries.npz') as f,np.load(kgout/'target_reference_queries.npz') as b:
                key='rank_'+final['primary_filter'];r=f[key];r0=b[key]
                paired[kg]={'mrr':float((1/r).mean()),'ntr_all':float((r>r0).mean()),'ptr':float((r<r0).mean()),
                    'mean_harm':float(np.maximum(1/r0-1/r,0).mean()),'coverage':float(available.any(1).mean()),'edges':float(counts.sum(1).mean())}
            results[kg]={'test':final,'checkpoint':str(kgout/'best.pt'),'checkpoint_sha256':sha256(kgout/'best.pt')}
            dependencies.append({'path':str(Path(opts.teacher_root)/kg/'best.pt'),'sha256':sha256(Path(opts.teacher_root)/kg/'best.pt')})
            del gate
        budget=out/'budgets'/kg;budget.mkdir(parents=True,exist_ok=True)
        export_source_edges(model,data,budget)
        utility=global_utilities(opts,model,data,budget)
        for split in ['val_select','test']:prefix_replay(opts,model,data,budget,utility,split)
        if opts.dataset=='dbp5l' and condition=='full' and not opts.skip_robustness:
            conditions=['corrupt10','corrupt20','corrupt40']
            if kg in ['el','ja']:conditions += [f'{kg}_one',f'{kg}_two',f'{kg}_all']
            for perturbation in conditions:
                model.set_condition(perturbation if perturbation.startswith('corrupt') else 'full')
                visible=list(model.sources)
                if '_' in perturbation:
                    count={'one':1,'two':2,'all':len(model.sources)}[perturbation.split('_')[1]]
                    visible=[s for s in ['en','fr','es','ja','el'] if s in model.sources][:count]
                    for s in model.sources:
                        if s not in visible:model.mapping[s]=np.full_like(model.mapping[s],-1)
                dest=out/'robustness'/perturbation/kg;dest.mkdir(parents=True,exist_ok=True)
                export_source_edges(model,data,dest)
                final=evaluate(data,model.all_scores_numpy,output=dest/'test_queries.npz',limit=64 if opts.smoke else None)
                evaluate(data,lambda b:model.all_scores_numpy(b,mode='target'),output=dest/'target_reference_queries.npz',limit=64 if opts.smoke else None)
                aligned,available,counts=model.availability(data['arrays']['test'][:final['count'],0])
                np.savez_compressed(dest/'source_trace.npz',query_index=np.arange(final['count']),aligned_entities=aligned,
                    available=available,edges=counts,sources=np.array(model.sources),visible_sources=np.array(visible))
                with np.load(dest/'test_queries.npz') as f,np.load(dest/'target_reference_queries.npz') as b:
                    key='rank_'+final['primary_filter'];r=f[key];r0=b[key]
                    pair={'mrr':float((1/r).mean()),'ntr_all':float((r>r0).mean()),'ptr':float((r<r0).mean()),
                        'mean_harm':float(np.maximum(1/r0-1/r,0).mean()),'coverage':float(available.any(1).mean()),'edges':float(counts.sum(1).mean())}
                entry=robust.setdefault(perturbation,{'domains':{},'paired':{}})
                checkpoint=Path(opts.teacher_root)/kg/'best.pt'
                entry['domains'][kg]={'test':final,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint)}
                entry['paired'][kg]=pair
                atomic_json(dest/'perturbation.json',{'condition':perturbation,'visible_sources':visible,'teacher_checkpoint':str(checkpoint),
                    'teacher_sha256':sha256(checkpoint),'gamma':model.gamma,'checkpoint_retrained':False,'corruption_seed':20260905})
                for split in ['val_select','test']:prefix_replay(opts,model,data,dest,utility,split)
        del model;torch.cuda.empty_cache()
    if results:
        atomic_json(out/'attention/result.json',{'status':'completed','full_data':not opts.smoke,'method':'Query Attention','dataset':opts.dataset,
            'seed':opts.seed,'condition':condition,'domains':results,'paired':paired,
            'macro':macro_metrics({kg:r['test']['metrics'][r['test']['primary_filter']] for kg,r in results.items()}),
            'paired_macro':{k:float(np.mean([v[k] for v in paired.values()])) for k in next(iter(paired.values()),{})},
            'dependencies':dependencies,'protocol':INTERNAL_VERSION})
    for perturbation,entry in robust.items():
        atomic_json(out/'robustness'/perturbation/'result.json',{'status':'completed','full_data':not opts.smoke,'method':'Uniform-All',
            'dataset':opts.dataset,'condition':perturbation,'seed':opts.seed,**entry,
            'macro':macro_metrics({kg:r['test']['metrics'][r['test']['primary_filter']] for kg,r in entry['domains'].items()}),
            'paired_macro':{k:float(np.mean([v[k] for v in entry['paired'].values()])) for k in next(iter(entry['paired'].values()),{})},
            'protocol':INTERNAL_VERSION})
    atomic_json(out/'result.json',{'status':'completed','full_data':not opts.smoke,'publication':False,'kind':'internal-control-bundle',
        'dataset':opts.dataset,'seed':opts.seed,'condition':condition,'attention_complete':bool(results),
        'robustness_skipped':opts.skip_robustness,
        'global_random_status':'full prefix results saved; table values await matching to the frozen QURA validation budget'})


if __name__=='__main__':main()
