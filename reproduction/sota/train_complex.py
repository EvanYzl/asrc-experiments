"""Validation-only reciprocal ComplEx/N3 pilot or confirmation training."""
import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from frozen_data import ROOT,DOMAINS,atomic_json,sha256,load_multikg

class Complex(torch.nn.Module):
    def __init__(self,n,nr,rank):
        super().__init__();self.rank=rank
        self.entity=torch.nn.Embedding(n,2*rank)
        self.relation=torch.nn.Embedding(2*nr,2*rank)
        with torch.no_grad():self.entity.weight.normal_(0,.001);self.relation.weight.normal_(0,.001)
    def query(self,h,r):
        a,b=self.entity(h).chunk(2,dim=-1);c,d=self.relation(r).chunk(2,dim=-1)
        return torch.cat([a*c-b*d,a*d+b*c],dim=-1)
    def scores(self,h,r,candidates):return self.query(h,r)@self.entity(candidates).T
    def n3(self,h,r,t):
        return sum(((x[:,:self.rank]**2+x[:,self.rank:]**2+1e-15)**1.5).sum()/len(h)
                   for x in [self.entity(h),self.relation(r),self.entity(t)])

def metric(ranks):
    r=np.asarray(ranks,dtype=np.float64)
    return {k:float(v) for k,v in {'mrr':np.mean(1/r),'h1':np.mean(r==1),'h10':np.mean(r<=10)}.items()}

@torch.no_grad()
def validate(model,data,maps,roff,targets,output=None):
    model.eval();per={}
    for kg in targets:
        d=data[kg];a=d['arrays']['val_select'];cand=maps[kg]
        known={}
        for arr in [d['arrays']['train'],a]:
            for h,r,t in arr:known.setdefault((int(h),int(r)),set()).add(int(t))
        ranks=[];gold_scores=[];ids=torch.arange(d['entities'],device='cuda')
        for start in range(0,len(a),256):
            b=a[start:start+256];tr=torch.as_tensor(b,device='cuda')
            scores=model.scores(cand[tr[:,0]],tr[:,1]+roff[kg],cand)
            assert torch.isfinite(scores).all()
            gold=scores.gather(1,tr[:,2,None]).squeeze(1)
            ahead=(scores>gold[:,None])|((scores==gold[:,None])&(ids[None,:]<tr[:,2,None]))
            rows=[];cols=[]
            for i,(h,r,t) in enumerate(b):
                tails=known[(int(h),int(r))];rows.extend([i]*len(tails));cols.extend(tails)
            ahead[rows,cols]=False
            ahead.scatter_(1,tr[:,2,None],False)
            ranks.append((1+ahead.sum(1)).cpu().numpy());gold_scores.append(gold.cpu().numpy())
        ranks=np.concatenate(ranks);per[kg]=metric(ranks)
        if output is not None:
            np.savez_compressed(Path(output)/f'{kg}_validation.npz',triples=a,rank_select=ranks,gold_score=np.concatenate(gold_scores))
    macro={k:float(np.mean([v[k] for v in per.values()])) for k in ['mrr','h1','h10']}
    model.train();return {'macro':macro,'per_kg':per}

def checkpoint(path,payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DOMAINS,required=True)
    p.add_argument('--sharing',choices=['shared','independent'],default='shared');p.add_argument('--rank',type=int,default=256)
    p.add_argument('--reg',type=float,default=.01);p.add_argument('--lr',type=float,default=.1)
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--valid-every',type=int,default=5);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--output',required=True)
    p.add_argument('--resume');args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    assert not (out/'result.json').exists(),'Refuse overwrite of completed experiment'
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data,entity_maps,roff,n,nr,audit=load_multikg(args.dataset,args.sharing)
    maps={k:torch.as_tensor(v,device='cuda') for k,v in entity_maps.items()}
    model=Complex(n,nr,args.rank).cuda();optimizer=torch.optim.Adagrad(model.parameters(),lr=args.lr)
    config={'method':'Aligned reciprocal ComplEx-N3','recipe':vars(args),'alignment':audit,
            'protocol':'kbs-baselines-v1-20260905','selection':'val_select macro tail MRR, train+val_select filter, full candidates, ascending entity-ID ties; earliest checkpoint on tie',
            'input':'published train triples and supplied alignments only; reciprocal train augmentation',
            'test_access':False,'manifests':{kg:d['manifest_sha256'] for kg,d in data.items()},
            'source_hashes':{p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('frozen_data.py')]},
            'environment':{'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'pid':os.getpid()}}
    atomic_json(out/'config.json',config);atomic_json(out/'alignment_audit.json',audit)
    arrays={}
    for kg,d in data.items():
        a=d['arrays']['train'].copy();inverse=a[:,[2,1,0]].copy();inverse[:,1]+=nr
        both=np.concatenate([a,inverse]);both[:,1]+=roff[kg]
        arrays[kg]=torch.as_tensor(both,device='cuda')
    start=0;best=-1.;best_epoch=0;stale=0;t0=time.monotonic();best_metrics=None
    if args.resume:
        saved=torch.load(args.resume,map_location='cuda',weights_only=False)
        old=saved['config']['recipe']
        for k in ['dataset','sharing','rank','reg','lr','batch_size','seed']:assert old[k]==getattr(args,k)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['epoch']
        torch.set_rng_state(saved['cpu_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        best=saved['best'];best_epoch=saved['best_epoch'];best_metrics=saved['best_metrics']
        old_best=Path(args.resume).with_name('best.pt')
        if old_best.resolve()!=(out/'best.pt').resolve():
            import shutil
            shutil.copy2(old_best,out/'best.pt')
    for epoch in range(start+1,args.epochs+1):
        et=time.monotonic();order=[];shuffled={}
        for kg,a in arrays.items():
            shuffled[kg]=a[torch.randperm(len(a),device='cuda')]
            order.extend((kg,i) for i in range(0,len(a),args.batch_size))
        # Use Torch RNG so resumption preserves batch order exactly.
        permutation=torch.randperm(len(order)).tolist();loss_sum=0.;n_batch=0
        for ix in permutation:
            kg,start_batch=order[ix];batch=shuffled[kg][start_batch:start_batch+args.batch_size];mapping=maps[kg]
            h=mapping[batch[:,0]];r=batch[:,1];t=mapping[batch[:,2]]
            scores=model.scores(h,r,mapping)
            loss=F.cross_entropy(scores,batch[:,2])+args.reg*model.n3(h,r,t)
            assert torch.isfinite(loss),'Nonfinite training objective'
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();loss_sum+=float(loss.detach());n_batch+=1
        row={'epoch':epoch,'loss':loss_sum/n_batch,'train_seconds':time.monotonic()-et,'elapsed_seconds':time.monotonic()-t0}
        if epoch%args.valid_every==0 or epoch==args.epochs:
            val=validate(model,data,maps,roff,DOMAINS[args.dataset]);row['validation']=val
            improved=val['macro']['mrr']>best
            if improved:best=val['macro']['mrr'];best_epoch=epoch;best_metrics=val;stale=0
            else:stale+=1
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,
                   'best_metrics':best_metrics,'config':config,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
            checkpoint(out/'last.pt',state)
            if improved:checkpoint(out/'best.pt',state)
            atomic_json(out/'progress.json',{'status':'running','epoch':epoch,'best_epoch':best_epoch,'best_validation':best_metrics,'elapsed_seconds':time.monotonic()-t0})
            print(json.dumps({'epoch':epoch,'validation':val['macro'],'best_epoch':best_epoch,'seconds':round(time.monotonic()-t0,1)}),flush=True)
        with (out/'learning_curve.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if stale>=args.patience:break
    saved=torch.load(out/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['model'])
    final=validate(model,data,maps,roff,DOMAINS[args.dataset],out)
    assert final['macro']==best_metrics['macro']
    result={'status':'completed','purpose':'validation_only','test_access':False,'dataset':args.dataset,'seed':args.seed,'validation':final,
            'best_epoch':best_epoch,'completed_epochs':epoch,'checkpoint_sha256':sha256(out/'best.pt'),'config_sha256':sha256(out/'config.json'),
            'wall_seconds':time.monotonic()-t0,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
    atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
