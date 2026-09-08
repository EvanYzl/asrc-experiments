"""Local KEnS reruns, with validation-only ensemble weight fitting.

Runs in the isolated TensorFlow 2.10 environment. No PyTorch dependency.
The released nomination-specific Hits@n is preserved; full-ranking MRR is
undefined for this interface and is deliberately not manufactured.
"""
import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DOMAINS={'dbp5l':['el','en','es','fr','ja'],'depkg':['de','es','fr','it','jp','uk']}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DOMAINS,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--target');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if args.target:
        run_target(args,out);return
    per_kg={};started=time.time()
    for kg in DOMAINS[args.dataset]:
        result=out/kg/'result.json'
        if not result.exists():
            cmd=[sys.executable,__file__,'--dataset',args.dataset,'--seed',str(args.seed),'--output',str(out/kg),'--target',kg]
            if args.smoke:cmd.append('--smoke')
            subprocess.run(cmd,check=True)
        per_kg[kg]=json.loads(result.read_text(encoding='utf-8'))
        assert per_kg[kg]['full_data']==(not args.smoke)
    macro={m:float(np.mean([v['metrics'][m] for v in per_kg.values()])) for m in ['h1','h3','h10']}
    save(out/'result.json',{'status':'completed','purpose':'smoke' if args.smoke else 'formal','full_data':not args.smoke,
        'method':'KEnS','dataset':args.dataset,'seed':args.seed,'per_kg':per_kg,'macro':macro,
        'protocol':'kens-nomination-hits-v1-validation-only','wall_time_s':time.time()-started,
        'mrr_status':'undefined: the original ensemble changes its nominations with n; no full ranking'})


def run_target(args,out):
    import pandas as pd
    import tensorflow as tf
    os.chdir(out);random.seed(args.seed);np.random.seed(args.seed);tf.random.set_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    source=ROOT/'reproduction/sources/KEnS';sys.path.insert(0,str(source))
    module=importlib.import_module('run');param=importlib.import_module('src.param')
    source_data=ROOT/f'data/raw/dmkgc/dataset{args.dataset}'
    raw_hashes={str(f.relative_to(source_data)):digest(f) for sub in ['kg','entity','seed_alignlinks'] for f in sorted((source_data/sub).glob('*')) if f.is_file()}
    raw_hashes['relations.txt']=digest(source_data/'relations.txt')
    code=inspect.getsource(module.main)
    code=code.replace("src_langs = ['fr', 'ja', 'es', 'el', 'en']",'src_langs = '+repr(DOMAINS[args.dataset]))
    code=code.replace("data_dir = './data/kg'",'data_dir = '+repr(str(source_data/'kg')))
    code=code.replace("seed_dir = './data/seed_alignlinks'",'seed_dir = '+repr(str(source_data/'seed_alignlinks')))
    code=code.replace("model_dir = join('./trained_model', f'kens-{param.knowledge}-{param.dim}', target_lang)",'model_dir = '+repr(str(out/'model')))
    (out/'adapted_training_function.py').write_text(code,encoding='utf-8')
    exec(compile(code,str(out/'adapted_training_function.py'),'exec'),module.__dict__)
    # Official TSV files are headerless. Preserve the first alignment row.
    def load_links(directory):
        return {tuple(f.stem.split('-')):pd.read_csv(f,sep='\t',header=None).values.astype(np.int64)
                for f in sorted(Path(directory).glob('*.tsv'))}
    module.load_all_to_all_seed_align_links=load_links
    trainargs=module.parse_args(['--target_language','ja','--knowledge_model','transe','--seed',str(args.seed),'--use_default'])
    trainargs.target_language=args.target
    if args.smoke:trainargs.MAX_SAM=64
    started=time.time();module.main(trainargs);training_s=time.time()-started
    langs=[args.target]+[kg for kg in DOMAINS[args.dataset] if kg!=args.target]
    val=pd.read_csv(out/'model/results-val.tsv',sep='\t')
    test=pd.read_csv(out/'model/results-test.tsv',sep='\t')
    for frame in [val,test]:
        for kg in langs:frame[kg]=frame[kg].map(ast.literal_eval)
    train=np.loadtxt(source_data/'kg'/f'{args.target}-train.tsv',delimiter='\t',dtype=np.int64,ndmin=2)
    known={}
    for h,r,t in train:known.setdefault((int(h),int(r)),set()).add(int(t))
    nentity=sum(1 for _ in (source_data/'entity'/f'{args.target}.tsv').open(encoding='utf-8'))
    # The released test.py incorrectly estimates this prior on test predictions.
    # Estimate it on validation only; a one-observation floor handles zero hits.
    base={}
    for kg in langs:
        hits=0
        for row in val.itertuples(index=False):
            nominations=getattr(row,kg)
            blocked=known.get((int(row.h),int(row.r)),set())-{int(row.t)}
            ranked=[int(e) for e,s in nominations if int(e) not in blocked][:10]
            hits+=int(row.t in ranked)
        base[kg]=max(hits,1)/len(val)
    val_entities=val.copy()
    for kg in langs:val_entities[kg]=val_entities[kg].map(lambda pairs:[int(e) for e,s in pairs])
    learning=importlib.import_module('src.weightlearning')
    weights={}
    for entity in range(nentity):
        weights[entity]=learning.learn_entity_specific_weights(args.target,langs,val_entities,entity,nentity,base)
    weight_array=np.array([[weights[e][kg] for kg in langs] for e in range(nentity)],dtype=np.float64)
    assert np.isfinite(weight_array).all()
    np.savez_compressed(out/'ensemble_weights.npz',weights=weight_array,languages=np.array(langs),entity_id=np.arange(nentity),validation_base=np.array([base[k] for k in langs]))
    truth=test[['h','r','t']].values.astype(np.int64)
    candidate_ids=np.full((len(test),len(langs),10),-1,dtype=np.int32)
    candidate_scores=np.full(candidate_ids.shape,np.nan,dtype=np.float32)
    final={n:np.full((len(test),n),-1,dtype=np.int32) for n in [1,3,10]}
    hit={n:np.zeros(len(test),dtype=np.bool_) for n in [1,3,10]}
    for i,row in enumerate(test.itertuples(index=False)):
        blocked=known.get((int(row.h),int(row.r)),set())-{int(row.t)}
        choices=[]
        for j,kg in enumerate(langs):
            nominations=getattr(row,kg);choices.append(nominations)
            for k,(e,s) in enumerate(nominations[:10]):candidate_ids[i,j,k]=int(e);candidate_scores[i,j,k]=float(s)
        for n in [1,3,10]:
            tally={}
            for kg,nominations in zip(langs,choices):
                for e,s in nominations[:n]:tally[int(e)]=tally.get(int(e),0.0)+weights[int(row.h)][kg]
            # Stable insertion order ties are the released ensemble policy.
            ordered=[e for e,s in sorted(tally.items(),key=lambda item:item[1],reverse=True) if e not in blocked][:n]
            final[n][i,:len(ordered)]=ordered;hit[n][i]=int(row.t) in ordered
    np.savez_compressed(out/'test_queries.npz',triples=truth,query_index=np.arange(len(truth)),
        candidate_ids=candidate_ids,candidate_scores=candidate_scores,languages=np.array(langs),
        **{f'hit_{n}':v for n,v in hit.items()},**{f'final_top{n}':v for n,v in final.items()})
    checkpoints={str(f.relative_to(out)):digest(f) for f in sorted((out/'model').glob('*')) if f.suffix in ['.h5','.json']}
    assert checkpoints,'No trained KEnS artifacts found'
    save(out/'config.json',{'method':'KEnS','dataset':args.dataset,'kg':args.target,'seed':args.seed,'recipe':vars(trainargs),
        'raw_file_sha256':raw_hashes,'source_hashes':{str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),source/'run.py',*sorted((source/'src').glob('*.py'))]},
        'tensorflow':tf.__version__,'numpy':np.__version__,'device':'CPU','supporter_facts':'train + public validation as released',
        'selection':'fixed two rounds; entity weights fitted on public validation only',
        'repairs':['read headerless alignment TSV without dropping first row','ensemble prior uses validation instead of test','zero validation hit prior floored at 1/number_validation_queries','filter retains current gold'],
        'candidate_protocol':'for Hits@n each model nominates n entities; stable weighted vote; train filtering'})
    save(out/'result.json',{'status':'completed','full_data':not args.smoke,'method':'KEnS','dataset':args.dataset,'kg':args.target,'seed':args.seed,
        'metrics':{f'h{n}':float(v.mean()) for n,v in hit.items()},'n':len(truth),'checkpoints':checkpoints,
        'query_sha256':digest(out/'test_queries.npz'),'weight_sha256':digest(out/'ensemble_weights.npz'),
        'training_s':training_s,'wall_time_s':time.time()-started})


if __name__=='__main__':main()
