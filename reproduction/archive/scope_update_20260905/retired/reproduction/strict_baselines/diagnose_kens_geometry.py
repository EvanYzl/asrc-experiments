"""Describe frozen KEnS training geometry; never change a model or tune on test."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'


def diagnose(directory):
    directory=Path(directory)
    result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    assert result['training_revision']=='kens-transe-paper-loss-v2'
    kg,ds=result['kg'],result['dataset']
    matrices=[]
    with h5py.File(directory/'model'/f'{kg}.h5') as handle:
        handle['model_weights'].visititems(lambda name,obj:matrices.append((name,obj[...])) if isinstance(obj,h5py.Dataset) else None)
    entity=next(x for name,x in matrices if name.startswith('kens_entity_'))
    relation=next(x for name,x in matrices if name.startswith('kens_relation_'))
    train=np.loadtxt(ROOT/f'data/raw/dmkgc/dataset{ds}/kg/{kg}-train.tsv',delimiter='\t',dtype=np.int64,ndmin=2)
    known={}
    for h,r,t in train:known.setdefault((int(h),int(r)),set()).add(int(t))
    rng=np.random.default_rng(20260905)
    indices=np.sort(rng.choice(len(train),size=min(1024,len(train)),replace=False))
    triples=train[indices]
    negative=np.array([rng.choice([e for e in range(len(entity)) if e not in known[int(h),int(r)]]) for h,r,t in triples])
    q=entity[triples[:,0]]+relation[triples[:,1]]
    positive_distance=np.linalg.norm(q-entity[triples[:,2]],axis=1)
    negative_distance=np.linalg.norm(q-entity[negative],axis=1)
    head_distance=np.linalg.norm(relation[triples[:,1]],axis=1)
    valid_head_negative=np.array([int(h) not in known[int(h),int(r)] for h,r,t in triples])
    margin=config['recipe']['transe_margin']
    random_loss=np.maximum(positive_distance-negative_distance+margin,0)
    head_loss=np.maximum(positive_distance-head_distance+margin,0)
    active_relations=np.unique(train[:,1])
    payload={'dataset':ds,'kg':kg,'seed':result['seed'],'run_id':directory.parent.name,
        'source_split':'train only; fixed 1024-row sample, no test-based tuning',
        'training_revision':result['training_revision'],'sample_queries':len(triples),'margin':margin,
        'known_positive_random_negatives':sum(int(n) in known[int(h),int(r)] for (h,r,t),n in zip(triples,negative)),
        'random_negative_active_hinge_rate':float((random_loss>0).mean()),
        'random_negative_mean_hinge':float(random_loss.mean()),
        'valid_head_negative_queries':int(valid_head_negative.sum()),
        'head_negative_active_hinge_rate':float((head_loss[valid_head_negative]>0).mean()),
        'head_negative_mean_hinge':float(head_loss[valid_head_negative].mean()),
        'head_distance_less_than_gold_rate':float((head_distance[valid_head_negative]<positive_distance[valid_head_negative]).mean()),
        'used_relation_norm_q05_q50_q95':np.quantile(np.linalg.norm(relation[active_relations],axis=1),[.05,.5,.95]).tolist(),
        'entity_norm_q05_q50_q95':np.quantile(np.linalg.norm(entity,axis=1),[.05,.5,.95]).tolist(),
        'checkpoint_sha256':{k.replace('\\','/'):v for k,v in result['checkpoints'].items()}[f'model/{kg}.h5'],
        'interpretation':'Descriptive loss/ranking diagnostic. Zero sampled hinge does not imply correct top-1 ranking; no candidate is removed and no training hyperparameter is selected here.'}
    folder=RUN/'results/kens_geometry';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{directory.parent.name}_{kg}.npz',train_row_index=indices,triples=triples,
                        random_negative=negative,positive_distance=positive_distance,random_negative_distance=negative_distance,
                        head_distance=head_distance,valid_head_negative=valid_head_negative,random_hinge=random_loss,head_hinge=head_loss)
    (folder/f'{directory.parent.name}_{kg}.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);args=parser.parse_args()
    print(json.dumps(diagnose(args.directory)))
