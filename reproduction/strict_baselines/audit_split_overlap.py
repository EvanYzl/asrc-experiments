"""Record duplicate and exact split-overlap counts without rewriting public data."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'
DOMAINS={'dbp5l':['el','en','es','fr','ja'],'depkg':['de','es','fr','it','jp','uk'],'dwy':['db','wk','yg'],'wk3l':['en','fr']}


def main():
    rows=[]
    for ds,kgs in DOMAINS.items():
        for kg in kgs:
            data={};counts={};raw={}
            for split in ['train','val','test']:
                if ds=='wk3l':
                    base=ROOT/'data/raw/atransn'/('WK3l-15k_EN_F' if kg=='en' else 'WK3l-15k_FR')
                    path=base/(('valid' if split=='val' else split)+'_triple_id.txt')
                else:path=ROOT/f'data/raw/dmkgc/dataset{ds}/kg/{kg}-{split}.tsv'
                a=np.loadtxt(path,dtype=np.int64,delimiter='\t',ndmin=2);unique=np.unique(a,axis=0)
                raw[split]=a
                data[split]=np.ascontiguousarray(unique).view(np.dtype((np.void,24))).ravel()
                counts[split+'_rows']=len(a);counts[split+'_duplicate_rows']=len(a)-len(unique)
            overlap={f'{a}_{b}_exact_overlap':len(np.intersect1d(data[a],data[b],assume_unique=True)) for a,b in [('train','val'),('train','test'),('val','test')]}
            queries=np.ascontiguousarray(raw['test']).view(np.dtype((np.void,24))).ravel()
            masks=RUN/'results/overlap_masks';masks.mkdir(exist_ok=True)
            np.savez_compressed(masks/f'{ds}_{kg}.npz',query_index=np.arange(len(queries)),
                test_seen_train=np.isin(queries,data['train']),test_seen_train_or_valid=np.isin(queries,np.union1d(data['train'],data['val'])))
            rows.append({'dataset':ds,'kg':kg,**counts,**overlap})
    payload={'public_splits_unchanged':True,'identity':'exact h/r/t within one KG, no inverse or cross-KG equivalence expansion','rows':rows,
        'total_train_test_exact_overlap':sum(r['train_test_exact_overlap'] for r in rows)}
    (RUN/'results/split_overlap_audit.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k!='rows'}))


if __name__=='__main__':main()
