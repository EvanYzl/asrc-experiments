"""Freeze train-derived query strata for later baseline and paired plots."""
import csv

from common import *
from internal_data import alignments,source_kgs

RUN=ROOT/'reproduction/runs/strict_baselines_20260905'
OUT=RUN/'results/query_profiles'
DEGREE_EDGES=np.array([1,5,17,65,257],dtype=np.int64)
RELATION_EDGES=np.array([1,11,101,1001],dtype=np.int64)


def training_profile(data):
    triples=np.unique(data['arrays']['train'],axis=0)
    incoming=np.bincount(triples[:,2],minlength=data['entities']).astype(np.int32)
    outgoing=np.bincount(triples[:,0],minlength=data['entities']).astype(np.int32)
    relation=np.bincount(triples[:,1],minlength=data['relations']).astype(np.int32)
    assert incoming.sum()==outgoing.sum()==relation.sum()==len(triples)
    return {'in':incoming,'out':outgoing,'degree':incoming+outgoing,'relation':relation}


def freeze_npz(path,values):
    if path.exists():
        with np.load(path) as old:
            assert set(old.files)==set(values),path
            assert all(np.array_equal(old[k],v) for k,v in values.items()),f'Frozen query profiles changed: {path}'
    else:
        np.savez_compressed(path,**values)


def main():
    OUT.mkdir(parents=True,exist_ok=True);records=[];files=[]
    for ds,targets in DOMAINS.items():
        data={kg:load_kg(ds,kg) for kg in source_kgs(ds)}
        train={kg:training_profile(d) for kg,d in data.items()}
        for kg in targets:
            maps=alignments(ds,kg);sources=list(maps);d=data[kg];counts=train[kg]
            for split in ['val_select','test']:
                triples=d['arrays'][split];h,r,t=triples.T
                aligned=np.stack([maps[s][h] for s in sources],axis=1)
                source_degree=np.stack([np.where(aligned[:,i]>=0,train[s]['degree'][np.maximum(aligned[:,i],0)],0) for i,s in enumerate(sources)],axis=1)
                available=(aligned>=0)&(source_degree>0)
                values={'query_index':np.arange(len(triples),dtype=np.int64),
                    'public_row_index':d['valid_indices'] if split=='val_select' else np.arange(len(triples),dtype=np.int64),
                    'triples':triples.astype(np.int32),'sources':np.array(sources),
                    'head_train_in_degree':counts['in'][h],'head_train_out_degree':counts['out'][h],
                    'tail_train_in_degree':counts['in'][t],'tail_train_out_degree':counts['out'][t],
                    'head_train_degree':counts['degree'][h],'tail_train_degree':counts['degree'][t],
                    'relation_train_frequency':counts['relation'][r],
                    'head_degree_bin':np.searchsorted(DEGREE_EDGES,counts['degree'][h],side='right').astype(np.uint8),
                    'tail_degree_bin':np.searchsorted(DEGREE_EDGES,counts['degree'][t],side='right').astype(np.uint8),
                    'relation_frequency_bin':np.searchsorted(RELATION_EDGES,counts['relation'][r],side='right').astype(np.uint8),
                    'aligned_source_entities':aligned,'source_alignment_exists':aligned>=0,'source_available':available,
                    'source_potential_edge_count':np.minimum(source_degree,32).astype(np.uint8)}
                if split=='test':
                    with np.load(RUN/'results/overlap_masks'/f'{ds}_{kg}.npz') as masks:
                        assert np.array_equal(masks['query_index'],values['query_index'])
                        values.update({name:masks[name] for name in ['test_seen_train','test_seen_train_or_valid']})
                path=OUT/f'{ds}_{kg}_{split}.npz';freeze_npz(path,values)
                files.append({'dataset':ds,'kg':kg,'split':split,'queries':len(triples),'file':path.name,'sha256':sha256(path),
                    'dataset_hash':d['manifest']['dataset_hash'],
                    'source_dataset_hashes':{s:data[s]['manifest']['dataset_hash'] for s in sources},
                    'alignment_manifest_sha256':sha256(SUITE/'internal_data'/ds/'alignment'/f'{kg}_full.json')})
                for field in ['head_degree_bin','tail_degree_bin','relation_frequency_bin']:
                    for index,n in zip(*np.unique(values[field],return_counts=True)):
                        records.append({'dataset':ds,'kg':kg,'split':split,'axis':field,'bin':int(index),'queries':int(n)})
    payload={'schema':'query-profiles-v1','definition':'Degrees and relation frequencies use unique target training triples only; self-loops contribute once to each direction.',
        'degree_bin_labels':['0','1-4','5-16','17-64','65-256','257+'],
        'relation_frequency_bin_labels':['0','1-10','11-100','101-1000','1001+'],
        'source_definition':'Frozen alignment mapping; source_available requires an aligned entity with a nonempty source training neighborhood. Potential edges are min(in+out degree,32), before any gate.',
        'query_index':'Zero-based row within saved split; public_row_index links val_select back to the public validation file.',
        'usage':'Join exact triples/query_index to saved ranks for F5 strata and cases; these data do not set QURA or baseline budgets.',
        'files':files,'source_hashes':source_hashes([Path(__file__),SUITE/'common.py',SUITE/'internal_data.py'])}
    atomic_json(OUT/'manifest.json',payload)
    with (OUT/'stratum_counts.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps({'profile_files':len(files),'test_queries':sum(r['queries'] for r in files if r['split']=='test'),
        'validation_selection_queries':sum(r['queries'] for r in files if r['split']=='val_select'),
        'datasets':list(DOMAINS),'proposed_method_trained':False}))


if __name__=='__main__':main()
