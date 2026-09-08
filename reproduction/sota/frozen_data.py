"""Manifest-exact train/validation loader. Test triples are never opened here."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = {'dbp5l':['el','en','es','fr','ja'], 'depkg':['de','es','fr','it','jp','uk'], 'dwy':['db','wk','yg'], 'wk3l':['fr']}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def load_kg(dataset,kg):
    manifest_path=ROOT/'reproduction/strict_baselines/data_manifests'/f'{dataset}_{kg}.json'
    m=json.loads(manifest_path.read_text(encoding='utf-8'))
    paths={}
    for name,digest in m['files'].items():
        p=ROOT/name.replace('\\','/')
        assert sha256(p)==digest,f'Input hash changed: {p}'
        if p.name in {f'{kg}-train.tsv','train_triple_id.txt'}:paths['train']=p
        if p.name in {f'{kg}-val.tsv','valid_triple_id.txt'}:paths['valid']=p
    arrays={k:np.loadtxt(v,dtype=np.int64,ndmin=2) for k,v in paths.items()}
    for k,a in arrays.items():
        assert len(a)==m['counts'][k] and a.shape[1]==3
        assert a.min()>=0 and a[:,[0,2]].max()<m['entities'] and a[:,1].max()<m['relations_dictionary']
    select=np.asarray(m['val_select_indices'],dtype=np.int64)
    cert=np.asarray(m['val_cert_indices'],dtype=np.int64)
    assert not set(select)&set(cert)
    assert sorted(np.r_[select,cert].tolist())==list(range(len(arrays['valid'])))
    arrays['val_select']=arrays['valid'][select]
    # valid is retained only for final protocol metadata, never consumed by training.
    return {'dataset':dataset,'kg':kg,'entities':m['entities'],'relations':m['relations_dictionary'],
            'arrays':arrays,'manifest':m,'manifest_sha256':sha256(manifest_path)}

def load_multikg(dataset,sharing):
    kgs=['en','fr'] if dataset=='wk3l' else DOMAINS[dataset]
    data={kg:load_kg(dataset,kg) for kg in kgs}
    offsets={};n=0
    for kg,d in data.items():offsets[kg]=n;n+=d['entities']
    parents=np.arange(n);masks=np.zeros(n,dtype=np.int64)
    for i,kg in enumerate(kgs):masks[offsets[kg]:offsets[kg]+data[kg]['entities']]=1<<i
    def find(a):
        while parents[a]!=a:parents[a]=parents[parents[a]];a=int(parents[a])
        return a
    audit={'sharing':sharing,'input_entities':n,'alignment_files':{},'accepted_links':0,'redundant_links':0,'conflicting_links':0}
    links=[]
    if dataset=='wk3l':
        path=ROOT/'data/raw/atransn/SHARED/wk3l-15k_en_f_fr_aligned_entity_id.txt'
        links=[('en','fr',path)]
    else:
        for path in sorted((ROOT/f'data/raw/dmkgc/dataset{dataset}/seed_alignlinks').glob('*.tsv')):
            names=path.stem.split('-')
            if len(names)==2 and all(x in kgs for x in names):links.append((*names,path))
    for a,b,path in links:
        pairs=np.unique(np.loadtxt(path,dtype=np.int64,ndmin=2),axis=0)
        assert pairs.shape[1]==2 and pairs.min()>=0
        assert pairs[:,0].max()<data[a]['entities'] and pairs[:,1].max()<data[b]['entities']
        audit['alignment_files'][str(path.relative_to(ROOT)).replace('\\','/')]={'sha256':sha256(path),'pairs':len(pairs)}
        if sharing=='independent':continue
        for x,y in pairs:
            u,v=find(int(x)+offsets[a]),find(int(y)+offsets[b])
            if u==v:audit['redundant_links']+=1;continue
            if masks[u]&masks[v]:audit['conflicting_links']+=1;continue
            if u>v:u,v=v,u
            parents[v]=u;masks[u]|=masks[v];audit['accepted_links']+=1
    roots=np.asarray([find(i) for i in range(n)])
    unique,inverse=np.unique(roots,return_inverse=True)
    maps={kg:inverse[offsets[kg]:offsets[kg]+d['entities']] for kg,d in data.items()}
    for kg,a in maps.items():assert len(np.unique(a))==len(a),'Two candidates in one KG share an embedding'
    relation_offsets={};nr=0
    for kg,d in data.items():
        # Core releases already use a global relation dictionary. WK3l dictionaries differ.
        relation_offsets[kg]=nr if dataset=='wk3l' or sharing=='independent' else 0
        nr=nr+d['relations'] if dataset=='wk3l' or sharing=='independent' else d['relations']
    audit.update(shared_entities=len(unique),relations=nr,entity_map_sha256={kg:hashlib.sha256(a.tobytes()).hexdigest() for kg,a in maps.items()},
                 rule='Stable supplied alignment union; reject merges that identify distinct entities within a KG; no learned or held-out alignment.')
    return data,maps,relation_offsets,len(unique),nr,audit
