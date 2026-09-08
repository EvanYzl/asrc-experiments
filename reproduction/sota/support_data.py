"""Pre-registered derived inputs; original train/validation/filter arrays stay immutable."""
import hashlib
import numpy as np
from frozen_data import ROOT, DOMAINS, load_kg, sha256

def array_hash(a):
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()

def order_rows(rows, seed, label):
    return np.asarray(sorted(range(len(rows)), key=lambda i: hashlib.sha256(
        (str(seed)+'|'+label+'|'+','.join(map(str,rows[i]))).encode()).digest()),dtype=np.int64)

def build_inputs(recipe):
    ds=recipe['dataset'];mode=recipe['mode'];seed=recipe['perturbation_seed']
    kgs=['en','fr'] if ds=='wk3l' else DOMAINS[ds]
    data={kg:load_kg(ds,kg) for kg in kgs};training={};derived={};train_audit={}
    for kg,d in data.items():
        original=d['arrays']['train'];unique,inverse=np.unique(original,axis=0,return_inverse=True)
        keep=recipe['train_keep'];selected=np.arange(len(unique))
        if keep<1: selected=order_rows(unique,seed,'train|'+ds+'|'+kg)[:int(len(unique)*keep)]
        mask=np.zeros(len(unique),dtype=bool);mask[selected]=True
        indices=np.flatnonzero(mask[inverse]);training[kg]=original[indices].copy()
        assert np.array_equal(original[indices],training[kg]) and len(indices)>0
        derived['train_'+kg+'_indices']=indices
        train_audit[kg]={'original_rows':len(original),'kept_rows':len(indices),'original_unique':len(unique),
            'kept_unique':len(selected),'kept_fraction':len(indices)/len(original),'original_sha256':array_hash(original),
            'derived_sha256':array_hash(training[kg]),'indices_sha256':array_hash(indices)}
    offsets={};n=0
    for kg,d in data.items(): offsets[kg]=n;n+=d['entities']
    parents=np.arange(n);masks=np.zeros(n,dtype=np.int64)
    for i,kg in enumerate(kgs):masks[offsets[kg]:offsets[kg]+data[kg]['entities']]=1<<i
    def find(x):
        while parents[x]!=x:parents[x]=parents[parents[x]];x=int(parents[x])
        return x
    links=[]
    if ds=='wk3l':links=[('en','fr',ROOT/'data/raw/atransn/SHARED/wk3l-15k_en_f_fr_aligned_entity_id.txt')]
    else:
        for path in sorted((ROOT/f'data/raw/dmkgc/dataset{ds}/seed_alignlinks').glob('*.tsv')):
            names=path.stem.split('-')
            if len(names)==2 and all(x in kgs for x in names):links.append((*names,path))
    audit={'mode':mode,'input_entities':n,'alignment_files':{},'accepted_links':0,'redundant_links':0,'conflicting_links':0,
           'training':train_audit,'perturbation_seed':seed,'filter_policy':'Original full-public-train validation/test filters; reduction applies only to optimization facts.'}
    for ix,(a,b,path) in enumerate(links):
        original=np.unique(np.loadtxt(path,dtype=np.int64,ndmin=2),axis=0)
        assert original.shape[1]==2 and original.min()>=0
        assert original[:,0].max()<data[a]['entities'] and original[:,1].max()<data[b]['entities']
        label=path.relative_to(ROOT).as_posix();indices=np.arange(len(original))
        if recipe['alignment_keep']<1:
            indices=np.sort(order_rows(original,seed,'align|'+label)[:int(len(original)*recipe['alignment_keep'])])
        pairs=original[indices].copy();changed=[]
        if recipe['alignment_noise']>0:
            changed=order_rows(pairs,seed,'noise|'+label)[:int(len(pairs)*recipe['alignment_noise'])]
            for i in changed:
                x,y=map(int,pairs[i]);digest=hashlib.sha256(f'{seed}|replacement|{label}|{x}|{y}'.encode()).digest()
                replacement=int.from_bytes(digest[:8],'big')%(data[b]['entities']-1)
                pairs[i,1]=replacement+(replacement>=y)
            assert np.all(pairs[changed,1]!=original[indices[changed],1])
        derived[f'align_{ix}_original_indices']=indices
        derived[f'align_{ix}_changed_indices']=np.asarray(changed,dtype=np.int64)
        pairs=np.unique(pairs,axis=0);derived[f'align_{ix}_pairs']=pairs
        audit['alignment_files'][label]={'sha256':sha256(path),'original_unique_pairs':len(original),'selected_pairs':len(indices),
            'derived_unique_pairs':len(pairs),'replaced_second_endpoints':len(changed),'derived_sha256':array_hash(pairs)}
        if mode not in ['shared','entity_only']:continue
        for x,y in pairs:
            u,v=find(int(x)+offsets[a]),find(int(y)+offsets[b])
            if u==v:audit['redundant_links']+=1;continue
            if masks[u]&masks[v]:audit['conflicting_links']+=1;continue
            if u>v:u,v=v,u
            parents[v]=u;masks[u]|=masks[v];audit['accepted_links']+=1
    roots=np.asarray([find(i) for i in range(n)]);unique,inverse=np.unique(roots,return_inverse=True)
    maps={kg:inverse[offsets[kg]:offsets[kg]+d['entities']] for kg,d in data.items()}
    for kg,m in maps.items():assert len(np.unique(m))==len(m)
    roff={};nr=0
    separate=ds=='wk3l' or mode in ['independent','entity_only']
    for kg,d in data.items():roff[kg]=nr if separate else 0;nr=nr+d['relations'] if separate else d['relations']
    audit.update(shared_entities=len(unique),relations=nr,entity_map_sha256={kg:array_hash(m) for kg,m in maps.items()},
        rule='Stable supplied alignment union, rejecting within-KG collisions; fixed derived perturbations; no held-out alignment.')
    return data,maps,roff,len(unique),nr,audit,training,derived
