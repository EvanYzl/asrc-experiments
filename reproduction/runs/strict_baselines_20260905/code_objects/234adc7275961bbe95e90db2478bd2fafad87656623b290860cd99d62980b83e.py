"""Frozen CSR, alignment and negative sampling data for internal controls."""
import copy
import hashlib
from pathlib import Path

from common import *

INTERNAL_VERSION='internal-transe128-csr32-v1'
STORE=SUITE/'internal_data'
HASH_SEED=20260905


def hash64(values,seed=HASH_SEED):
    x=np.asarray(values,dtype=np.uint64)+np.uint64(seed)+np.uint64(0x9E3779B97F4A7C15)
    x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
    x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
    return x^(x>>np.uint64(31))


def source_kgs(dataset):return ['en','fr'] if dataset=='wk3l' else DOMAINS[dataset]


def load_setting(dataset,kg,condition='full'):
    data=load_kg(dataset,kg)
    data['arrays']=dict(data['arrays']);data['manifest']=dict(data['manifest'])
    if condition=='target20':
        train=data['arrays']['train'];order=np.argsort(hash64(np.arange(len(train))),kind='stable')
        # Group duplicate facts so retention cannot split identical facts.
        unique,first,inverse=np.unique(train,axis=0,return_index=True,return_inverse=True)
        group_order=np.argsort(hash64(first),kind='stable')
        chosen=np.zeros(len(unique),dtype=bool);chosen[group_order[:max(1,int(len(unique)*.2))]]=True
        ids=np.flatnonzero(chosen[inverse]);data['arrays']['train']=train[ids]
        d=STORE/dataset/'retention';d.mkdir(parents=True,exist_ok=True)
        np.save(d/f'{kg}_target20_indices.npy',ids)
        data['manifest'].update(condition=condition,retained_train_indices_sha256=sha256(d/f'{kg}_target20_indices.npy'),
            original_training_rows=len(train),retained_training_rows=len(ids),retention_rule='20% of unique train triples, grouped duplicates, SplitMix64 seed 20260905')
    data['condition']=condition
    return data


class CSRStore:
    def __init__(self,dataset,kg):
        self.dataset=dataset;self.kg=kg;self.path=STORE/dataset/kg
        data=load_kg(dataset,kg);self.n=data['entities'];self.r=data['relations']
        path=self.path;path.mkdir(parents=True,exist_ok=True)
        manifest_path=path/'manifest.json'
        if not manifest_path.exists():
            triples=np.unique(data['arrays']['train'],axis=0)
            # Fixed lexicographic order within each direction: relation, neighbor.
            for name,nodecol,othercol in [('in',2,0),('out',0,2)]:
                order=np.lexsort((triples[:,othercol],triples[:,1],triples[:,nodecol]));a=triples[order]
                counts=np.bincount(a[:,nodecol],minlength=self.n)
                np.save(path/f'{name}_ptr.npy',np.r_[0,np.cumsum(counts)].astype(np.int64))
                np.save(path/f'{name}_rel.npy',a[:,1].astype(np.int32));np.save(path/f'{name}_nbr.npy',a[:,othercol].astype(np.int32))
            sketch=np.zeros((self.n,512),dtype=np.uint32)
            for nodecol,offset in [(2,0),(0,256)]:
                buckets=(hash64(triples[:,1])%np.uint64(256)).astype(np.int64)+offset
                np.add.at(sketch,(triples[:,nodecol],buckets),1)
            saturated=np.argwhere(sketch>65535).astype(np.int32)
            np.save(path/'saturated_buckets.npy',saturated)
            np.save(path/'sketch.npy',np.minimum(sketch,65535).astype(np.uint16))
            atomic_json(manifest_path,{'version':INTERNAL_VERSION,'dataset':dataset,'kg':kg,
                'source_dataset_hash':data['manifest']['dataset_hash'],'nentities':self.n,'nrelations':self.r,
                'train_rows':len(data['arrays']['train']),'unique_train_rows':len(triples),
                'duplicates_removed':len(data['arrays']['train'])-len(triples),'hash':'SplitMix64','hash_seed':HASH_SEED,
                'sketch_layout':'incoming 256 then outgoing 256; uint16 clipped 65535',
                'edge_order':'incoming before outgoing, each direction lexicographic relation ID then neighbor ID',
                'k':32,'files':{f.name:{'sha256':sha256(f),'shape':list(np.load(f,mmap_mode='r').shape),'dtype':str(np.load(f,mmap_mode='r').dtype)} for f in sorted(path.glob('*.npy'))}})
        meta=json.loads(manifest_path.read_text(encoding='utf-8'))
        assert meta['source_dataset_hash']==data['manifest']['dataset_hash']
        self.ptr=[np.load(path/f'{k}_ptr.npy',mmap_mode='r') for k in ['in','out']]
        self.rel=[np.load(path/f'{k}_rel.npy',mmap_mode='r') for k in ['in','out']]
        self.nbr=[np.load(path/f'{k}_nbr.npy',mmap_mode='r') for k in ['in','out']]
        self.sketch=np.load(path/'sketch.npy',mmap_mode='r')
        self.degree=np.diff(self.ptr[0])+np.diff(self.ptr[1])

    def fetch(self,entities,k=32):
        entities=np.asarray(entities,dtype=np.int64)
        n=len(entities);nbr=np.zeros((n,k),dtype=np.int64);rel=np.zeros_like(nbr)
        direction=np.zeros_like(nbr);offset=np.full_like(nbr,-1);valid=np.zeros((n,k),dtype=bool)
        for i,e in enumerate(entities):
            if e<0:continue
            assert e<self.n;used=0
            for d in range(2):
                start=int(self.ptr[d][e]);stop=min(int(self.ptr[d][e+1]),start+k-used);count=stop-start
                if count:
                    sl=slice(used,used+count);nbr[i,sl]=self.nbr[d][start:stop];rel[i,sl]=self.rel[d][start:stop]
                    direction[i,sl]=d;offset[i,sl]=np.arange(start,stop);valid[i,sl]=True;used+=count
                if used==k:break
        return {'neighbor':nbr,'relation':rel,'direction':direction,'offset':offset,'valid':valid}


def alignments(dataset,target,condition='full'):
    domains=source_kgs(dataset);sources=[kg for kg in domains if kg!=target]
    target_n=load_kg(dataset,target)['entities'];maps={};audit={}
    for s in sources:
        source_n=load_kg(dataset,s)['entities']
        if dataset=='wk3l':
            path=ROOT/'data/raw/atransn/SHARED/wk3l-15k_en_f_fr_aligned_entity_id.txt'
            pairs=np.loadtxt(path,dtype=np.int64,ndmin=2)
            if target=='fr':pairs=pairs[:,::-1]
        else:
            parent=ROOT/f'data/raw/dmkgc/dataset{dataset}/seed_alignlinks'
            path=parent/f'{target}-{s}.tsv'
            if path.exists():pairs=np.loadtxt(path,dtype=np.int64,delimiter='\t',ndmin=2)
            else:path=parent/f'{s}-{target}.tsv';pairs=np.loadtxt(path,dtype=np.int64,delimiter='\t',ndmin=2)[:,::-1]
        assert pairs.min()>=0 and pairs[:,0].max()<target_n and pairs[:,1].max()<source_n
        pairs=np.unique(pairs,axis=0);ordered=pairs[np.lexsort((pairs[:,1],pairs[:,0]))]
        # One source entity per target/source key. Keep smallest stable source ID.
        mapping=np.full(target_n,-1,dtype=np.int32)
        _,first=np.unique(ordered[:,0],return_index=True);chosen=ordered[first]
        raw_chosen=chosen.copy()
        if condition=='align20':
            order=np.argsort(hash64(chosen[:,0].astype(np.uint64)*np.uint64(source_n)+chosen[:,1]),kind='stable')
            chosen=chosen[order[:max(1,int(len(chosen)*.2))]]
        mapping[chosen[:,0]]=chosen[:,1]
        if condition.startswith('corrupt'):
            fraction=int(condition.removeprefix('corrupt'))/100
            order=np.argsort(hash64(chosen[:,0]),kind='stable');bad=chosen[order[:int(len(chosen)*fraction)],0]
            shift=(hash64(bad,seed=HASH_SEED+123)%(source_n-1)+1).astype(np.int64)
            mapping[bad]=(mapping[bad]+shift)%source_n
            assert np.all(mapping[bad]!=raw_chosen[np.searchsorted(raw_chosen[:,0],bad),1])
        maps[s]=mapping
        audit[s]={'source_path':str(path),'source_sha256':sha256(path),'distinct_pairs':len(pairs),
            'multiple_source_ids_discarded':len(pairs)-len(first),'mapped_target_entities':int((mapping>=0).sum()),
            'mapping_sha256':hashlib.sha256(mapping.tobytes()).hexdigest()}
    d=STORE/dataset/'alignment';d.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(d/f'{target}_{condition}.npz',**maps)
    atomic_json(d/f'{target}_{condition}.json',{'target':target,'condition':condition,'rule':'deduplicate exact pairs; for multiple sources in one KG keep smallest source entity ID','sources':audit})
    return maps


class NegativeSampler:
    def __init__(self,data,seed):
        self.n=data['entities'];self.known=tail_map([data['arrays']['train']]);self.rng=np.random.default_rng(seed)

    def draw(self,triples,count=256):
        result=np.zeros((len(triples),count),dtype=np.int64);valid=np.ones(len(triples),dtype=bool)
        for i,(h,r,t) in enumerate(triples):
            excluded=self.known.get((int(h),int(r)),set())|{int(t)};available=self.n-len(excluded)
            if available<=0:valid[i]=False;continue
            if available<count:
                pool=np.setdiff1d(np.arange(self.n),np.fromiter(excluded,dtype=np.int64),assume_unique=True)
                result[i]=self.rng.choice(pool,count,replace=True)
            else:
                candidate=self.rng.choice(self.n,min(self.n,count+len(excluded)),replace=False)
                candidate=candidate[~np.isin(candidate,np.fromiter(excluded,dtype=np.int64))]
                result[i]=candidate[:count]
        return result,valid
