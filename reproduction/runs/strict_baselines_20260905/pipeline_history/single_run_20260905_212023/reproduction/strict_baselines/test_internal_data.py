"""Independent invariants for CSR edges, hashing and filtered negatives."""
import numpy as np
from common import atomic_json,SUITE,load_kg
from internal_data import CSRStore,NegativeSampler,hash64,alignments


def reference_hash(v,seed=20260905):
    mask=(1<<64)-1;x=(int(v)+seed+0x9E3779B97F4A7C15)&mask
    x=((x^(x>>30))*0xBF58476D1CE4E5B9)&mask
    x=((x^(x>>27))*0x94D049BB133111EB)&mask
    return x^(x>>31)


def main():
    values=np.array([0,1,256,991,(1<<63)-1],dtype=np.uint64)
    assert hash64(values).tolist()==[reference_hash(x) for x in values]
    fake={'entities':300,'arrays':{'train':np.array([[1,0,2],[1,0,3],[1,0,4]])}}
    sampler=NegativeSampler(fake,17);neg,valid=sampler.draw(np.tile([1,0,2],(20,1)),256)
    assert valid.all() and not np.isin(neg,[2,3,4]).any()
    assert all(len(set(row))==256 for row in neg)
    fake['entities']=5;sampler=NegativeSampler(fake,17);neg,valid=sampler.draw(np.array([[1,0,2]]),256)
    assert valid[0] and set(neg[0])=={0,1}
    full={'entities':2,'arrays':{'train':np.array([[0,0,0],[0,0,1]])}}
    assert not NegativeSampler(full,17).draw(np.array([[0,0,0]]))[1][0]
    data=load_kg('dbp5l','el');store=CSRStore('dbp5l','el');unique=np.unique(data['arrays']['train'],axis=0)
    queries=np.arange(100);observed=store.fetch(queries)
    for i,e in enumerate(queries):
        incoming=sorted((int(r),int(h)) for h,r,t in unique if t==e)
        outgoing=sorted((int(r),int(t)) for h,r,t in unique if h==e)
        ref=([(0,r,n) for r,n in incoming]+[(1,r,n) for r,n in outgoing])[:32]
        assert ref==[(int(d),int(r),int(n)) for d,r,n,v in zip(observed['direction'][i],observed['relation'][i],observed['neighbor'][i],observed['valid'][i]) if v]
        sketch=np.zeros(512,dtype=int)
        for r,n in incoming:sketch[reference_hash(r)%256]+=1
        for r,n in outgoing:sketch[256+reference_hash(r)%256]+=1
        assert np.array_equal(np.minimum(sketch,65535),store.sketch[e])
    mapping=alignments('wk3l','fr')['en'];assert (mapping>=0).sum()==2458
    atomic_json(SUITE/'internal_protocol_test_report.json',{'status':'passed','checks':['independent scalar SplitMix64 reference','256 unique negatives excluding all positives','small pool replacement','empty pool skipped','100 entity CSR/reference equality','512 bucket sketch/reference equality','WK alignment ambiguity deterministic']})
    print('Internal data invariants passed')


if __name__=='__main__':main()
