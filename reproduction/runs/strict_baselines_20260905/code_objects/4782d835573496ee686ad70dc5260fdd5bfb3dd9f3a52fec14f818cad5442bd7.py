"""Golden checks for ranking, gold preservation, held-out folds and negatives."""
import json
import numpy as np
import torch
from common import *
from run_kge import negative_batch

def main():
    scores=torch.tensor([[1.,2.,2.,0.],[4.,3.,2.,1.],[1.,1.,1.,1.]])
    triples=np.array([[0,0,2],[1,0,1],[2,0,3]])
    actual=filtered_ranks(scores,triples,{(0,0):{1,2},(1,0):{0,1}})
    assert actual.tolist()==[1,1,4],actual
    assert filtered_ranks(scores,triples,{}).tolist()==[2,2,4]
    valid=np.array([[i//2,i%3,i//2] for i in range(30)]+[[1,0,2],[1,0,2]])
    a,b=validation_partition(valid)
    assert len(a)+len(b)==len(valid) and set(a).isdisjoint(b)
    assert set(map(tuple,valid[a])).isdisjoint(map(tuple,valid[b]))
    assert np.array_equal(a,validation_partition(valid)[0])
    seed_all(17)
    train=torch.tensor([[0,0,1],[0,0,2],[1,1,0]],device='cuda')
    keys=torch.unique((train[:,0]*2+train[:,1])*5+train[:,2],sorted=True)
    for mode in ['head-batch','tail-batch']:
        neg=negative_batch(train,5,2,keys,train,mode,256)
        for row,(h,r,t) in enumerate(train.cpu().tolist()):
            for e in neg[row].cpu().tolist():
                assert ((e,r,t) if mode=='head-batch' else (h,r,e)) not in set(map(tuple,train.cpu().tolist()))
    atomic_json(SUITE/'protocol_test_report.json',{'passed':True,'checks':['stable-ID ties','gold preserved under filtering','unfiltered ranks','grouped independent validation split','train-positive exclusion for both corruption modes']})
    print('Protocol golden checks passed.')

if __name__=='__main__':main()
