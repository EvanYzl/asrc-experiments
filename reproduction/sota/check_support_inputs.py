"""Meaningful input equivalence, isolation and deterministic perturbation checks."""
import json
import numpy as np
from frozen_data import ROOT,load_multikg,atomic_json,sha256
from support_data import build_inputs,array_hash
plan=json.loads((ROOT/'reproduction/sota/paper_support/PLAN.json').read_text());checks=[]
for ds in ['dbp5l','depkg','dwy','wk3l']:
    for mode in ['independent','shared']:
        recipe=plan['models'][f'{ds}_{mode}_s17']['recipe'];old=load_multikg(ds,mode);new=build_inputs(recipe)
        assert old[2:5]==new[2:5]
        for kg,m in old[1].items():
            assert np.array_equal(m,new[1][kg])
            assert np.array_equal(old[0][kg]['arrays']['train'],new[6][kg])
        checks.append({'check':'exact_original_clean_mapping_and_training','dataset':ds,'mode':mode,'passed':True})
base=build_inputs(plan['models']['dbp5l_shared_s17']['recipe'])
for variant in ['entity_only','relation_only','no_reciprocal','no_n3','train50_independent','train50_shared','align50_shared','noise10_shared']:
    recipe=plan['models'][f'dbp5l_{variant}_s17']['recipe'];a=build_inputs(recipe);b=build_inputs(recipe)
    assert a[2:6]==b[2:6]
    for k,v in a[7].items():assert np.array_equal(v,b[7][k])
    for kg,d in a[0].items():
        assert np.array_equal(d['arrays']['train'],base[0][kg]['arrays']['train'])
        assert np.array_equal(d['arrays']['val_select'],base[0][kg]['arrays']['val_select'])
        if recipe['train_keep']<1:
            original_unique=np.unique(d['arrays']['train'],axis=0)
            assert len(np.unique(a[6][kg],axis=0))==len(original_unique)//2
        if variant=='entity_only':assert np.array_equal(a[1][kg],base[1][kg])
    if variant=='relation_only':
        assert a[3]==sum(d['entities'] for d in a[0].values()) and a[4]==base[4]
    if variant=='entity_only':assert a[4]==sum(d['relations'] for d in a[0].values())
    checks.append({'check':'immutable_filters_and_deterministic_derived_inputs','variant':variant,'passed':True})
atomic_json(ROOT/'reproduction/sota/paper_support/INPUT_CHECKS.json',{'status':'passed','checks':checks,'source_sha256':sha256(ROOT/'reproduction/sota/support_data.py'),'test_access':False})
print(json.dumps({'status':'passed','checks':len(checks),'test_access':False}))
