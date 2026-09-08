"""Audit raw ranks and aggregate every pre-registered supporting result."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from frozen_data import ROOT,DOMAINS,atomic_json,sha256
from support_data import build_inputs

def scalar(values):
    values=[float(x) for x in values];assert len(values)==3 and np.isfinite(values).all()
    return {'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1)),'seeds':values}

def metrics(r):
    r=np.asarray(r,dtype=np.float64)
    return {'mrr':float(np.mean(1/r)),'h1':float(np.mean(r==1)),'h3':float(np.mean(r<=3)),'h10':float(np.mean(r<=10))}

def paired(a,b):
    # a is the reference independent rank; b is the compared variant.
    return {'ntr':float(np.mean(b>a)),'ptr':float(np.mean(b<a)),
        'harm':float(np.maximum(1/a.astype(float)-1/b.astype(float),0).mean()),
        'delta':float(np.mean(1/b.astype(float)-1/a.astype(float)))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--require-profiles',action='store_true');p.add_argument('--check-only',action='store_true');args=p.parse_args()
    phase=ROOT/'reproduction/sota/paper_support';fp=phase/'EVALUATION_FREEZE.json';freeze=json.loads(fp.read_text());models=freeze['models'];seeds=[17,29,43]
    for path,digest in freeze['input_hashes'].items():assert sha256(ROOT/path)==digest,path
    assert sha256(ROOT/'reproduction/sota/three_seed/RESULTS.json')==freeze['table2_results_sha256']
    datasets={};expected={};strata={};ranks={};results={};rawfiles={}
    for ds in DOMAINS:
        data,maps,_,n,_,_,_,_=build_inputs(models[f'{ds}_shared_s17']['recipe']);datasets[ds]=data
        membership=np.bincount(np.concatenate(list(maps.values())),minlength=n)
        for kg in DOMAINS[ds]:
            d=data[kg];paths=[ROOT/x.replace('\\','/') for x in d['manifest']['files'] if Path(x.replace('\\','/')).name in [f'{kg}-test.tsv','test_triple_id.txt']]
            assert len(paths)==1;test=np.loadtxt(paths[0],dtype=np.int64,ndmin=2);expected[(ds,kg)]=test
            strata[(ds,kg)]=membership[maps[kg][test[:,0]]]>1
    for ident,e in models.items():
        rp=ROOT/e['evaluation_path']/'result.json';r=json.loads(rp.read_text());assert r['status']=='completed' and r['test_used_for_selection'] is False
        if e['reuse_locked_test']:assert sha256(rp)==e['locked_result_sha256']
        else:assert r['freeze_sha256']==sha256(fp) and (rp.parent/'TEST_OPENED.json').exists()
        assert r['checkpoint_sha256']==e['checkpoint_sha256'];per={};ranks[ident]={};rawfiles[rp.relative_to(ROOT).as_posix()]=sha256(rp)
        for kg,v in r['per_kg'].items():
            path=rp.parent/kg/'test_queries.npz';z=np.load(path,allow_pickle=False);test=expected[(e['dataset'],kg)]
            assert np.array_equal(z['triples'],test) and np.array_equal(z['query_index'],np.arange(len(test)))
            assert np.isfinite(z['gold_score']).all() and z['top10_ids'].shape==(len(test),10)
            for protocol,m in v['metrics'].items():
                a=z['rank_'+protocol];assert len(a)==len(test) and (a>=1).all() and (a<=datasets[e['dataset']][kg]['entities']).all()
                calculated=metrics(a)
                for key,val in calculated.items():assert abs(val-m[key])<1e-14,(ident,kg,protocol,key)
            primary='all' if e['dataset']=='wk3l' else 'train_valid';assert v['primary_filter']==primary
            ranks[ident][kg]=z['rank_'+primary];per[kg]=metrics(ranks[ident][kg]);rawfiles[path.relative_to(ROOT).as_posix()]=sha256(path)
        macro={k:float(np.mean([v[k] for v in per.values()])) for k in ['mrr','h1','h3','h10']}
        for key,val in macro.items():assert abs(val-r['macro'][key])<1e-14
        results[ident]={'macro':macro,'per_kg':per,'parameters':e['parameters'],'checkpoint_sha256':e['checkpoint_sha256']}
    tables={f'T{i}':[] for i in range(4,9)}
    def sample(ds,var,seed):return results[f'{ds}_{var}_s{seed}']['macro']
    def paired_macro(ds,ref,var,seed,group=None):
        vals=[]
        for kg in DOMAINS[ds]:
            a=ranks[f'{ds}_{ref}_s{seed}'][kg];b=ranks[f'{ds}_{var}_s{seed}'][kg]
            if group is not None:
                mask=strata[(ds,kg)] if group=='aligned' else ~strata[(ds,kg)];a=a[mask];b=b[mask]
            if len(a):vals.append(dict(paired(a,b),independent_mrr=metrics(a)['mrr'],shared_mrr=metrics(b)['mrr']))
        if not vals:return None
        return {key:float(np.mean([v[key] for v in vals])) for key in vals[0]}
    for ds in DOMAINS:
        for row,var in [('independent','independent'),('shared','shared'),('asrc','independent' if ds=='depkg' else 'shared')]:
            stats={'mrr':scalar([100*sample(ds,var,s)['mrr'] for s in seeds])}
            for key in ['delta','ntr','ptr','harm']:stats[key]=scalar([100*paired_macro(ds,'independent',var,s)[key] for s in seeds])
            tables['T4'].append({'key':ds+'.'+row,'dataset':ds,'variant':row,'model_variant':var,'stats':stats})
    for var in ['shared','independent','entity_only','relation_only','no_reciprocal','no_n3']:
        stats={key:scalar([100*sample('dbp5l',var,s)[key] for s in seeds]) for key in ['mrr','h1','h10']}
        stats['delta']=scalar([100*(sample('dbp5l',var,s)['mrr']-sample('dbp5l','shared',s)['mrr']) for s in seeds])
        stats['params_m']=scalar([models[f'dbp5l_{var}_s{s}']['parameters']/1e6 for s in seeds])
        tables['T5'].append({'key':'dbp5l.'+var,'dataset':'dbp5l','variant':var,'stats':stats})
    for ds in DOMAINS:
        for group in ['aligned','unaligned']:
            count=sum(int((a if group=='aligned' else ~a).sum()) for (d,kg),a in strata.items() if d==ds)
            nkg=sum(bool((a if group=='aligned' else ~a).any()) for (d,kg),a in strata.items() if d==ds)
            values=[paired_macro(ds,'independent','shared',s,group) for s in seeds]
            stats={key:scalar([100*v[key] for v in values]) for key in ['independent_mrr','shared_mrr','delta','ntr']} if count else {}
            tables['T6'].append({'key':ds+'.'+group,'dataset':ds,'group':group,'queries':count,'nonempty_kgs':nkg,'stats':stats})
    for condition in ['clean','train50','align50','noise10']:
        for mode in ['independent','shared']:
            var=mode if condition=='clean' or (mode=='independent' and condition in ['align50','noise10']) else condition+'_'+mode
            stats={key:scalar([100*sample('dbp5l',var,s)[key] for s in seeds]) for key in ['mrr','h1','h10']}
            stats['delta']=scalar([100*(sample('dbp5l',var,s)['mrr']-sample('dbp5l',mode,s)['mrr']) for s in seeds])
            tables['T7'].append({'key':'dbp5l.'+condition+'.'+mode,'condition':condition,'mode':mode,'model_variant':var,'stats':stats})
    profiles={};required=set()
    for ds in DOMAINS:
        for row,var in [('independent','independent'),('asrc','independent' if ds=='depkg' else 'shared')]:
            ids=[f'{ds}_{var}_s{s}' for s in seeds];required.update(ids)
            if not all((phase/'profiling'/i/'result.json').exists() for i in ids):continue
            for ident in ids:
                path=phase/'profiling'/ident/'result.json';profile=json.loads(path.read_text());assert profile['status']=='completed' and profile['spec']==freeze['profiling']
                for kg,v in profile['per_kg'].items():
                    tp=path.parent/kg/'timings.npz';z=np.load(tp);a=z['latency'];b=z['throughput'];assert a.shape[0]==10*v['queries'] and b.shape[0]==10
                    assert np.array_equal(z['triples'],datasets[ds][kg]['arrays']['val_select'][:256])
                    assert (a[:,3]>0).all() and (b[:,3]>0).all()
                    assert abs(np.quantile(a[:,3]*1000,.5)-v['p50_ms'])<1e-10
                    assert abs(np.quantile(a[:,3]*1000,.95)-v['p95_ms'])<1e-10
                    assert abs(b[:,2].sum()/b[:,3].sum()-v['qps'])<1e-9
                    rawfiles[tp.relative_to(ROOT).as_posix()]=sha256(tp)
                profiles[ident]=profile;rawfiles[path.relative_to(ROOT).as_posix()]=sha256(path)
            stats={k:scalar([profiles[i]['macro'][k] for i in ids]) for k in ['params_m','vram_mib','rss_mib','p50_ms','p95_ms','qps']}
            tables['T8'].append({'key':ds+'.'+row,'dataset':ds,'variant':row,'model_variant':var,'stats':stats})
    assert len(required)==21
    if args.require_profiles:assert set(profiles)==required and len(tables['T8'])==8
    report={'status':'completed' if len(profiles)==21 else 'accuracy_complete_profiling_pending','freeze_sha256':sha256(fp),
            'models':results,'tables':tables,'raw_hashes':rawfiles,'raw_rank_models':48,'profiles':len(profiles),'seed_order':seeds,
            'aggregation':'equal KG macro then three-seed mean and sample SD; T6 conditional macro over nonempty KGs',
            'table2_unchanged':True,'test_used_for_selection':False,'external_baselines_started':0}
    if args.check_only:assert json.loads((phase/'RESULTS.json').read_text())==report
    else:atomic_json(phase/'RESULTS.json',report)
    print(json.dumps({'status':report['status'],'models':len(results),'profiles':len(profiles),'table_rows':{k:len(v) for k,v in tables.items()},'audited_raw_files':len(rawfiles)}))

if __name__=='__main__':main()
