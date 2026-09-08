"""Verify every raw test rank, then report exactly three-seed mean and sample SD."""
import argparse
import csv
import datetime
import json
import numpy as np
from frozen_data import ROOT,DOMAINS,atomic_json,sha256

p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');args=p.parse_args()
base=ROOT/'reproduction/sota';phase=base/'three_seed';fp=phase/'freeze.json';freeze=json.loads(fp.read_text())
refs_path=base/'SINGLE_SEED_REFERENCES.json';refs=json.loads(refs_path.read_text());assert sha256(refs_path)==freeze['reference_registry_sha256']
for path,digest in freeze['source_hashes'].items():assert sha256(ROOT/path)==digest
common=base/'runtime/server_common.py' if (base/'runtime/server_common.py').exists() else ROOT/'reproduction/strict_baselines/common.py'
assert sha256(common)==freeze['common_sha256']
datasets={};rows=[];raw=[];wins=0;all_deficits=True
for ds in DOMAINS:
    seeds={}
    for seed in [17,29,43]:
        e=freeze['entries'][f'{ds}_s{seed}'];rp=ROOT/e['evaluation_result'];result=json.loads(rp.read_text());out=rp.parent
        assert result['status']=='completed' and result['seed']==seed and result['test_used_for_selection'] is False
        assert result['protocol']==freeze['protocol'] and result['checkpoint_sha256']==e['checkpoint_sha256']
        cp=ROOT/e['checkpoint'];assert sha256(cp)==e['checkpoint_sha256']
        cfgpath=cp.with_name('config.json');cfg=json.loads(cfgpath.read_text())
        assert sha256(cfgpath)==e['config_sha256'] and sha256(cp.with_name('result.json'))==e['training_result_sha256']
        for kg,digest in cfg['manifests'].items():
            mp=ROOT/'reproduction/strict_baselines/data_manifests'/f'{ds}_{kg}.json';assert sha256(mp)==digest
            for name,fd in json.loads(mp.read_text())['files'].items():assert sha256(ROOT/name.replace('\\','/'))==fd
        for name,item in cfg['alignment']['alignment_files'].items():assert sha256(ROOT/name)==item['sha256']
        if seed==17:
            assert sha256(rp)==e['existing_test_result_sha256']
            assert sha256(base/'single_seed'/ds/'LOCKED_RESULT.json')==e['existing_lock_sha256']
        else:
            assert result['freeze_sha256']==sha256(fp)
            assert json.loads((out/'TEST_OPENED.json').read_text())['freeze_sha256']==sha256(fp)
        per={}
        for kg in DOMAINS[ds]:
            mp=ROOT/'reproduction/strict_baselines/data_manifests'/f'{ds}_{kg}.json';m=json.loads(mp.read_text())
            paths=[ROOT/name.replace('\\','/') for name in m['files'] if name.replace('\\','/').endswith((f'{kg}-test.tsv','test_triple_id.txt'))];assert len(paths)==1
            expected=np.loadtxt(paths[0],dtype=np.int64,ndmin=2);qp=out/kg/'test_queries.npz'
            with np.load(qp) as a:
                np.testing.assert_array_equal(a['triples'],expected);np.testing.assert_array_equal(a['query_index'],np.arange(len(expected)))
                assert len(expected)==m['counts']['test'] and np.isfinite(a['gold_score']).all()
                assert (a['top10_ids']>=0).all() and (a['top10_ids']<m['entities']).all()
                calculated={}
                for filt in ['train','train_valid','all']:
                    rank=a['rank_'+filt].astype(np.float64)
                    assert len(rank)==len(expected) and np.isfinite(rank).all() and (rank>=1).all() and (rank<=m['entities']).all()
                    metrics={'mrr':float(np.mean(1/rank)),'h1':float(np.mean(rank<=1)),'h3':float(np.mean(rank<=3)),'h10':float(np.mean(rank<=10))}
                    for k,v in metrics.items():assert v==result['per_kg'][kg]['metrics'][filt][k]
                    calculated[filt]=metrics
                primary='all' if ds=='wk3l' else 'train_valid';assert result['per_kg'][kg]['primary_filter']==primary
                per[kg]=calculated[primary]
            raw.append({'dataset':ds,'seed':seed,'kg':kg,'path':qp.relative_to(ROOT).as_posix(),'sha256':sha256(qp),'queries':len(expected)})
        macro={k:float(np.mean([v[k] for v in per.values()])) for k in ['mrr','h1','h3','h10']};assert macro==result['macro']
        seeds[str(seed)]={'macro':macro,'per_kg':per,'result_path':e['evaluation_result'],'result_sha256':sha256(rp),'checkpoint_sha256':e['checkpoint_sha256'],'best_epoch':e['best_epoch']}
    mean={k:float(np.mean([s['macro'][k] for s in seeds.values()])) for k in ['mrr','h1','h3','h10']}
    std={k:float(np.std([s['macro'][k] for s in seeds.values()],ddof=1)) for k in mean}
    comparisons=[]
    for metric in ['mrr','h1','h10']:
        ref=refs['datasets'][ds][metric];diff=mean[metric]-ref['value'];wins+=int(diff>0);all_deficits &= diff>=-.005
        row={'dataset':ds,'metric':metric,'seeds':'17;29;43','mean':mean[metric],'sample_sd':std[metric],'reference':ref['value'],
            'difference_percentage_points':diff*100,'strict_win':diff>0,'reference_status':ref['status']};rows.append(row);comparisons.append(row)
    datasets[ds]={'seeds':seeds,'mean':mean,'sample_sd':std,'comparisons':comparisons}
report={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'ASRC','seeds':[17,29,43],'datasets':datasets,
    'aggregation':'equal KG macro within seed; arithmetic mean and sample SD (ddof=1) across exactly three seeds',
    'raw_query_artifacts':raw,'raw_artifacts_verified':len(raw),'test_result_files_verified':12,'new_test_runs':8,'seed17_retested':False,
    'new_training_runs':0,'baseline_runs':0,'strict_wins':wins,'all_deficits_within_half_point':bool(all_deficits),
    'threshold_met':bool(wins>=10 and all_deficits),'verdict':'三种子均值达到暂定门槛' if wins>=10 and all_deficits else '三种子均值未达到暂定门槛',
    'full_comparable_sota_claim':False,'unknown_comparison_cells':refs['unknown_cells'],'user_provisional_references_retained':True,
    'freeze_sha256':sha256(fp),'reference_registry_sha256':sha256(refs_path),'no_test_based_reselection':True}
if args.check_only:
    old=json.loads((phase/'RESULTS.json').read_text());assert old['datasets']==datasets and old['raw_query_artifacts']==raw
else:
    atomic_json(phase/'RESULTS.json',report)
    with (phase/'COMPARISON.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({'verdict':report['verdict'],'wins':wins,'seeds':[17,29,43],'raw_rank_files_verified':len(raw),
    'results':{ds:{'mean':d['mean'],'sample_sd':d['sample_sd']} for ds,d in datasets.items()}},ensure_ascii=False))
