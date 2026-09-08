"""Recompute all single-seed test numbers from raw ranks and verify frozen provenance."""
import csv
import argparse
import datetime
import json
from pathlib import Path
import numpy as np
from frozen_data import ROOT,DOMAINS,atomic_json,sha256
parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
base=ROOT/'reproduction/sota';refs_path=base/'SINGLE_SEED_REFERENCES.json';refs=json.loads(refs_path.read_text())
order=['dbp5l','depkg','dwy','wk3l'];datasets={};checks=[];wins=0;deficits_ok=True;rows=[]
for ds in order:
    folder=base/'single_seed'/ds;lock_path=folder/'LOCKED_RESULT.json'
    if not lock_path.exists():break
    lock=json.loads(lock_path.read_text());freeze_path=folder/'freeze.json';freeze=json.loads(freeze_path.read_text())
    out=folder/'evaluation';rp=out/'result.json';r=json.loads(rp.read_text())
    assert lock['result_sha256']==sha256(rp) and r['freeze_sha256']==sha256(freeze_path)
    assert r['seed']==17 and r['purpose']=='single_seed_frozen_test' and r['test_used_for_selection'] is False
    assert r['references_sha256']==sha256(refs_path) and sha256(ROOT/r['checkpoint'])==r['checkpoint_sha256']
    for name,digest in freeze['source_hashes'].items():assert sha256(ROOT/name)==digest
    runtime_common=base/'runtime/server_common.py' if (base/'runtime/server_common.py').exists() else ROOT/'reproduction/strict_baselines/common.py'
    assert sha256(runtime_common)==freeze['common_sha256']
    pilot=ROOT/r['checkpoint'];pilot_result=json.loads(pilot.with_name('result.json').read_text())
    pilot_config_path=pilot.with_name('config.json');pilot_config=json.loads(pilot_config_path.read_text())
    assert sha256(pilot.with_name('result.json'))==freeze['validation_result_sha256']
    assert sha256(pilot_config_path)==pilot_result['config_sha256']
    for kg,digest in pilot_config['manifests'].items():
        mp=ROOT/'reproduction/strict_baselines/data_manifests'/f'{ds}_{kg}.json'
        assert sha256(mp)==digest
        for name,file_digest in json.loads(mp.read_text())['files'].items():assert sha256(ROOT/name.replace('\\','/'))==file_digest
    for name,item in pilot_config['alignment']['alignment_files'].items():assert sha256(ROOT/name)==item['sha256']
    for prior in freeze['predecessor_locks']:assert sha256(base/'single_seed'/prior['dataset']/'LOCKED_RESULT.json')==prior['lock_sha256']
    opened=json.loads((out/'TEST_OPENED.json').read_text());assert opened['freeze_sha256']==sha256(freeze_path)
    per={};raw=[]
    for kg in DOMAINS[ds]:
        query=out/kg/'test_queries.npz';manifest_path=ROOT/'reproduction/strict_baselines/data_manifests'/f'{ds}_{kg}.json'
        manifest=json.loads(manifest_path.read_text());test_paths=[ROOT/k.replace('\\','/') for k in manifest['files'] if k.replace('\\','/').endswith((f'{kg}-test.tsv','test_triple_id.txt'))]
        assert len(test_paths)==1
        original=np.loadtxt(test_paths[0],dtype=np.int64,ndmin=2)
        with np.load(query) as a:
            np.testing.assert_array_equal(a['triples'],original);np.testing.assert_array_equal(a['query_index'],np.arange(len(original)))
            assert len(original)==manifest['counts']['test']
            primary='all' if ds=='wk3l' else 'train_valid';ranks=a['rank_'+primary].astype(np.float64)
            assert len(ranks)==len(original) and np.isfinite(ranks).all() and (ranks>=1).all() and (ranks<=manifest['entities']).all()
            assert np.isfinite(a['gold_score']).all() and (a['top10_ids']>=0).all() and (a['top10_ids']<manifest['entities']).all()
            per[kg]={'mrr':float(np.mean(1/ranks)),'h1':float(np.mean(ranks<=1)),'h3':float(np.mean(ranks<=3)),'h10':float(np.mean(ranks<=10))}
            for protocol in ['train','train_valid','all']:
                rr=a['rank_'+protocol].astype(np.float64)
                calculated={'mrr':float(np.mean(1/rr)),'h1':float(np.mean(rr<=1)),'h3':float(np.mean(rr<=3)),'h10':float(np.mean(rr<=10))}
                for k,v in calculated.items():assert v==r['per_kg'][kg]['metrics'][protocol][k]
        raw.append({'kg':kg,'queries':len(original),'artifact':query.relative_to(ROOT).as_posix(),'sha256':sha256(query),'manifest_sha256':sha256(manifest_path)})
    macro={k:float(np.mean([p[k] for p in per.values()])) for k in ['mrr','h1','h3','h10']}
    assert macro==r['macro'] and macro==lock['macro']
    dsrows=[]
    for metric in ['mrr','h1','h10']:
        ref=refs['datasets'][ds][metric];delta=macro[metric]-ref['value'];win=delta>0;deficits_ok &= delta>=-.005;wins+=int(win)
        item={'dataset':ds,'metric':metric,'seed':17,'value':macro[metric],'reference':ref['value'],'difference_percentage_points':delta*100,
              'strict_win':win,'reference_status':ref['status'],'result_path':rp.relative_to(ROOT).as_posix(),'result_sha256':sha256(rp)}
        rows.append(item);dsrows.append(item)
    datasets[ds]={'seed':17,'macro':macro,'per_kg':per,'comparisons':dsrows,'raw_artifacts':raw,'locked_result_sha256':sha256(lock_path),
                  'checkpoint_sha256':r['checkpoint_sha256'],'best_epoch':freeze['best_epoch'],'recipe':freeze['recipe']}
remaining=12-3*len(datasets)
report={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'ASRC','seed':17,'datasets':datasets,
        'verified_datasets':len(datasets),'strict_wins':wins,'metrics_verified':len(rows),'remaining_metrics':remaining,
        'all_deficits_within_half_point':bool(deficits_ok),'can_still_reach_overall_threshold':bool(deficits_ok and wins+remaining>=10),
        'threshold_met':bool(remaining==0 and wins>=10 and deficits_ok),'comparison_status':'single_seed_provisional',
        'verdict':'单种子达到暂定门槛' if remaining==0 and wins>=10 and deficits_ok else ('尚未完成四数据集验收' if remaining else '单种子未达到暂定门槛'),
        'full_comparable_sota_claim':False,'unknown_comparison_cells':refs['unknown_cells'],
        'user_provisional_reference_lines_retained':True,'new_baseline_runs':0,'additional_seeds_used_in_acceptance':[],
        'integrity_checks':['test triples and order exactly match original frozen inputs','all three filter variants recomputed from full raw ranks',
        'KG macro averages recomputed without rounding','locked test/checkpoint/code/reference hashes verified','single seed17 only; no result chosen from other seeds',
        'all 75 external comparison cells retained, including unknowns; user references not labeled reproduction']}
if not args.check_only:
    atomic_json(base/'SINGLE_SEED_ACCEPTANCE.json',report)
    with (base/'SINGLE_SEED_COMPARISON.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['dataset']);writer.writeheader();writer.writerows(rows)
else:
    previous=json.loads((base/'SINGLE_SEED_ACCEPTANCE.json').read_text())
    assert previous['datasets']==datasets and previous['threshold_met']==report['threshold_met']
print(json.dumps({'verdict':report['verdict'],'verified_datasets':len(datasets),'wins':wins,'metrics':len(rows),'macro':{ds:d['macro'] for ds,d in datasets.items()}},ensure_ascii=False))
