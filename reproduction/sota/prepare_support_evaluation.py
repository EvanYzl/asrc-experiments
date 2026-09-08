"""Freeze all validation-selected checkpoints before opening any new test arrays."""
import datetime
import json
from frozen_data import ROOT,atomic_json,sha256
from support_data import build_inputs
from run_sota_queue import append_log
base=ROOT/'reproduction/sota';phase=base/'paper_support';plan=json.loads((phase/'PLAN.json').read_text())
assert (phase/'TRAINING_COMPLETED.json').exists()
assert not (phase/'EVALUATION_FREEZE.json').exists()
sf=json.loads((phase/'TRAINING_SOURCE_FREEZE.json').read_text())
assert sha256(phase/'PLAN.json')==sf['plan_sha256']
for path,digest in sf['source_hashes'].items():assert sha256(ROOT/path)==digest
assert sha256(ROOT/'outputs/kbs/_main/_tables/tables/t02.tex')==plan['table2_source_sha256']
assert sha256(base/'three_seed/RESULTS.json')==plan['table2_results_sha256']
models={};jobs=[];inputfiles={}
for ident,e in plan['models'].items():
    cp=ROOT/e['training_path']/'best.pt';result=json.loads(cp.with_name('result.json').read_text());config=json.loads(cp.with_name('config.json').read_text())
    assert result['status']=='completed' and result['test_access'] is False
    assert result['checkpoint_sha256']==sha256(cp)
    assert sha256(cp.with_name('config.json'))==result['config_sha256']
    assert result['seed']==e['seed'] and result['dataset']==e['dataset']
    curve=[json.loads(x) for x in cp.with_name('learning_curve.jsonl').read_text().splitlines() if x.strip()]
    measured=[x for x in curve if 'validation' in x]
    selected=max(measured,key=lambda x:x['validation']['macro']['mrr'])
    assert selected['epoch']==result['best_epoch'] and selected['validation']['macro']==result['validation']['macro']
    actual_recipe=config['recipe']
    for key in ['rank','reg','lr','epochs','batch_size','valid_every','patience','seed']:
        assert actual_recipe[key]==e['recipe'][key],(ident,key)
    if e['reuse_training']:assert sha256(cp.with_name('result.json'))==e['reused_training_result_sha256']
    else:
        assert result['completed_epochs']<=40 and not config.get('smoke_only',False)
        if e['trainer']=='support':assert config['recipe']==e['recipe']
    data,maps,roff,n,nr,audit,_,_=build_inputs(e['recipe'])
    assert audit['entity_map_sha256']==config['alignment']['entity_map_sha256']
    assert nr==config['alignment']['relations'] and n==config['alignment']['shared_entities']
    for kg,d in data.items():
        inputfiles.update({p.replace('\\','/'):h for p,h in d['manifest']['files'].items()})
        mp=ROOT/'reproduction/strict_baselines/data_manifests'/f"{e['dataset']}_{kg}.json";inputfiles[mp.relative_to(ROOT).as_posix()]=sha256(mp)
    inputfiles.update({p:v['sha256'] for p,v in audit['alignment_files'].items()})
    e=dict(e,checkpoint_sha256=sha256(cp),training_result_sha256=sha256(cp.with_name('result.json')),validation=result['validation'],
           best_epoch=result['best_epoch'],entity_map_sha256=audit['entity_map_sha256'],parameters=(n+2*nr)*2*e['recipe']['rank'])
    if e['reuse_locked_test']:assert sha256(ROOT/e['evaluation_path']/'result.json')==e['locked_result_sha256']
    else:assert not (ROOT/e['evaluation_path']/'TEST_OPENED.json').exists()
    models[ident]=e
    if not e['reuse_locked_test']:
        jobs.append({'id':'test_'+ident,'purpose':'Once-only pre-registered support test, retaining all positive and negative results',
            'changes':e['recipe'],'seed':e['seed'],'output':e['evaluation_path'],'min_free_mib':9500,
            'command':['/root/zhishitupui/.envs/kgc/bin/python','reproduction/sota/evaluate_support.py','--id',ident],
            'next':'Aggregate all registered controls without reselection or any change to Table2.'})
assert len(jobs)==36
paths=['train_complex.py','frozen_data.py','train_support.py','support_data.py','evaluate_support.py','profile_support.py','aggregate_support.py']
sources={('reproduction/sota/'+p):sha256(base/p) for p in paths}
sources['reproduction/strict_baselines/common.py']=sha256(ROOT/'reproduction/strict_baselines/common.py')
freeze={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':models,'source_hashes':sources,'input_hashes':inputfiles,
        'protocol':'kbs-baselines-v1-20260905','plan_sha256':sha256(phase/'PLAN.json'),'table2_results_sha256':plan['table2_results_sha256'],
        'selection':'earliest maximum val_select macro MRR, train+val_select filter; no test selection','statistics':plan['statistics'],'profiling':plan['profiling']}
atomic_json(phase/'EVALUATION_FREEZE.json',freeze);atomic_json(phase/'test_queue/manifest.json',{'jobs':jobs})
append_log('## Supporting evaluation freeze — '+freeze['timestamp']+'\nPurpose: close selection before new tests. All 48 recipes and checkpoints frozen; 14 trainings and 12 tests reused.\nChanges/config/seeds: PLAN.json and EVALUATION_FREEZE.json, seeds 17/29/43. Command: python reproduction/sota/prepare_support_evaluation.py; GPU: none.\nOutputs: EVALUATION_FREEZE.json; test_queue/manifest.json (36 once-only tests). Conclusion: validation-only checkpoints and immutable source/input hashes verified. Next: run frozen test queue and isolated validation-query timing, retain every outcome.')
print(json.dumps({'frozen_models':len(models),'new_once_only_tests':len(jobs),'input_files':len(inputfiles)}))
