"""Freeze reuse of the existing 17/29/43 checkpoints under the user's new request."""
import datetime
import json
import shutil
from frozen_data import ROOT,atomic_json,sha256

base=ROOT/'reproduction/sota';phase=base/'three_seed';phase.mkdir(exist_ok=True)
assert not (phase/'freeze.json').exists(),'Three-seed confirmation already frozen; inspect existing artifacts.'
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
archive=base/'history/before_three_seed';archive.mkdir(parents=True,exist_ok=True)
for name in ['TASK_SCOPE.json','PLAN.md','FINAL_DELIVERY_RECEIPT.json','FINAL_SYNC_VERIFICATION.json','FINAL_LOCAL_VERIFICATION.json']:
    if (base/name).exists():shutil.copy2(base/name,archive/name)
shutil.copy2(ROOT/'SOTA_LOG.md',archive/'SOTA_LOG.md')
old=json.loads((base/'history/unused_multiseed_acceptance/FINAL_EVALUATION_FREEZE.json').read_text())
entries={};jobs=[]
keys=['dataset','sharing','rank','reg','lr','epochs','batch_size','valid_every','patience','resume']
for ds in ['dbp5l','depkg','dwy','wk3l']:
    single=json.loads((base/'single_seed'/ds/'freeze.json').read_text())
    for seed in [17,29,43]:
        run=(ROOT/single['checkpoint']).parent if seed==17 else base/'confirmation'/f'asrc_{ds}_s{seed}'
        cfg=json.loads((run/'config.json').read_text());res=json.loads((run/'result.json').read_text())
        assert res['status']=='completed' and res['test_access'] is False and cfg['recipe']['seed']==seed
        assert {k:cfg['recipe'][k] for k in keys}=={k:single['recipe'][k] for k in keys}
        assert sha256(run/'config.json')==res['config_sha256']
        assert sha256(run/'best.pt')==res['checkpoint_sha256']==old['checkpoints'][f'{ds}_s{seed}']['sha256']
        for name,digest in cfg['source_hashes'].items():assert sha256(base/name)==digest
        result_path=base/'single_seed'/ds/'evaluation/result.json' if seed==17 else phase/'evaluation'/f'{ds}_s{seed}'/'result.json'
        entry={'dataset':ds,'seed':seed,'checkpoint':(run/'best.pt').relative_to(ROOT).as_posix(),
            'checkpoint_sha256':res['checkpoint_sha256'],'training_result_sha256':sha256(run/'result.json'),
            'config_sha256':sha256(run/'config.json'),'recipe':cfg['recipe'],'best_epoch':res['best_epoch'],
            'validation':res['validation'],'evaluation_result':result_path.relative_to(ROOT).as_posix(),
            'reuse_locked_test':seed==17,'checkpoint_already_frozen_before_seed17_tests':True}
        if seed==17:
            lock=base/'single_seed'/ds/'LOCKED_RESULT.json'
            entry.update(existing_test_result_sha256=sha256(result_path),existing_lock_sha256=sha256(lock))
            assert json.loads(lock.read_text())['result_sha256']==entry['existing_test_result_sha256']
        else:
            assert not result_path.exists() and not result_path.with_name('TEST_OPENED.json').exists()
            jobs.append({'id':f'three_{ds}_s{seed}','purpose':f'Complete ASRC three-seed reporting: one frozen test for {ds}, seed {seed}',
                'changes':'Reuse previously trained validation-only checkpoint; same frozen algorithm/hyperparameters; no training, tuning or baseline runs',
                'seed':seed,'min_free_mib':9500,'output':result_path.parent.relative_to(ROOT).as_posix(),
                'command':['/root/zhishitupui/.envs/kgc/bin/python','reproduction/sota/evaluate_three_seed.py','--dataset',ds,'--seed',str(seed)],
                'next':'Retain immutable result; aggregate exactly seeds17/29/43, KG macro then seed mean and sample SD; update Table2 and compile.'})
        entries[f'{ds}_s{seed}']=entry
freeze={'timestamp':stamp,'authorization':'User requested completing our method to three seeds and updating/compiling KBS_Main_Text_Tables.pdf.',
    'method':'ASRC','seeds':[17,29,43],'entries':entries,'new_test_jobs':8,'seed17_retested':False,'new_training_jobs':0,'new_baseline_jobs':0,
    'protocol':'kbs-baselines-v1-20260905','aggregation':'Equal KG macro per seed, then arithmetic mean and sample standard deviation (ddof=1) across seeds17/29/43.',
    'reference_registry_sha256':sha256(base/'SINGLE_SEED_REFERENCES.json'),
    'prior_global_freeze_sha256':sha256(base/'history/unused_multiseed_acceptance/FINAL_EVALUATION_FREEZE.json'),
    'source_hashes':{f'reproduction/sota/{name}':sha256(base/name) for name in ['train_complex.py','frozen_data.py','evaluate_three_seed.py']},
    'common_sha256':sha256(ROOT/'reproduction/strict_baselines/common.py'),
    'test_policy':'Once for each missing seed; seed17 locked results reused; no test-based reselection or tuning.'}
atomic_json(phase/'freeze.json',freeze);atomic_json(phase/'queue/manifest.json',{'jobs':jobs})
atomic_json(base/'TASK_SCOPE.json',{'timestamp':stamp,'phase':'three_seed_confirmation','status':'frozen_evaluation_only','method':'ASRC','seeds':[17,29,43],
    'authorized_new_tests':8,'new_training':False,'baseline_runs':False,'retest_seed17':False,'tuning':False,
    'next':'Evaluate missing tests, aggregate mean and sample SD, update requested Table2 PDF and sync, then stop.'})
with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## New user authorization: three-seed completion — '+stamp+'\n')
    f.write('Purpose: complete only our ASRC method at seeds17/29/43, update Table2 and compile the requested PDF. This supersedes the prior single-seed stopping scope.\n')
    f.write('Changes/config: all twelve checkpoints and input/code hashes checked; all eight existing seeds29/43 match seed17 hyperparameters and the archived pre-test global freeze. No retraining or new baseline.\n')
    f.write('Command/PID/GPU: prepare_three_seed.py creates eight evaluation-only jobs; run_sota_queue.py assigns six free GPUs and records every PID/command.\n')
    f.write('Artifacts: three_seed/freeze.json, three_seed/queue/manifest.json, three_seed/evaluation/<dataset>_s<seed>. Seed17 locked results are reused.\n')
    f.write('Conclusion: historical confirmations are now explicitly authorized for three-seed reporting; no candidate or checkpoint changed using test results. Next: one test for each missing seed, then mean/sample-SD Table2 and compilation.\n')
plan=base/'PLAN.md';plan.write_text('# Current authorization: ASRC three-seed confirmation\n\nUser requested completing our method at seeds17/29/43 and updating/compiling outputs/kbs/_main/_tables/KBS_Main_Text_Tables.pdf. Reuse existing training and locked seed17 tests; run only eight missing once-only tests, then aggregate equal-KG macro followed by seed mean and sample SD. No tuning or baseline runs. Current freeze: three_seed/freeze.json. Previous single-seed phase and stopping instructions are historical below.\n\n'+plan.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'frozen_checkpoints':12,'new_tests':8,'training_jobs':0,'baseline_jobs':0,'seeds':[17,29,43]}))
