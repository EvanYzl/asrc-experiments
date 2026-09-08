"""Require all twelve fixed checkpoints, then freeze a provisional acceptance pass."""
import datetime
import json
from pathlib import Path
from frozen_data import ROOT,DOMAINS,atomic_json,sha256
base=ROOT/'reproduction/sota';config_path=base/'CANDIDATE_FREEZE.json'
candidate=json.loads(config_path.read_text());comparators=base/'PROVISIONAL_COMPARATORS_FREEZE.json'
assert comparators.exists()
frozen={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':candidate['method'],
    'candidate_freeze_sha256':sha256(config_path),'comparators_sha256':sha256(comparators),'checkpoints':{},
    'source_hashes':{},'protocol':'kbs-baselines-v1-20260905','mode':'provisional_acceptance_with_all_comparison_rows_retained',
    'scope_authority':'User asked to focus on SOTA and reuse local baselines. No additional baselines are launched. Unknowns/placeholders/repetition mismatches prevent a full comparable SOTA claim.',
    'stop_rule':'One final pass. Stop all newly launched task experiments after aggregation, regardless of threshold outcome; no candidate changes after observing test.'}
for name,digest in candidate['source_hashes'].items():assert sha256(base/name)==digest
for p in [base/'train_complex.py',base/'frozen_data.py',base/'final_evaluate.py',ROOT/'reproduction/strict_baselines/common.py']:
    frozen['source_hashes'][p.relative_to(ROOT).as_posix()]=sha256(p)
jobs=[]
for ds,opts in candidate['datasets'].items():
    for seed in candidate['seeds']:
        run=ROOT/opts['seed17_run'] if seed==17 else base/'confirmation'/f'asrc_{ds}_s{seed}'
        r=json.loads((run/'result.json').read_text());assert r['status']=='completed' and r['purpose']=='validation_only' and r['test_access'] is False
        assert r['seed']==seed and r['dataset']==ds
        assert sha256(run/'best.pt')==r['checkpoint_sha256']
        cfg=json.loads((run/'config.json').read_text());recipe=cfg['recipe']
        expected={'rank':256,'reg':.01,'lr':.1,'batch_size':512,'epochs':40,'valid_every':5,'patience':5,'sharing':opts['sharing']}
        for k,v in expected.items():assert recipe[k]==v,(ds,seed,k)
        assert cfg['source_hashes']==candidate['source_hashes']
        frozen['checkpoints'][f'{ds}_s{seed}']={'checkpoint':(run/'best.pt').relative_to(ROOT).as_posix(),'sha256':r['checkpoint_sha256'],
            'validation':r['validation'],'best_epoch':r['best_epoch'],'run_result_sha256':sha256(run/'result.json')}
        jid=f'final_{ds}_s{seed}';out=f'reproduction/sota/final/{jid}'
        jobs.append({'id':jid,'purpose':'One final evaluation of globally frozen candidate, with provisional comparison status disclosed',
            'changes':'No training, no tuning, no checkpoint selection. Audit validation replay and evaluate final test using shared strict evaluator.',
            'seed':seed,'output':out,'min_free_mib':10500,
            'command':['/root/zhishitupui/.envs/kgc/bin/python','reproduction/sota/final_evaluate.py','--dataset',ds,'--seed',str(seed),'--output',out],
            'next':'Aggregate all twelve fixed final results, compare exact provisional maxima, update all artifacts, stop this task experiments and await instruction.'})
assert len(frozen['checkpoints'])==12
target=base/'FINAL_EVALUATION_FREEZE.json';assert not target.exists(),'Never rewrite a final acceptance freeze'
atomic_json(target,frozen);atomic_json(base/'final_queue/manifest.json',{'jobs':jobs,'freeze_sha256':sha256(target)})
with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## Final provisional acceptance freeze — '+frozen['timestamp']+'\nPurpose: evaluate the single frozen candidate once after all three seeds complete.\n')
    f.write('Configuration: all 12 checkpoints, source and comparator registry hashes frozen in FINAL_EVALUATION_FREEZE.json; training settings unchanged.\n')
    f.write('Comparator status: existing local unrounded results reused; 2 user approximate cells, 16 unknown cells and 18 cells with unmatched single-seed repetition remain. These prevent a full comparable SOTA claim. E-PKG user approximations replaced by its already-completed local DMKGC run; exact column maxima recomputed over all retained cells.\n')
    f.write('Seeds/commands/PIDs/GPUs: 17/29/43 on each dataset, 12 evaluation-only jobs in final_queue/manifest.json; scheduler logs individual launches.\n')
    f.write('Preflight: scalar ComplEx formula and full-candidate filtered ranking/gold retention/ascending-ID ties passed; every final run replays validation before test without selecting another checkpoint.\n')
    f.write('Result paths: final/. Next: compute three-seed means/SD and exact threshold verdict; retain unknowns; publish only a provisional result if threshold passes; no test-guided changes or added baselines.\n')
print(json.dumps({'global_freeze':True,'checkpoints':12,'evaluation_jobs':12,'comparison':'provisional'}))
