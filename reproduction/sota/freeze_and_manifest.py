"""Freeze candidate from validation only and prepare bounded confirmation/comparator jobs."""
import datetime
import json
from pathlib import Path
from frozen_data import ROOT,atomic_json,sha256
base=ROOT/'reproduction/sota'
settings={'dbp5l':'shared','depkg':'independent','dwy':'shared','wk3l':'shared'}
freeze={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'candidate_configuration_frozen_before_confirmation',
        'method':'ASRC (Alignment-selective Reciprocal ComplEx)','source_hashes':{p.name:sha256(p) for p in [base/'train_complex.py',base/'frozen_data.py']},
        'selection_basis':'Only seed-17 val_select macro MRR; six predeclared configurations; no candidate test access.',
        'datasets':{},'seeds':[17,29,43],
        'rule':'rank256, N3=0.01, Adagrad lr0.1, batch512, initialization std0.001, max40 epochs, validate every5, patience5; select maximum val_select macro MRR, earliest tie',
        'no_further_candidate_search':True,'test_gate':'Do not run candidate test until all confirmation checkpoints and all comparable baseline maxima are frozen. Final failure stops optimization; no test-guided retry.'}
jobs=[]
for ds,sharing in settings.items():
    folder=base/'pilot02'/f'{ds}_{sharing}_r256_n3p01_s17'
    r=json.loads((folder/'result.json').read_text());assert r['purpose']=='validation_only' and r['test_access'] is False
    assert sha256(folder/'best.pt')==r['checkpoint_sha256']
    freeze['datasets'][ds]={'sharing':sharing,'seed17_run':str(folder.relative_to(ROOT)),
        'seed17_checkpoint_sha256':r['checkpoint_sha256'],'seed17_validation':r['validation'],'seed17_best_epoch':r['best_epoch']}
    for seed in [29,43]:
        jid=f'asrc_{ds}_s{seed}';out=f'reproduction/sota/confirmation/{jid}'
        cmd=['/root/zhishitupui/.envs/kgc/bin/python','reproduction/sota/train_complex.py','--dataset',ds,'--sharing',sharing,
             '--rank','256','--reg','0.01','--lr','0.1','--batch-size','512','--epochs','40','--valid-every','5','--patience','5','--seed',str(seed),'--output',out]
        jobs.append({'id':jid,'purpose':'Confirm frozen ASRC at the two remaining seeds; validation only; no hyperparameter changes','changes':f'Only seed changes to {seed}; dataset sharing setting frozen from val_select','seed':seed,'output':out,'command':cmd,'min_free_mib':10500})
for ds in ['wk3l','dbp5l','depkg','dwy']:
    for method in ['LSMGA','DMKGC','IMKGC']:
        for seed in ([17,29,43] if ds=='wk3l' else [29,43]):
            jid=f'{method.lower()}_{ds}_s{seed}';out=f'reproduction/sota/baselines/{jid}'
            jobs.append({'id':jid,'purpose':'Complete every Table2 comparator with matched three-seed repetitions; replace provisional/unknown baseline entries',
                         'changes':'WK3l audited lossless format adapter, original graph losses/default recipes, FR validation selection' if ds=='wk3l' else 'Exact existing audited core recipe and fixed split; repeat with the missing seed on the six-GPU server',
                         'seed':seed,'output':out,'min_free_mib':10500,
                         'command':['/root/zhishitupui/.envs/kgc/bin/python','reproduction/sota/run_graph_sota.py','--method',method,'--dataset',ds,'--seed',str(seed),'--output',out],
                         'next':'Retain all scores and query ranks; aggregate only once all three seeds are present; never use baseline tests to retune the frozen candidate.'})
for m in ['lsmga','dmkgc','imkgc']:
    r=json.loads((base/'graph_smoke01'/f'{m}_wk3l_s17_smoke'/'result.json').read_text());assert r['status']=='completed' and r['purpose']=='smoke'
atomic_json(base/'CANDIDATE_FREEZE.json',freeze)
atomic_json(base/'formal_queue/manifest.json',{'stage':'frozen_candidate_confirmation_and_complete_comparators','candidate_freeze_sha256':sha256(base/'CANDIDATE_FREEZE.json'),'jobs':jobs})
with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## Candidate configuration freeze — '+freeze['timestamp']+'\n')
    f.write('Purpose: end single-seed screening, confirm the selected candidate without further search.\n')
    f.write('Decision: DBP-5L/DWY/WK3l shared embeddings; E-PKG independent embeddings because validation macro MRR 0.5478413912912097 exceeds shared 0.537865541242454. DBP independent direction rejected (0.4169560385034036 versus shared 0.6858191508457022). All failures retained.\n')
    f.write('Configuration: '+freeze['rule']+'; seeds 17,29,43. Seed17 checkpoints already exist and are hash-locked; only 29/43 remain.\n')
    f.write('Commands/PIDs/GPUs: 35 jobs in reproduction/sota/formal_queue/manifest.json; eight candidate confirmation jobs followed by 27 graph baseline jobs. GPU assignments and PIDs appended by scheduler before/after each launch.\n')
    f.write('Results: reproduction/sota/CANDIDATE_FREEZE.json, confirmation/, baselines/. Three WK3l validation-only smokes passed; no smoke value enters Table2.\n')
    f.write('Conclusion: configuration frozen on validation evidence, candidate test untouched. Next: confirm, finish comparator reproductions, freeze exact maxima and all checkpoints, then one final candidate acceptance pass.\n')
print(json.dumps({'candidate_frozen':True,'jobs':len(jobs),'confirmation_jobs':8,'baseline_jobs':27}))
