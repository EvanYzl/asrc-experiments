"""Pre-register bounded ASRC supporting experiments and preserve the previous paper."""
import datetime
import json
import shutil
from pathlib import Path
from frozen_data import ROOT,atomic_json,sha256

base=ROOT/'reproduction/sota';phase=base/'paper_support';phase.mkdir(exist_ok=True)
assert not (phase/'PLAN.json').exists(),'Existing support campaign must be resumed, not replaced.'
stamp=datetime.datetime.now(datetime.timezone.utc);tag=stamp.strftime('%Y%m%d_%H%M%S')
archive=base/'history'/f'before_paper_support_{tag}';archive.mkdir(parents=True)
tables=ROOT/'outputs/kbs/_main/_tables';idea=ROOT/'work/ideaspark/_run/multidomain-kgc-local/_2/phase4'
for folder,label in [(tables,'tables'),(idea,'idea')]:
    dest=archive/label;dest.mkdir()
    for path in folder.iterdir():
        if path.is_file() and path.suffix in {'.md','.tex','.pdf','.json','.csv','.py','.ps1'}:shutil.copy2(path,dest/path.name)
    if label=='tables':shutil.copytree(folder/'tables',dest/'tables')
for path in [base/'TASK_SCOPE.json',base/'PLAN.md',ROOT/'refine-logs/EXPERIMENT_PLAN.md',ROOT/'refine-logs/EXPERIMENT_TRACKER.md']:
    if path.exists():shutil.copy2(path,archive/path.name)
seeds=[17,29,43];models={};wave1=[];main=json.loads((base/'three_seed/freeze.json').read_text())
base_recipe={'rank':256,'reg':.01,'lr':.1,'epochs':40,'batch_size':512,'valid_every':5,'patience':5,
             'reciprocal':True,'train_keep':1.,'alignment_keep':1.,'alignment_noise':0.,'perturbation_seed':20260906}
def add(ds,variant,seed,mode,**changes):
    ident=f'{ds}_{variant}_s{seed}';recipe=dict(base_recipe,dataset=ds,mode=mode,seed=seed,**changes)
    recipe.update(changes)
    run=phase/'training'/ident;test=phase/'evaluation'/ident;reused_training=False;reused_test=False;existing=None
    if variant in ['independent','shared']:
        selected='independent' if ds=='depkg' else 'shared'
        if mode==selected:
            existing=main['entries'][f'{ds}_s{seed}'];run=(ROOT/existing['checkpoint']).parent;test=(ROOT/existing['evaluation_result']).parent;reused_training=True;reused_test=True
        elif seed==17 and ((ds=='dbp5l' and mode=='independent') or (ds=='depkg' and mode=='shared')):
            run=base/'pilot02'/f'{ds}_{mode}_r256_n3p01_s17';reused_training=True
    entry={'id':ident,'dataset':ds,'variant':variant,'seed':seed,'recipe':recipe,
        'training_path':run.relative_to(ROOT).as_posix(),'evaluation_path':test.relative_to(ROOT).as_posix(),
        'reuse_training':reused_training,'reuse_locked_test':reused_test,'trainer':'original' if variant in ['independent','shared'] else 'support'}
    if reused_training:
        result=json.loads((run/'result.json').read_text());entry['reused_training_result_sha256']=sha256(run/'result.json');entry['checkpoint_sha256']=result['checkpoint_sha256']
    if reused_test:entry['locked_result_sha256']=sha256(test/'result.json')
    models[ident]=entry
    if variant in ['independent','shared'] and not reused_training:
        cmd=['/root/zhishitupui/.envs/kgc/bin/python','reproduction/sota/train_complex.py','--dataset',ds,'--sharing',mode,'--seed',str(seed),'--output',entry['training_path']]
        wave1.append({'id':ident,'purpose':'Matched ComplEx control for Tables4/5/6/8: isolate parameter sharing without changing the backbone',
            'changes':'Fixed original recipe; validation-only training; no Table2 reselection','seed':seed,'output':entry['training_path'],'command':cmd,'min_free_mib':9500,
            'next':'Freeze val_select-best checkpoint; retain every outcome; test only after the support evaluation freeze.'})
for ds in ['dbp5l','depkg','dwy','wk3l']:
    for mode in ['independent','shared']:
        for seed in seeds:add(ds,mode,seed,mode)
for seed in seeds:
    add('dbp5l','entity_only',seed,'entity_only')
    add('dbp5l','relation_only',seed,'relation_only')
    add('dbp5l','no_reciprocal',seed,'shared',reciprocal=False)
    add('dbp5l','no_n3',seed,'shared',reg=0.)
    add('dbp5l','train50_independent',seed,'independent',train_keep=.5)
    add('dbp5l','train50_shared',seed,'shared',train_keep=.5)
    add('dbp5l','align50_shared',seed,'shared',alignment_keep=.5)
    add('dbp5l','noise10_shared',seed,'shared',alignment_noise=.1)
plan={'timestamp':stamp.isoformat(),'authorization':'User requested starting experiments for other tables and improving idea and their design.',
    'method':'ASRC','seeds':seeds,'models':models,'new_training_runs':sum(not e['reuse_training'] for e in models.values()),
    'new_test_runs':sum(not e['reuse_locked_test'] for e in models.values()),'reused_training_runs':sum(e['reuse_training'] for e in models.values()),
    'table2_frozen':True,'table2_source_sha256':sha256(tables/'tables/t02.tex'),'table2_results_sha256':sha256(base/'three_seed/RESULTS.json'),
    'old_version_archive':archive.relative_to(ROOT).as_posix(),'external_baseline_runs':0,
    'hypotheses':{'C1':'Aligned parameter sharing yields measurable query-level gains and harms beyond the fixed reciprocal ComplEx/N3 backbone; validation chooses the representation mode.',
                  'C2':'Training-time sharing can change model size and measured full-candidate ranking cost without a query-time source gate.'},
    'tables':{'T4':'Four-dataset independent/shared/frozen-ASRC paired transfer analysis','T5':'DBP component ablations: entity sharing, relation sharing, reciprocal augmentation and N3',
              'T6':'Clean aligned-head versus unaligned-head strata; paired independent/shared results','T7':'DBP fixed train50, align50 and noise10 perturbations with matched independent controls',
              'T8':'Isolated RTX2080Ti filtered-rank and top10 evaluation benchmark; fixed validation queries; no concurrent GPU jobs'},
    'statistics':{'scalar_results':'equal KG macro, then seed mean and sample SD ddof=1','ntr':'Pr(rank_variant > rank_independent)',
                  'ptr':'Pr(rank_variant < rank_independent)','harm':'mean(max(1/rank_independent - 1/rank_variant,0))',
                  'strata':'head belongs to clean deterministic cross-KG alignment component; mean across nonempty KGs, retain all groups',
                  'low_resource_filter':'Original full public train+valid/test-filter protocol unchanged; only optimizer training facts are reduced.',
                  'missing_group':'N/A, never zero-imputed'},
    'freeze_policy':'No main-model tuning or repeated test selection. All 48 recipes are registered before new tests; all ablation and negative results retained.',
    'profiling':{'gpu':0,'queries_per_kg':256,'warmup_passes':3,'measured_passes':10,'batch_latency':1,'batch_throughput':256,
                 'query_split':'first 256 frozen val_select rows per KG','operation':'CPU query tensor construction, all-candidate scoring, filtering, gold rank and stable top10; exclude model/data loading and disk output',
                 'aggregate':'KG-macro per-seed p50/p95/QPS, then three-seed mean and sample SD; peaks max across KG within seed'},
    'budget_estimate':'34 short trainings plus 36 once-only tests; approximately 2-4 GPU-hours including checks/profiling, estimate not a cutoff.',
    'stop_rule':'After these registered support tables, idea/design and raw artifacts are updated and verified, stop. Do not optimize Table2 or add search directions.'}
assert len(models)==48 and plan['new_training_runs']==34 and len(wave1)==10
atomic_json(phase/'PLAN.json',plan);atomic_json(phase/'wave1/manifest.json',{'jobs':wave1})
atomic_json(base/'TASK_SCOPE.json',{'phase':'paper_support','status':'registered_training','seeds':seeds,'table2_locked':True,
    'authorization':plan['authorization'],'plan':'reproduction/sota/paper_support/PLAN.json','further_experiments_authorized':True})
with (base/'PLAN.md').open('r',encoding='utf-8') as f:old=f.read()
(base/'PLAN.md').write_text('# Current phase: ASRC supporting tables\n\nUser authorized Tables4–8 and idea refinement. Execute only the registered 48 configurations (14 reused trainings, 34 new), reuse 12 locked tests, and run 36 new once-only tests after checkpoint freeze. No Table2 retuning. See paper_support/PLAN.json and refine-logs/EXPERIMENT_PLAN.md. Earlier phases are historical below.\n\n'+old,encoding='utf-8')
with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## New authorized phase: supporting tables — '+stamp.isoformat()+'\nPurpose: update the ASRC idea and replace incompatible unrun QURA table designs with testable support experiments.\n')
    f.write('Registered changes: T4 paired sharing controls; T5 four component deletions; T6 fixed aligned-head strata; T7 three DBP perturbations; T8 isolated full-candidate evaluation cost. Table2 and every existing comparison remain locked.\n')
    f.write('Configuration/seeds: original rank256/N3.01/Adagrad.1/batch512/40 epochs/val every5/patience5, seeds17/29/43; changed factors explicit in paper_support/PLAN.json. 34 new trainings, 14 reused; 36 new one-test evaluations, 12 locked tests reused.\n')
    f.write('Command/PID/GPU: wave1/manifest.json contains ten original-trainer matched controls; six-GPU queue logs actual launches. New variant source must pass validation-only checks before wave2.\n')
    f.write('Outputs/archive: paper_support/; '+plan['old_version_archive']+'. Conclusion: design frozen before any new test; no test-guided main-method reselection. Next: validation-only training, freeze checkpoints, evaluate once, profile in isolation, update tables/idea and sync.\n')
print(json.dumps({'models':48,'new_training':34,'reused_training':14,'new_tests':36,'reused_tests':12,'wave1':len(wave1),'archive':plan['old_version_archive']}))
