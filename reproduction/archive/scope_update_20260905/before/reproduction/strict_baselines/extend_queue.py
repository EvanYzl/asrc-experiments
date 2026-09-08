"""Append the local method-specific reruns requested for Table 3."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SUITE=ROOT/'reproduction/strict_baselines'
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'
AI='C:/Users/evan/.conda/envs/ai4/python.exe'
KENS=str(ROOT/'reproduction/envs/kens-tf210/Scripts/python.exe')


def extend():
    path=RUN/'manifest.json';manifest=json.loads(path.read_text(encoding='utf-8'))
    jobs=list({j['id']:j for j in [*manifest['jobs'],*manifest.get('deferred_jobs',[])]}.values())
    manifest['jobs']=jobs;manifest.pop('deferred_jobs',None);ids={j['id'] for j in jobs}
    def add(ident,script,argv,smoke=False,depends=(),python=AI,env=None):
        if ident in ids:return
        output=RUN/('smoke' if smoke else 'jobs')/ident
        jobs.append({'id':ident,'cwd':str(SUITE),'command':[python,str(SUITE/script),*argv,'--output',str(output)],
                     'result':str(output/'result.json'),'depends_on':list(depends),'env':env or {}})
        ids.add(ident)
    teacher=RUN/'jobs/atransn_teacher_s17/en'
    add('smoke_atransn_wk3l','run_atransn.py',['--seed','17','--teacher',str(teacher),'--smoke'],True,['atransn_teacher_s17'])
    for j in jobs:
        if j['id'].startswith('atransn_wk3l_'):
            j['depends_on']=list(dict.fromkeys(j.get('depends_on',[])+['smoke_atransn_wk3l']))
    env={'PYTHONPATH':'','CUDA_VISIBLE_DEVICES':'-1','TF_CPP_MIN_LOG_LEVEL':'2','TF_NUM_INTRAOP_THREADS':'4','TF_NUM_INTEROP_THREADS':'2'}
    add('smoke_kens_dbp5l','run_kens.py',['--dataset','dbp5l','--seed','17','--smoke'],True,python=KENS,env=env)
    add('smoke_kens_depkg','run_kens.py',['--dataset','depkg','--seed','17','--smoke'],True,python=KENS,env=env)
    for ds in ['dbp5l','depkg']:
        for seed in [17,29,43]:
            add(f'kens_{ds}_s{seed}','run_kens.py',['--dataset',ds,'--seed',str(seed)],depends=[f'smoke_kens_{ds}'],python=KENS,env=env)
    for job in jobs:
        if job['id'].startswith('kens_depkg_s'):job['depends_on']=['smoke_kens_depkg']
    for ds in ['dbp5l','depkg']:
        smoke_id=f'smoke_kens_v2_{ds}'
        add(smoke_id,'run_kens.py',['--dataset',ds,'--seed','17','--smoke'],True,python=KENS,env=env)
        for seed in [17,29,43]:
            add(f'kens_v2_{ds}_s{seed}','run_kens.py',['--dataset',ds,'--seed',str(seed)],depends=[smoke_id],python=KENS,env=env)
    for job in jobs:
        if job['id'].startswith('kens_v2_'):
            job['supersedes']=job['id'].replace('kens_v2_','kens_',1)
        if job['id'].startswith('smoke_kens_v2_'):
            job['supersedes']=job['id'].replace('smoke_kens_v2_','smoke_kens_',1)
    for short,script in [('alignkgc','run_alignkgc.py'),('ssaga','run_ssaga.py')]:
        for ds in ['dbp5l','depkg']:
            smoke_id=f'smoke_{short}_{ds}'
            add(smoke_id,script,['--dataset',ds,'--seed','17','--smoke'],True)
            for seed in [17,29,43]:
                add(f'{short}_{ds}_s{seed}',script,['--dataset',ds,'--seed',str(seed)],depends=[smoke_id])
    add('smoke_alignkgc_depkg_r2','run_alignkgc.py',['--dataset','depkg','--seed','17','--smoke'],True)
    for job in jobs:
        if job['id'].startswith('alignkgc_depkg_s'):job['depends_on']=['smoke_alignkgc_depkg_r2']
        if job['id']=='smoke_alignkgc_depkg_r2':job['supersedes']='smoke_alignkgc_depkg'
    add('smoke_internal_base','run_internal_train.py',['--stage','base','--dataset','dbp5l','--seed','17','--smoke'],True)
    add('smoke_internal_teacher','run_internal_train.py',['--stage','teacher','--dataset','dbp5l','--seed','17','--smoke',
        '--base-root',str(RUN/'smoke/smoke_internal_base')],True,['smoke_internal_base'])
    add('smoke_internal_controls','run_internal_controls.py',['--dataset','dbp5l','--seed','17','--smoke',
        '--teacher-root',str(RUN/'smoke/smoke_internal_teacher')],True,['smoke_internal_teacher'])
    add('smoke_internal_benchmark','benchmark_internal.py',['--dataset','dbp5l','--seed','17','--smoke',
        '--teacher-root',str(RUN/'smoke/smoke_internal_teacher')],True,['smoke_internal_teacher'])
    for method,datasets in [('DMKGC',['depkg','dwy']),('LSMGA',['depkg','dwy']),('IMKGC',['depkg'])]:
        for ds in datasets:
            ident=f'smoke_{method.lower()}_{ds}'
            add(ident,'run_graph_baseline.py',['--method',method,'--dataset',ds,'--seed','17','--smoke'],True)
            for job in jobs:
                if job['id'].startswith(f'{method.lower()}_{ds}_s'):
                    job['depends_on']=list(dict.fromkeys(job.get('depends_on',[])+[ident]))
    add('smoke_lsmga_dwy_r2','run_graph_baseline.py',['--method','LSMGA','--dataset','dwy','--seed','17','--smoke'],True)
    for job in jobs:
        if job['id']=='smoke_lsmga_dwy_r2':job['supersedes']='smoke_lsmga_dwy'
        if job['id'].startswith('lsmga_dwy_s'):
            job['depends_on']=[d for d in job.get('depends_on',[]) if d!='smoke_lsmga_dwy']
            job['depends_on']=list(dict.fromkeys(job['depends_on']+['smoke_lsmga_dwy_r2']))
    for ds in ['wk3l','dbp5l','depkg','dwy']:
        for seed in [17,29,43]:
            base_id=f'internal_base_{ds}_s{seed}';teacher_id=f'internal_teacher_{ds}_s{seed}'
            base_root=RUN/'jobs'/base_id
            add(base_id,'run_internal_train.py',['--stage','base','--dataset',ds,'--seed',str(seed)],depends=['smoke_internal_base'])
            add(teacher_id,'run_internal_train.py',['--stage','teacher','--dataset',ds,'--seed',str(seed),'--base-root',str(base_root)],depends=[base_id,'smoke_internal_teacher'])
            add(f'internal_controls_{ds}_s{seed}','run_internal_controls.py',['--dataset',ds,'--seed',str(seed),
                '--teacher-root',str(RUN/'jobs'/teacher_id)],depends=[teacher_id,'smoke_internal_controls'])
            add(f'internal_benchmark_{ds}_s{seed}','benchmark_internal.py',['--dataset',ds,'--seed',str(seed),
                '--teacher-root',str(RUN/'jobs'/teacher_id)],depends=[teacher_id,'smoke_internal_benchmark'])
            if ds=='dbp5l':
                target20=f'internal_base_dbp5l_target20_s{seed}'
                add(target20,'run_internal_train.py',['--stage','base','--dataset',ds,'--seed',str(seed),'--condition','target20'],depends=['smoke_internal_base'])
                for condition in ['target20','align20']:
                    args=['--stage','teacher','--dataset',ds,'--seed',str(seed),'--base-root',str(base_root),'--condition',condition]
                    deps=[base_id,'smoke_internal_teacher']
                    if condition=='target20':args+=['--target-base-root',str(RUN/'jobs'/target20)];deps.append(target20)
                    add(f'internal_teacher_dbp5l_{condition}_s{seed}','run_internal_train.py',args,depends=deps)
                    add(f'internal_controls_dbp5l_{condition}_s{seed}','run_internal_controls.py',['--dataset',ds,'--seed',str(seed),
                        '--teacher-root',str(RUN/'jobs'/f'internal_teacher_dbp5l_{condition}_s{seed}')],
                        depends=[f'internal_teacher_dbp5l_{condition}_s{seed}','smoke_internal_controls'])
    def priority(j):
        name=j['id']
        if name.startswith('smoke_internal'):return -1
        if name.startswith('smoke_'):return 0
        if any(name.startswith(m+'_wk3l') for m in ['transe','distmult','rotate']):return 1
        if name.startswith('atransn'):return 2
        if name.startswith(('kens','alignkgc','internal_')):return 3
        return 4
    jobs.sort(key=priority)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
    print('Queue jobs:',len(jobs))
    if manifest.get('focus','').startswith('tables_'):
        from focus_tables import prioritize
        prioritize()


if __name__=='__main__':extend()
