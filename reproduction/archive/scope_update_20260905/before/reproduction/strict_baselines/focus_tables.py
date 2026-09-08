"""Limit the live queue to the user's selected tables and keep other jobs deferred."""
import argparse
import csv
import datetime
import json
from pathlib import Path

from table_scope import ROOT, RUN, cell_scope


def prioritize(tables=None):
    path=RUN/'manifest.json';manifest=json.loads(path.read_text(encoding='utf-8'))
    active_tables=tables or manifest.get('active_tables',['T1','T2','T3'])
    assert active_tables and set(active_tables)<=set(['T1','T2','T3','T4','T7','T8'])
    all_jobs={j['id']:j for j in [*manifest['jobs'],*manifest.get('deferred_jobs',[])]}
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    archive=RUN/'manifest_history';archive.mkdir(exist_ok=True)
    (archive/f'before_table_focus_{stamp}.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    required=set()
    with (ROOT/'outputs/kbs_main_tables/cells_template.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['table_id'] in active_tables:
                required.update(cell_scope(row['cell_id'])['jobs'])
    while True:
        expanded=required|{dep for name in required for dep in all_jobs[name].get('depends_on',[])}
        if expanded==required:break
        required=expanded
    state=json.loads((RUN/'queue_state.json').read_text(encoding='utf-8'))
    current=state.get('current_job')
    if current and state['jobs'].get(current,{}).get('status')=='running':
        assert current in required,'A deferred job is already running; preserve its checkpoint before changing phase'
    # Prioritize corrected KEnS reruns and complete three-seed baseline groups.
    def priority(job):
        name=job['id']
        if name.startswith('kens_v2_'):return (0,name)
        if name.startswith('smoke_'):return (1,name)
        if name.startswith(('transe_','distmult_','rotate_','atransn_')):return (2,name)
        if name.startswith('alignkgc_'):return (3,name)
        if name.startswith(('internal_base_','internal_teacher_','internal_controls_')):
            # Complete each dataset's shared backbone and active teacher/control
            # stages before starting another dataset. Dependencies order stages.
            _,stage,dataset,*rest=name.split('_')
            order={'base':0,'teacher':1,'controls':2}[stage]
            return (4,f'{dataset}_{order}_'+('_'.join(rest)))
        if name.startswith('kens_'):return (5,name)
        if name.startswith('ssaga_'):return (6,name)
        return (7,name)
    manifest['jobs']=sorted([j for name,j in all_jobs.items() if name in required],key=priority)
    manifest['deferred_jobs']=[j for name,j in all_jobs.items() if name not in required]
    for job in manifest['jobs']:
        if job['id'].startswith('internal_controls_'):
            job['command']=[arg for arg in job['command'] if arg!='--skip-robustness']
            if 'T7' not in active_tables:job['command'].append('--skip-robustness')
    manifest['focus']='tables_'+('_'.join(t[1:] for t in active_tables))+'_without_qura'
    manifest['active_tables']=active_tables
    manifest['focus_set_at']=datetime.datetime.now().astimezone().isoformat()
    manifest['focus_instruction']='Fill only non-QURA entries of '+','.join(active_tables)+'. Leave QURA rows/effects empty. Other tables remain unfilled.'
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)
    summary={'active_jobs':len(manifest['jobs']),'active_formal_jobs':sum(not j['id'].startswith('smoke_') for j in manifest['jobs']),
             'deferred_jobs':len(manifest['deferred_jobs']),'focus':manifest['focus']}
    print(json.dumps(summary,ensure_ascii=False))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tables',nargs='+');args=parser.parse_args()
    prioritize(['T'+v.removeprefix('T') for v in args.tables] if args.tables else None)
