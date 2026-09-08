"""Freeze all current Table 2 comparison cells, preserving unknowns and provenance."""
import csv
import datetime
import json
from pathlib import Path
from frozen_data import ROOT,sha256,atomic_json
source=ROOT/'outputs/kbs_main_tables/cells_results.csv'
rows=list(csv.DictReader(source.open(encoding='utf-8-sig')))
provisional={'T2.dbp.imkgc.h10':.7888,'T2.epkg.dmkgc.mrr':.4640,'T2.epkg.dmkgc.h1':.3303,'T2.epkg.dmkgc.h10':.6955,'T2.dwy.dmkgc.h10':.6497}
registry={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'frozen_provisional_comparison','source_csv_sha256':sha256(source),
          'scope_correction':'User directed reuse of local baselines; no new server baseline experiments. All comparisons retained.',
          'cells':{},'column_maxima':{},'unknown_cells':[],'provisional_cells':[],'repetition_mismatch_cells':[],
          'acceptance':{'required_strict_wins':10,'total_metrics':12,'max_other_deficit':.005},
          'full_comparable_sota_eligible':False}
for row in rows:
    cid=row['cell_id']
    if not cid.startswith('T2.') or '.qura.' in cid:continue
    item=dict(row);value=None
    if row['value']:
        assert row['source_type']=='rerun'
        value=float(row['value']);item['status']='local_reproduction';item['seeds']=[int(s) for s in row['seed'].split(';')]
        ds={'dbp':'dbp5l','epkg':'depkg','dwy':'dwy','wk3l':'wk3l'}[cid.split('.')[1]];method=cid.split('.')[2]
        artifacts=[];metrics=[]
        for seed in item['seeds']:
            rp=ROOT/'reproduction/runs/strict_baselines_20260905/jobs'/f'{method}_{ds}_s{seed}'/'result.json'
            assert rp.exists(),rp
            r=json.loads(rp.read_text(encoding='utf-8'))
            assert r['status']=='completed' and r['full_data'] is True and r.get('purpose','formal')=='formal'
            assert r['protocol']=='kbs-baselines-v1-20260905' and r['seed']==seed and r['dataset']==ds
            m=r.get('macro')
            if m is None:
                # Existing WK3l ATransN has a dataset-level macro too; fail closed otherwise.
                raise AssertionError(f'Missing macro in {rp}')
            metrics.append(m[cid.split('.')[-1]])
            artifacts.append({'path':rp.relative_to(ROOT).as_posix(),'sha256':sha256(rp)})
        exact=sum(metrics)/len(metrics)
        assert abs(exact-value)<1e-14,(cid,exact,value)
        item['value']=value;item['raw_result_artifacts']=artifacts
        if cid in provisional:item['replaced_user_provisional']=provisional[cid]
        if item['seeds']!=[17,29,43]:registry['repetition_mismatch_cells'].append(cid)
    elif cid in provisional:
        value=provisional[cid];item.update(value=value,status='user_provisional_reference',source_type='user_provided_approximation',seeds=[])
        registry['provisional_cells'].append(cid)
    else:
        item.update(value=None,status='unknown',seeds=[]);registry['unknown_cells'].append(cid)
    registry['cells'][cid]=item
    if value is not None:
        column='.'.join([cid.split('.')[1],cid.split('.')[-1]])
        if column not in registry['column_maxima'] or value>registry['column_maxima'][column]['value']:
            registry['column_maxima'][column]={'value':value,'cell_id':cid,'status':item['status'],'seeds':item['seeds']}
assert len(registry['cells'])==75 and len(registry['column_maxima'])==12
out=ROOT/'reproduction/sota/PROVISIONAL_COMPARATORS_FREEZE.json'
assert not out.exists(),'Frozen comparison registry already exists; do not alter after test'
atomic_json(out,registry)
print(json.dumps({'frozen_columns':registry['column_maxima'],'unknown':len(registry['unknown_cells']),'provisional':len(registry['provisional_cells']),'repetition_mismatches':len(registry['repetition_mismatch_cells'])},ensure_ascii=False))
