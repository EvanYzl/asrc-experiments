"""Fill the user's screenshot values from matching, full-precision local results."""
import csv
import datetime
import json
import re
import shutil
from pathlib import Path
import numpy as np
from frozen_data import ROOT,atomic_json,sha256
base=ROOT/'reproduction/sota';tables=ROOT/'outputs/kbs/_main/_tables';source=ROOT/'outputs/kbs_main_tables/cells_results.csv'
stamp=datetime.datetime.now(datetime.timezone.utc);tag=stamp.strftime('%Y%m%d_%H%M%S')
revision=base/'paper_support/document_versions'/('table2_screenshot_'+tag);revision.mkdir(parents=True)
for name in ['values.tex','cells_results.csv','cells_template.csv','table_manifest.json','validation_report.json','KBS_Main_Text_Tables.pdf']:
    shutil.copy2(tables/name,revision/name)
image=Path('C:/Users/evan/AppData/Local/Temp/codex-clipboard-a8a96c34-c463-40ac-8a60-db67871dea7e.png')
shutil.copy2(image,revision/'user_screenshot.png')
def csvread(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);return reader.fieldnames,list(reader)
fields,rows=csvread(tables/'cells_results.csv');_,sources=csvread(source);lookup={r['cell_id']:r for r in rows};local={r['cell_id']:r for r in sources}
means={
 'transe':[35.77,23.20,59.25,44.01,30.63,68.38,31.86,16.59,61.86],
 'distmult':[31.10,22.56,47.58,18.05,11.91,30.15,44.22,33.02,63.67],
 'rotate':[36.88,25.86,57.34,39.09,28.35,60.12,40.46,27.46,63.33],
 'lsmga':[45.47,27.77,77.18,45.13,32.27,67.00,22.79,8.87,51.15],
 'dmkgc':[50.47,37.31,76.16,42.52,28.65,67.31,28.79,15.29,55.46],
 'imkgc':[None,None,None,None,None,None,31.48,15.53,63.31]}
sds={'transe':[.07,.15,.05,.27,.38,.15,.15,.24,.15],
 'distmult':[.15,.17,.43,.76,.81,.68,.13,.06,.26],
 'rotate':[.11,.25,.22,.21,.24,.16,.17,.08,.49]}
checked=[];changed=[]
for method,values in means.items():
    for ix,val in enumerate(values):
        cid=f"T2.{['dbp','epkg','dwy'][ix//3]}.{method}.{['mrr','h1','h10'][ix%3]}"
        if val is None:assert not lookup[cid]['value'];continue
        r=local[cid];assert r['unit']=='fraction displayed as percent' and f"{100*float(r['value']):.2f}"==f'{val:.2f}',cid
        if method in sds:assert f"{100*float(r['standard_deviation']):.2f}"==f'{sds[method][ix]:.2f}',cid
        checked.append(cid)
        if lookup[cid]['value']:assert lookup[cid]==r,cid
        else:changed.append(cid);lookup[cid]=r.copy()
assert len(checked)==48 and set(changed)=={f'T2.dwy.dmkgc.{k}' for k in ['mrr','h1','h10']}
job=ROOT/'reproduction/runs/strict_baselines_20260905/jobs/dmkgc_dwy_s17';result=json.loads((job/'result.json').read_text())
assert result['status']=='completed' and result['purpose']=='formal' and result['dataset']=='dwy' and result['seed']==17
source_files={};computed=[]
for kg in ['db','wk','yg']:
    path=job/kg/'test_queries.npz';z=np.load(path,allow_pickle=False);r=z['rank_train_valid'].astype(float)
    assert result['per_kg'][kg]['primary_filter']=='train_valid' and len(r)==result['per_kg'][kg]['count'] and (r>=1).all()
    metrics={'mrr':float(np.mean(1/r)),'h1':float(np.mean(r==1)),'h10':float(np.mean(r<=10))};computed.append(metrics)
    for k,v in metrics.items():assert abs(v-result['per_kg'][kg]['metrics']['train_valid'][k])<1e-14
    source_files[path.relative_to(ROOT).as_posix()]=sha256(path)
for k in ['mrr','h1','h10']:
    value=float(np.mean([m[k] for m in computed]));assert value==result['macro'][k] and value==float(local[f'T2.dwy.dmkgc.{k}']['value'])
source_files[(job/'result.json').relative_to(ROOT).as_posix()]=sha256(job/'result.json')
with (tables/'cells_results.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(lookup[r['cell_id']] for r in rows)
vp=tables/'values.tex';text=vp.read_text(encoding='utf-8')
for cid in changed:
    assert '\\SetResult{'+cid+'}' not in text;text+='\\SetResult{'+cid+'}{\\('+f"{100*float(lookup[cid]['value']):.2f}"+'\\)}\n'
vp.write_text(text,encoding='utf-8')
approval=base/'paper_support/APPROVED_BASELINE_UPDATES.json';approved=json.loads(approval.read_text()) if approval.exists() else {'cells':{},'revisions':[]}
approved['cells'].update({cid:lookup[cid] for cid in changed});approved['revisions'].append(revision.relative_to(ROOT).as_posix());atomic_json(approval,approved)
manifest=json.loads((tables/'table_manifest.json').read_text());manifest.update(filled_cells=sum(bool(r['value']) for r in lookup.values()),placeholder_cells=sum(not r['value'] for r in lookup.values()),
    task_scope='ASRC and supporting experiments complete; user-authorized document-only Table2 baseline import; no new experiment or candidate selection',
    latest_baseline_revision=revision.relative_to(ROOT).as_posix())
group={'method':'DMKGC','dataset':'dwy','condition':'full','seeds':[17]}
if group not in manifest['single_run_groups']:manifest['single_run_groups'].append(group)
atomic_json(tables/'table_manifest.json',manifest)
audit={'timestamp':stamp.isoformat(),'purpose':'User requested filling the attached screenshot numbers','screenshot_sha256':sha256(image),
    'checked_visible_numeric_cells':checked,'newly_filled_cells':changed,'retained_unknown_cells':[f'T2.{ds}.imkgc.{k}' for ds in ['dbp','epkg'] for k in ['mrr','h1','h10']],
    'source_csv':source.relative_to(ROOT).as_posix(),'source_hashes':source_files,'raw_ranks_recomputed':True,'raw_queries':sum(result['per_kg'][kg]['count'] for kg in result['per_kg']),
    'macro':result['macro'],'unrounded_values_preserved':True,'new_experiments':0,'archive':revision.relative_to(ROOT).as_posix()}
atomic_json(revision/'IMPORT_AUDIT.json',audit);atomic_json(base/'paper_support/LATEST_BASELINE_IMPORT.json',audit)
with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## User-authorized Table2 screenshot import — '+stamp.isoformat()+'\nPurpose: fill supplied baseline values in the current paper. All 48 visible numeric cells match existing full-precision local results; only DMKGC/DWY MRR/H1/H10 were missing.\nChanges: imported 28.79/15.29/55.46 from completed dmkgc_dwy_s17, verified against 32,486 saved raw ranks across three KGs; retained seed17-only metadata and all unknown cells. ASRC and Tables4–8 unchanged.\nCommand: python reproduction/sota/import_table2_screenshot.py; PID/GPU: document-only, no experiment launched. Archive/result: '+audit['archive']+'/IMPORT_AUDIT.json.\nConclusion: local reproduction provenance verified; no screenshot rounding overwrote raw precision. Next: compile main.tex and verify PDF plus approved cell changes. Historical frozen SOTA references remain historical; this edit does not perform a new SOTA acceptance.\n')
print(json.dumps({'checked':len(checked),'filled':changed,'archive':audit['archive']}))
