"""Export validation diagnostics from the selected formal graph-model runs."""
import csv
import datetime
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN=Path(__file__).resolve().parents[2]/'reproduction/runs/strict_baselines_20260905'
METRICS=('mrr','h1','h3','h10')


def export_curves(run=RUN):
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    state=json.loads((run/'queue_state.json').read_text(encoding='utf-8'))
    folder=run/'figures/training';folder.mkdir(parents=True,exist_ok=True)
    data_folder=run/'figures/data';data_folder.mkdir(parents=True,exist_ok=True)
    rows=[];groups=defaultdict(list);sources=[]
    for job_id in manifest.get('method_batch',{}).get('job_ids',[]):
        job=run/'jobs'/job_id;curve=job/'learning_curve.jsonl';config_path=job/'config.json'
        if not curve.exists() or not config_path.exists():continue
        config_bytes=config_path.read_bytes();config=json.loads(config_bytes)
        if config.get('purpose')!='formal':continue
        raw=curve.read_bytes()
        # An active writer may be midway through appending its last JSON line.
        complete=raw[:raw.rfind(b'\n')+1]
        records=[json.loads(line) for line in complete.splitlines() if line.strip()]
        if not records:continue
        method,dataset,seed=(config[key] for key in ('method','dataset','seed'))
        job_status=state.get('jobs',{}).get(job_id,{}).get('status','unknown')
        prefix_hash=hashlib.sha256(complete).hexdigest()
        latest={}
        for record_index,record in enumerate(records,1):
            evaluation=int(record['evaluation_index'])
            assert set(record['per_kg'])==set(config['domain_order'])
            per_kg={kg:value['metrics']['select'] for kg,value in record['per_kg'].items()}
            macro=record['val_select_macro']
            for metric in METRICS:
                assert np.isclose(np.mean([v[metric] for v in per_kg.values()]),macro[metric],rtol=0,atol=1e-12)
            for kg,values in [('__equal_kg_macro__',macro),*per_kg.items()]:
                assert all(np.isfinite(values[k]) and 0<=values[k]<=1 for k in METRICS)
                rows.append({'job_id':job_id,'method':method,'dataset':dataset,'seed':seed,
                    'record_index':record_index,'evaluation_index':evaluation,'kg':kg,
                    'validation_queries':sum(v['n'] for v in per_kg.values()) if kg=='__equal_kg_macro__' else values['n'],
                    **{k:values[k] for k in METRICS},'run_status':job_status,
                    'source_prefix_bytes':len(complete),'source_prefix_sha256':prefix_hash,
                    'config_sha256':hashlib.sha256(config_bytes).hexdigest()})
            latest[evaluation]=macro['mrr']
        sources.append({'job_id':job_id,'source':str(curve),'records':len(records),
            'evaluations':len(latest),'replayed_records':len(records)-len(latest),
            'last_evaluation':max(latest),'planned_rounds':config['rounds'],
            'run_status':job_status,'prefix_bytes':len(complete),'prefix_sha256':prefix_hash,
            'incomplete_trailing_bytes':len(raw)-len(complete)})
        groups[method,dataset].append({'seed':seed,'status':job_status,'values':latest,'rounds':config['rounds']})
    if rows:
        with (data_folder/'training_validation_per_seed.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figures=[]
    for (method,dataset),runs in sorted(groups.items()):
        with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42}):
            fig,ax=plt.subplots(figsize=(6.4,3.5))
            maximum=0
            for item in sorted(runs,key=lambda v:v['seed']):
                x=sorted(item['values']);y=np.array([item['values'][i] for i in x])*100
                maximum=max(maximum,float(y.max()))
                ax.plot(x,y,label=f"Seed {item['seed']} ({item['status']})",linewidth=1.5)
                ax.scatter(x[-1],y[-1],s=18)
            ax.set(xlabel='Validation evaluation',ylabel='val_select MRR (%)',
                title=f'{method} / {dataset} · equal-KG validation macro',
                xlim=(0,max(item['rounds'] for item in runs)),ylim=(0,min(100,max(40,maximum*1.15))))
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
            ax.legend(fontsize=8,loc='best')
            fig.text(.02,.015,'Observed validation history; each line is one seed. Replayed indices use the latest record.',fontsize=7)
            fig.tight_layout(rect=(0,.045,1,1))
            name=f'{method.lower()}_{dataset}_validation_mrr'
            for suffix in ('png','pdf'):
                target=folder/f'{name}.{suffix}';fig.savefig(target,dpi=180,bbox_inches='tight')
                figures.append(str(target.relative_to(run/'figures')))
            plt.close(fig)
    receipt={'generated_at':datetime.datetime.now().astimezone().isoformat(),
        'role':'validation training diagnostic','eligible_for_table':False,
        'csv':'data/training_validation_per_seed.csv' if rows else None,'rows':len(rows),
        'figures':figures,'sources':sources,
        'replay_policy':'CSV retains every record; plots use the latest value at a replayed evaluation index.'}
    (folder/'EXPORT_STATUS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return receipt


if __name__=='__main__':
    result=export_curves()
    print(json.dumps({'validation_rows':result['rows'],'runs':len(result['sources']),'figures':result['figures']}))
