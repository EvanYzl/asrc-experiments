"""Regenerate baseline-only scientific figures from audited local artifacts."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'
OUT=RUN/'figures'
NAMES={'dbp5l':'DBP-5L','depkg':'E-PKG','dwy':'DWY','wk3l':'WK3l-FR'}
METHODS=['TransE','DistMult','RotatE','ATransN','LSMGA','DMKGC','IMKGC','Target-only','Uniform-All']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    audit=json.loads((RUN/'results/table_fill_audit.json').read_text(encoding='utf-8'))
    records=list(csv.DictReader((RUN/'results/per_kg_metrics.csv').open(encoding='utf-8')))
    ready={(g['method'],g['dataset'],g.get('condition','full')) for g in audit['three_seed_groups']}
    data=defaultdict(list)
    for row in records:
        if row['metric']!='mrr' or row.get('condition','full')!='full' or row['method'] not in METHODS:continue
        if (row['method'],row['dataset'],'full') not in ready:continue
        data[row['dataset'],row['kg'],row['method']].append(float(row['value']))
    figure_rows=[]
    for ds in NAMES:
        keys=[k for k in data if k[0]==ds]
        if not keys:continue
        kgs=sorted({k[1] for k in keys});methods=[m for m in METHODS if any(k[2]==m for k in keys)]
        fig,ax=plt.subplots(figsize=(max(5,len(methods)*.8),max(2.5,len(kgs)*.7)))
        means=np.array([[np.mean(data[ds,kg,m]) if (ds,kg,m) in data else np.nan for m in methods] for kg in kgs])
        artist=ax.imshow(means*100,cmap='viridis',vmin=0,vmax=max(.6,float(np.nanmax(means)))*100,aspect='auto')
        for i,kg in enumerate(kgs):
            for j,m in enumerate(methods):
                if (ds,kg,m) not in data:continue
                values=np.array(data[ds,kg,m]);assert len(values)==3
                mean=float(values.mean());sd=float(values.std(ddof=1))
                ax.text(j,i,f'{mean*100:.2f}\n± {sd*100:.2f}',ha='center',va='center',fontsize=8,color='white' if mean<.3 else 'black')
                figure_rows.append({'dataset':ds,'kg':kg,'method':m,'metric':'MRR','mean':mean,'sample_sd':sd,'seeds':'17;29;43'})
        ax.set(xticks=np.arange(len(methods)),xticklabels=methods,yticks=np.arange(len(kgs)),yticklabels=kgs,
            title=f'{NAMES[ds]} · local baseline reruns',xlabel='Method',ylabel='Target KG')
        ax.tick_params(axis='x',rotation=28);fig.colorbar(artist,ax=ax,label='MRR (%)',shrink=.8)
        fig.text(.01,-.02,'Three-seed mean ± sample SD. Only completed audited groups are shown.',fontsize=8)
        fig.tight_layout();save(fig,f'F1_{ds}_per_kg_mrr')
    if figure_rows:
        with (OUT/'data/F1_per_kg_mrr.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(figure_rows[0]));w.writeheader();w.writerows(figure_rows)
    paired=defaultdict(list)
    robustness=defaultdict(list)
    for row in audit['accepted_runs']:
        group=(row['method'],row['dataset'],row.get('condition','full'))
        if group not in ready or not row.get('paired_macro'):continue
        if group[2]=='full':paired[row['dataset'],row['method']].append(row['paired_macro'])
        if row['dataset']=='dbp5l' and row['method']=='Uniform-All' and (group[2]=='full' or group[2].startswith('corrupt')):
            corruption=0 if group[2]=='full' else int(group[2].replace('corrupt',''))
            robustness[corruption].append(row['paired_macro'])
    if paired:
        fig,ax=plt.subplots(figsize=(6,4))
        for (ds,method),items in paired.items():
            x=np.array([v['ntr_all']*100 for v in items]);y=np.array([v['mrr']*100 for v in items])
            ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),fmt='o',capsize=3,label=f'{NAMES[ds]} / {method}')
        ax.set(xlabel='Negative transfer rate (%)',ylabel='MRR (%)',title='Completion and negative transfer');ax.legend(fontsize=7)
        fig.tight_layout();save(fig,'F2_accuracy_negative_transfer')
    if len(robustness)>1:
        x=sorted(robustness);fig,axes=plt.subplots(1,2,figsize=(7,3))
        for ax,metric,label in zip(axes,['mrr','ntr_all'],['MRR (%)','Negative transfer rate (%)']):
            y=[np.mean([v[metric] for v in robustness[c]])*100 for c in x]
            sd=[np.std([v[metric] for v in robustness[c]],ddof=1)*100 for c in x]
            ax.errorbar(x,y,yerr=sd,fmt='o-',capsize=3);ax.set(xlabel='Corrupted alignment (%)',ylabel=label)
        fig.suptitle('DBP-5L · frozen Uniform-All checkpoint');fig.tight_layout();save(fig,'F3_alignment_corruption')
    (OUT/'FIGURE_STATUS.json').write_text(json.dumps({'figures':[p.name for p in OUT.glob('*.pdf')],
        'source_audit':str(RUN/'results/table_fill_audit.json'),'proposed_method_included':False,
        'pending':['F2 budget curves: final QURA-matched policy remains pending','F4 latency ECDF requires full benchmark jobs','F5 relation/degree paired analyses use saved query artifacts']},indent=2),encoding='utf-8')
    print('Updated scientific plots:',len(list(OUT.glob('*.pdf'))))


if __name__=='__main__':main()
