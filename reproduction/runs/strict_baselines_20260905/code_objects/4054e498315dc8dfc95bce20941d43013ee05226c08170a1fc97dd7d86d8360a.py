"""Regenerate baseline-only scientific figures from audited local artifacts."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from export_graph_learning_curves import export_curves

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'
OUT=RUN/'figures'
NAMES={'dbp5l':'DBP-5L','depkg':'E-PKG','dwy':'DWY','wk3l':'WK3l-FR'}
METHODS=['TransE','DistMult','RotatE','ATransN','LSMGA','DMKGC','IMKGC']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def export_profile_metrics(records,ready):
    """Save per-seed strata before the later paired QURA figure is available."""
    folder=RUN/'results/query_profiles'
    if not (folder/'manifest.json').exists():return
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    labels={'head_degree_bin':manifest['degree_bin_labels'],
            'tail_degree_bin':manifest['degree_bin_labels'],
            'relation_frequency_bin':manifest['relation_frequency_bin_labels']}
    profiles={};rows=[]
    for record in records:
        ds,kg,method=record['dataset'],record['kg'],record['method']
        if record['metric']!='mrr' or record.get('condition','full')!='full':continue
        if (method,ds,'full') not in ready:continue
        if int(record['seed']) not in ready[method,ds,'full']:continue
        if (ds,kg) not in profiles:
            with np.load(folder/f'{ds}_{kg}_test.npz') as values:
                profiles[ds,kg]={key:values[key] for key in ['triples','query_index',*labels]}
        profile=profiles[ds,kg]
        with np.load(record['query_artifact']) as queries:
            assert np.array_equal(profile['triples'],queries['triples'])
            assert np.array_equal(profile['query_index'],queries['query_index'])
            key='rank_train' if method=='SS-AGA' else ('rank_all' if ds=='wk3l' else 'rank_train_valid')
            ranks=queries[key].astype(np.float64)
        for axis,names in labels.items():
            for index,label in enumerate(names):
                selected=ranks[profile[axis]==index]
                if len(selected):rows.append({'dataset':ds,'kg':kg,'method':method,'seed':record['seed'],
                    'axis':axis,'stratum':label,'queries':len(selected),'mrr':float((1/selected).mean()),
                    'h1':float((selected==1).mean()),'h10':float((selected<=10).mean()),
                    'query_sha256':record['query_sha256']})
    if not rows:return
    with (OUT/'data/F5_query_strata_per_seed.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row[key] for key in ['dataset','kg','method','axis','stratum'])].append(row)
    means=[]
    for key,values in grouped.items():
        seeds=ready[values[0]['method'],values[0]['dataset'],'full']
        assert sorted(int(row['seed']) for row in values)==seeds
        assert len({row['queries'] for row in values})==1
        result=dict(zip(['dataset','kg','method','axis','stratum'],key));result['queries']=values[0]['queries']
        result.update(n_runs=len(seeds),seeds=';'.join(map(str,seeds)))
        for metric in ['mrr','h1','h10']:
            samples=np.array([row[metric] for row in values]);result[metric+'_mean']=float(samples.mean());result[metric+'_sd']=float(samples.std(ddof=1)) if len(samples)>1 else None
        means.append(result)
    with (OUT/'data/F5_query_strata_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(means[0]));writer.writeheader();writer.writerows(means)


def main():
    OUT.mkdir(exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    training_curves=export_curves(RUN)
    audit=json.loads((RUN/'results/table_fill_audit.json').read_text(encoding='utf-8'))
    records=list(csv.DictReader((RUN/'results/per_kg_metrics.csv').open(encoding='utf-8')))
    published=audit.get('published_groups',audit['three_seed_groups'])
    ready={(g['method'],g['dataset'],g.get('condition','full')):g['seeds'] for g in published}
    export_profile_metrics(records,ready)
    data=defaultdict(list)
    for row in records:
        if row['metric']!='mrr' or row.get('condition','full')!='full' or row['method'] not in METHODS:continue
        if (row['method'],row['dataset'],'full') not in ready:continue
        if int(row['seed']) not in ready[row['method'],row['dataset'],'full']:continue
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
                values=np.array(data[ds,kg,m]);seeds=ready[m,ds,'full'];assert len(values)==len(seeds)
                mean=float(values.mean());sd=float(values.std(ddof=1)) if len(values)>1 else None
                label=f'{mean*100:.2f}†' if sd is None else f'{mean*100:.2f}\n± {sd*100:.2f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if mean<.3 else 'black')
                figure_rows.append({'dataset':ds,'kg':kg,'method':m,'metric':'MRR','mean':mean,'sample_sd':sd,'seeds':';'.join(map(str,seeds)),'n_runs':len(seeds)})
        ax.set(xticks=np.arange(len(methods)),xticklabels=methods,yticks=np.arange(len(kgs)),yticklabels=kgs,
            title=f'{NAMES[ds]} · local baseline reruns',xlabel='Method',ylabel='Target KG')
        ax.tick_params(axis='x',rotation=28);fig.colorbar(artist,ax=ax,label='MRR (%)',shrink=.8)
        has_single=any(len(ready[m,ds,'full'])==1 for m in methods)
        note='† Single run (seed 17); other cells: three-seed mean ± sample SD.\nOnly completed audited results are shown.' if has_single else 'Three-seed mean ± sample SD. Only completed audited groups are shown.'
        fig.text(.01,-.02,note,fontsize=8)
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
        'three_seed_groups':audit['three_seed_groups'],
        'published_groups':published,'single_run_groups':audit.get('single_run_groups',[]),
        'main_table_methods':METHODS,
        'training_curves':training_curves['figures'],
        'training_curve_status':'training/EXPORT_STATUS.json',
        'query_profiles':str(RUN/'results/query_profiles/manifest.json'),
        'strata_data':[p.name for p in (OUT/'data').glob('F5_*.csv')],
        'pending':['F2 budget curves: final QURA-matched policy remains pending','F4 latency ECDF requires full benchmark jobs','F5 paired figures await paired method results; baseline strata are exported now']},indent=2),encoding='utf-8')
    print('Updated scientific plots:',len(list(OUT.glob('*.pdf'))))


if __name__=='__main__':main()
