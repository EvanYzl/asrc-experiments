"""Replace unrun QURA support-table designs with the registered ASRC study."""
import csv
import datetime
import json
import re
import shutil
from frozen_data import ROOT,atomic_json,sha256
phase=ROOT/'reproduction/sota/paper_support';plan=json.loads((phase/'PLAN.json').read_text());tables=ROOT/'outputs/kbs/_main/_tables'
archive=ROOT/plan['old_version_archive'];report=json.loads((phase/'RESULTS.json').read_text()) if (phase/'RESULTS.json').exists() else None
names={'dbp5l':'DBP-5L','depkg':'E-PKG','dwy':'DWY','wk3l':'WK3l-15k'};registered=[]
def cell(table,key,metric,kind='PM',unit='%'):
    ident=f'{table}.{key}.{metric}';registered.append({'cell_id':ident,'table_id':table,'placeholder_kind':kind,'unit':unit,'row_key':key,'metric':metric})
    return '\\'+kind+'{'+ident+'}'
def table(number,caption,columns,header,rows,note):
    text=r'% !TEX root = ../main.tex'+'\n'+r'\begin{table*}[t]'+'\n'+r'\centering\fontsize{8.5}{10.1}\selectfont'+'\n'+r'\setlength{\parskip}{0pt}'+'\n'
    text+='\\caption{'+caption+'}\\label{tab:'+str(number)+'}\n\\begin{tabularx}{\\linewidth}{@{}'+columns+'@{}}\n\\toprule\n'+header+r' \\'+'\n\\midrule\n'
    text+='\n'.join(rows)+'\n\\bottomrule\n\\end{tabularx}\n\\tnote{'+note+'}\n\\end{table*}\n'
    (tables/f'tables/t{number:02}.tex').write_text(text,encoding='utf-8')
    return {'id':number,'file':f't{number:02}.tex','title':caption}
def row(label,values):return label+' & '+' & '.join(values)+r' \\'
titles=[];rows=[]
for ds,name in names.items():
    for variant,label in [('independent','Independent'),('shared','Always shared'),('asrc',r'\textbf{ASRC}')]:
        rows.append(row(name+' & '+label,[cell('T4',ds+'.'+variant,k,unit='pp' if k=='delta' else '%') for k in ['mrr','delta','ntr','ptr','harm']]))
titles.append(table(4,'Paired transfer effects of aligned parameter sharing.',r'L{22mm}L{25mm}*{5}{C}',
    r'Dataset & Representation & MRR & $\Delta$MRR & NTR & PTR & Mean RR harm',rows,
    r'All scores are three-seed mean $\pm$ sample SD; MRR, rates and mean RR harm are in \%, differences in percentage points (pp). The same reciprocal ComplEx/N3 backbone is used throughout. Relative to the seed-matched Independent rank $r_I$, NTR is $\Pr(r>r_I)$, PTR is $\Pr(r<r_I)$, and mean RR harm is $100\mathbb{E}[\max(1/r_I-1/r,0)]$. Metrics are computed per KG and then macro-averaged. ASRC uses the previously frozen validation choice: shared on DBP-5L/DWY/WK3l-15k and independent on E-PKG. ASRC rows reuse their corresponding control results; no new test-based choice is made.'))
rows=[]
for variant,label in [('shared',r'\textbf{ASRC (full shared)}'),('independent','Independent entities and relations'),('entity_only','Entity sharing only'),('relation_only','Relation sharing only'),('no_reciprocal','Without reciprocal augmentation'),('no_n3','Without N3 regularization')]:
    rows.append(row(label,[cell('T5','dbp5l.'+variant,k,unit='pp' if k=='delta' else '%') for k in ['mrr','h1','h10','delta']]+[cell('T5','dbp5l.'+variant,'params_m','V','million')]))
titles.append(table(5,'Component ablations on DBP-5L.',r'L{57mm}*{5}{C}',
    r'Variant & MRR & H@1 & H@10 & $\Delta$MRR & Params (M)',rows,
    r'Metrics are percentages; differences are pp. Three seeds (17, 29, 43); mean $\pm$ sample SD. The full row is reused from Table~\ref{tab:2}; $\Delta$MRR is relative to that row. Rank, optimizer, batch size and validation selection stay fixed. Entity-only and relation-only sharing separate the two parameter-sharing mechanisms. Removing reciprocal augmentation repeats the observed orientation to preserve the number of optimizer steps per epoch; its unused inverse-relation parameters remain allocated. Removing N3 sets its coefficient to zero. Parameter counts include all allocated embedding parameters, including unused inverse rows. All ablations are retained.'))
rows=[]
for ds,name in names.items():
    for group,label in [('aligned','Aligned head'),('unaligned','Unaligned head')]:
        key=ds+'.'+group;rows.append(row(name+' & '+label,[cell('T6',key,'queries','N','count')]+[cell('T6',key,k,unit='pp' if k=='delta' else '%') for k in ['independent_mrr','shared_mrr','delta','ntr']]))
titles.append(table(6,'Transfer by pre-defined alignment coverage of query heads.',r'L{22mm}L{26mm}*{5}{C}',
    r'Dataset & Query group & Queries & Indep. MRR & Shared MRR & $\Delta$MRR & NTR',rows,
    r'MRR and NTR are percentages; differences are pp. A head is aligned if its clean, deterministic alignment component contains entities from multiple KGs. Groups are defined from the supplied alignment inputs, without test-tail information. Each statistic is macro-averaged over nonempty KGs in the group, then reported as three-seed mean $\pm$ sample SD; query counts are pooled counts, not averaging weights. Empty groups are N/A. Both groups and all datasets are retained. Shared versus Independent ranks are paired by the exact test query and seed. These conditional group means need not average to the full-dataset mean.'))
rows=[]
for condition,label in [('clean','Clean inputs'),('train50',r'50\% unique training facts'),('align50',r'50\% alignment links'),('noise10',r'10\% endpoint replacements')]:
    for mode,model_label in [('independent','Independent'),('shared','ASRC (shared)')]:
        rows.append(row(label+' & '+model_label,[cell('T7','dbp5l.'+condition+'.'+mode,k,unit='pp' if k=='delta' else '%') for k in ['mrr','h1','h10','delta']]))
titles.append(table(7,'Robustness under fixed DBP-5L input perturbations.',r'L{45mm}L{27mm}*{4}{C}',
    r'Condition & Representation & MRR & H@1 & H@10 & $\Delta$MRR',rows,
    r'Metrics are percentages; differences are pp. Three training seeds; hyperparameters and shared mode are fixed before perturbation tests. Differences are relative to the same representation on clean inputs. Perturbation seed 20260906 is shared across training seeds. Training reduction selects half of unique triple groups per KG (duplicates kept together); only optimization facts are reduced and the original full-public-train positive filters remain unchanged. Alignment reduction keeps half of unique links per file. Noise replaces the second endpoint of 10\% of links deterministically, then deduplicates and applies the unchanged within-KG collision guard; actual counts are recorded in the input audits. Independent does not consume alignments, so its last two rows reuse its clean results. No claim of formal noise tolerance is made.'))
rows=[]
for ds,name in names.items():
    for variant,label in [('independent','Independent'),('asrc',r'\textbf{ASRC}')]:
        key=ds+'.'+variant
        rows.append(row(name+' & '+label,[cell('T8',key,k,'V' if k=='params_m' else 'PM',unit) for k,unit in [('params_m','million'),('vram_mib','MiB'),('rss_mib','MiB'),('p50_ms','ms'),('p95_ms','ms'),('qps','queries/s')]]))
titles.append(table(8,'Isolated full-candidate ranking cost on one RTX 2080 Ti.',r'L{22mm}L{22mm}*{6}{C}',
    r'Dataset & Model & Params (M) & VRAM (MiB) & RSS (MiB) & p50 (ms) & p95 (ms) & Queries/s',rows,
    r'GPU 0, 11 GB, FP32, four CPU threads, no concurrent GPU work. Each model uses the first 256 frozen validation-selection queries per KG, with three warmup and ten measured passes. Batch 1 supplies latency; batch 256 supplies throughput. Timing includes query tensor construction, all-candidate scores, filtering, gold rank, stable top-10 extraction and transfer of these outputs to CPU; model/data loading and disk output are excluded. p50/p95 and throughput are KG-macro means, then three-seed mean $\pm$ sample SD. VRAM is peak PyTorch-allocated memory; RSS is peak process memory, sampled every 0.1 s; peaks are maximized over KGs within each seed. E-PKG ASRC reuses Independent. These are warmed ranking measurements, not cold-start or training-lifecycle costs.'))
with (archive/'tables/cells_results.csv').open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;old=list(reader)
assert not any(r['value'] for r in old if r['table_id'] in ['T4','T5','T6','T7','T8'])
preserved=[r for r in old if r['table_id'] in ['T1','T2','T3']];rows=list(preserved)
lookup={(t,r['key']):r for t,items in (report['tables'].items() if report else []) for r in items}
values=[line for line in (archive/'tables/values.tex').read_text(encoding='utf-8').splitlines() if not re.match(r'\\SetResult\{T[4-8]\.',line)]
for cell_spec in registered:
    r={k:'' for k in fields};r.update({k:cell_spec[k] for k in ['cell_id','table_id','placeholder_kind','unit']});r['paper_table_cell']=r['cell_id']
    result=lookup.get((r['table_id'],cell_spec['row_key']));key=cell_spec['metric'];value=None;sd=None
    if result:
        if key=='queries':value=result[key]
        elif key in result['stats']:value=result['stats'][key]['mean'];sd=result['stats'][key]['sd']
        elif result.get('queries')==0:value='N/A'
    if value is not None:
        r.update(value=str(value),standard_deviation='' if sd is None or r['placeholder_kind']!='PM' else str(sd),source_type='local rerun' if key!='queries' else 'derived',
            run_id='paper_support/'+cell_spec['row_key'],dataset_hash='sha256:'+sha256(phase/'EVALUATION_FREEZE.json'),split='val_select' if r['table_id']=='T8' else 'test',
            filter_protocol='train+val_select' if r['table_id']=='T8' else 'train+valid (core); all (WK3l)',code_commit='sha256:'+sha256(ROOT/'reproduction/sota/aggregate_support.py'),
            seed='17;29;43',checkpoint='paper_support/EVALUATION_FREEZE.json',selection_protocol='earliest maximum val_select KG-macro MRR; original ASRC mode frozen',
            candidate_scope='all entities in target KG',aggregation=report['aggregation'],feature_policy='published structural facts and supplied alignments; registered T7 input perturbations only',
            notes='Full precision, per-seed values and exact raw hashes in reproduction/sota/paper_support/RESULTS.json; no test-based reselection')
        decimals=1 if key in ['qps','rss_mib','vram_mib'] else 2
        if value=='N/A':shown=r'\NA'
        elif r['placeholder_kind']=='N':shown=str(int(value))
        elif r['placeholder_kind']=='PM':shown='\\('+f'{value:.{decimals}f}'+r'\mathbin{\pm}'+f'{sd:.{decimals}f}'+r'\)'
        else:shown='\\('+f'{value:.{decimals}f}'+r'\)'
        values.append('\\SetResult{'+r['cell_id']+'}{'+shown+'}')
    rows.append(r)
for name in ['cells_results.csv','cells_template.csv']:
    with (tables/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            if name=='cells_template.csv':r={k:(r[k] if k in ['cell_id','table_id','placeholder_kind','unit','paper_table_cell'] else '') for k in fields}
            writer.writerow(r)
(tables/'values.tex').write_text('\n'.join(values)+'\n',encoding='utf-8')
manifest=json.loads((tables/'table_manifest.json').read_text());manifest['tables']=manifest['tables'][:3]+titles
manifest.update(title='ASRC: frozen main comparison and registered three-seed supporting experiments',
    filled_cells=sum(bool(r['value']) for r in rows),placeholder_cells=sum(not bool(r['value']) for r in rows),support_plan='reproduction/sota/paper_support/PLAN.json',
    support_status=report['status'] if report else 'registered_validation_training',support_cells=len(registered),previous_design_archive=plan['old_version_archive'])
atomic_json(tables/'table_manifest.json',manifest)
assert sha256(tables/'tables/t02.tex')==plan['table2_source_sha256']
pp=tables/'preamble.tex';text=pp.read_text(encoding='utf-8');text=re.sub(r'pdftitle=\{[^}]+\}',r'pdftitle={ASRC: Main Comparison and Supporting Experiments}',text);pp.write_text(text,encoding='utf-8')
atomic_json(phase/'TABLE_SCHEMA.json',{'support_cells':registered,'preserved_cells':len(preserved),'total_cells':len(rows),'old_real_cells_preserved':True})
print(json.dumps({'total_cells':len(rows),'support_cells':len(registered),'filled_support':sum(bool(r['value']) for r in rows[len(preserved):]),'table2_unchanged':True}))
