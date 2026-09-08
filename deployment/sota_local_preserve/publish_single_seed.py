"""Fill only the new Table2 ASRC row and publish a raw-rank verified single-seed report."""
import csv
import datetime
import json
import re
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'reproduction/sota';tables=ROOT/'outputs/kbs/_main/_tables'
r=json.loads((base/'SINGLE_SEED_ACCEPTANCE.json').read_text(encoding='utf-8'))
assert r['verified_datasets']==4 and r['seed']==17
refs=json.loads((base/'SINGLE_SEED_REFERENCES.json').read_text())
source=ROOT/'outputs/kbs_main_tables'
# Freeze has already captured comparator values. Preserve all external cells exactly as frozen.
with (tables/'cells_results.csv').open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f);fieldnames=reader.fieldnames;rows=list(reader)
for row in rows:
    cid=row['cell_id']
    if cid.startswith('T2.') and '.qura.' not in cid and '.asrc.' not in cid:
        old=refs['original_comparison_cells'][cid]
        # User proxy reference must not overwrite a method's pending or actual result cell.
        if old['status']=='local_reproduction':
            for field in fieldnames:
                if field in old and field not in ['value']:row[field]=str(old[field]) if old[field] is not None else ''
            row['value']=str(old['value'])
        else:
            row['value']='';row['standard_deviation']='';row['source_type']='pending'
    if cid.startswith('T2.') and any(x in cid for x in ['.qura.','.asrc.']):
        cid=cid.replace('.qura.','.asrc.');row['cell_id']=cid;parts=cid.split('.')
        ds={'dbp':'dbp5l','epkg':'depkg'}.get(parts[1],parts[1]);metric=parts[3];d=r['datasets'][ds]
        row.update(placeholder_kind='V',value=str(d['macro'][metric]),standard_deviation='',ci_lower='',ci_upper='',source_type='rerun',
            run_id=f'asrc_{ds}_single_s17',split='test',seed='17',filter_protocol='all' if ds=='wk3l' else 'train+valid',
            checkpoint=f'reproduction/sota/single_seed/{ds}/freeze.json',selection_protocol='val_select macro tail MRR; earliest tie; frozen before one test',
            candidate_scope='all target entities; tail prediction; ascending entity-ID ties',aggregation='single seed 17; equal KG macro',
            feature_policy='structure-only; original train triples and supplied alignments; no held-out edges or external text',
            paper_table_cell=cid,notes='ASRC replaces the unimplemented QURA-Cert candidate for Table2. Single-seed provisional acceptance; no SD; raw ranks and frozen provenance verified.')
        row['dataset_hash']=';'.join(x['manifest_sha256'] for x in d['raw_artifacts'])
        row['code_commit']='sha256:'+json.loads((base/'single_seed'/ds/'freeze.json').read_text())['source_hashes']['reproduction/sota/train_complex.py']
with (tables/'cells_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
template=tables/'cells_template.csv'
with template.open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f);fields=reader.fieldnames;trows=list(reader)
for row in trows:
    if row['cell_id'].startswith('T2.') and any(x in row['cell_id'] for x in ['.qura.','.asrc.']):
        row['cell_id']=row['cell_id'].replace('.qura.','.asrc.');row['placeholder_kind']='V'
        if 'paper_table_cell' in row:row['paper_table_cell']=row['cell_id']
with template.open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(trows)
value_lines=['% From verified raw results; user reference lines are explicitly separate.']
for row in rows:
    if row['value']=='':continue
    cid=row['cell_id'];value=float(row['value']);kind=row['placeholder_kind']
    if cid.startswith('T1.'):formatted=f'{value:,.0f}'
    elif kind=='PM' and row['standard_deviation']:formatted=f"{value*100:.2f}\\,\\(\\pm\\)\\,{float(row['standard_deviation'])*100:.2f}"
    elif row['unit']=='percent' or cid.startswith(('T2.','T3.')):formatted=f'{value*100:.2f}'
    else:formatted=f'{value:.2f}'
    value_lines.append('\\SetResult{'+cid+'}{'+formatted+'}')
(tables/'values.tex').write_text('\n'.join(value_lines)+'\n',encoding='utf-8')
table=tables/'tables/t02.tex';s=table.read_text(encoding='utf-8')
s=re.sub(r'\\textbf\{(?:QURA-Cert|ASRC \(candidate\)|ASRC)\}',r'\\textbf{ASRC}',s)
s=s.replace('.qura.','.asrc.');s=re.sub(r'\\PM\{(T2\.[^.]+\.asrc\.[^}]+)\}',r'\\V{\1}',s)
note=r'''\tnote{ASRC reports one run (seed 17), selected only on the fixed validation subset and tested once per dataset; no SD is inferred. Core graph rows marked $\dagger$ also have one run; other existing filled baseline rows retain their three-seed mean $\pm$ sample SD. Full target candidates, tail prediction and ascending entity-ID ties are fixed. Core results use train+valid filtering and equal KG macro averages; WK3l evaluates FR with all filtering. Acceptance is provisional: user-supplied approximate references and unknown graph baselines remain, and repetitions are not uniformly matched. Pending cells are not treated as defeated. Raw references and comparison differences are in the accompanying single-seed report.}'''
s=re.sub(r'\\tnote\{.*\}\s*\\end\{table\*\}',lambda _:note+'\n\\end{table*}',s,flags=re.S)
table.write_text(s,encoding='utf-8')
preamble=tables/'preamble.tex';ps=preamble.read_text(encoding='utf-8').replace('QURA-Cert: Main-text Experimental Tables','ASRC: Single-seed Table 2 and Preserved Experimental Tables');preamble.write_text(ps,encoding='utf-8')
(tables/'Table_2_SOTA_single_seed.tex').write_text(r'''\documentclass[a4paper,10pt]{article}
\usepackage[margin=16mm]{geometry}
\input{preamble.tex}
\begin{document}
\setcounter{table}{1}
\input{tables/t02.tex}
\end{document}
''',encoding='utf-8')
labels={'dbp5l':'DBP-5L','depkg':'E-PKG','dwy':'DWY','wk3l':'WK3l-15k'}
lines=['# 单种子 SOTA 暂定验收','',f"**结论：{r['verdict']}。** 固定种子 17；{r['strict_wins']}/12 项严格超过参考，总体标准仍为至少 10/12，其余落后不超过 0.5 个百分点。",'',
       '| 数据集 | MRR (%) | H@1 (%) | H@10 (%) | 超过参考 |','|---|---:|---:|---:|---:|']
for ds,d in r['datasets'].items():
    m=d['macro'];wins=sum(x['strict_win'] for x in d['comparisons']);lines.append(f"| {labels[ds]} | {m['mrr']*100:.2f} | {m['h1']*100:.2f} | {m['h10']*100:.2f} | {wins}/3 |")
lines+=['','## 冻结参考与差值','', '| 数据集 | 指标 | 未四舍五入参考 (%) | 提升 (百分点) | 来源 |','|---|---|---:|---:|---|']
for ds,d in r['datasets'].items():
    for item in d['comparisons']:
        lines.append(f"| {labels[ds]} | {item['metric']} | {item['reference']*100:.10f} | {item['difference_percentage_points']:+.4f} | {'用户暂代，不是复现' if item['reference_status']=='user_provisional_reference' else '已有本地复现原值'} |")
lines+=['','## 方法、范围与证据','',
        'ASRC 使用同一互逆 ComplEx/N3 算法，依据各数据集 seed17 验证集选择是否共享对齐实体。DBP-5L、DWY、WK3l 为 shared，E-PKG 为 independent。秩256、N3=0.01、Adagrad lr=0.1、batch512、最多40 epoch，每5 epoch 验证，取验证宏 MRR 最佳检查点。复用已完成初筛，不重复训练。',
        '依序完成 DBP-5L → E-PKG → DWY → WK3l 的单次测试，每套先冻结后测试、测试后锁定再迁移；没有依据测试重新选模或训练。保留原划分、实体 ID、输入和全候选尾预测；核心 train+valid 过滤，WK3l all 过滤，同分按实体 ID 升序。',
        '原始查询与顺序、三种过滤排名、宏平均、代码/数据/检查点/冻结参考哈希均已核验。可审计文件：SINGLE_SEED_ACCEPTANCE.json、SINGLE_SEED_COMPARISON.csv、SINGLE_SEED_REFERENCES.json，以及 single_seed/<dataset>/ 下的 freeze.json、LOCKED_RESULT.json 和 evaluation/*/test_queries.npz。',
        '星号参考为用户暂代，不计作本地复现。WK3l 的 LSMGA、DMKGC、IMKGC 等未完成比较项仍未知；外部基线重复次数不完全一致。因此结论仅为“单种子达到暂定门槛”，不宣称完整可比 SOTA 或统计显著优势。全部75个既有外部比较单元保留。',
        '旧目标下完成的其他种子验证和三项 WK3l 小预算适配检查已保留为历史；本阶段不使用它们，不新增基线、种子或论文实验。','',
        '## 保留方向与待办','',
        '- 最佳：当前已锁定的 ASRC 单种子结果，不再改动。',
        '- 失败/淘汰：原128维 TransE 等权教师在 DBP/DWY 较弱；DBP independent 验证劣于 shared；E-PKG shared 验证劣于 independent；初次 queue.py 命名冲突在训练前失败，日志保留。',
        '- 待办仅记录：未知基线、暂代参考、重复次数一致性和统计确认；没有授权新阶段前不执行。','']
report='\n'.join(lines)
(tables/'SOTA_SINGLE_SEED_REPORT.zh.md').write_text(report,encoding='utf-8');(base/'SOTA_SINGLE_SEED_REPORT.zh.md').write_text(report,encoding='utf-8')
for name in ['SINGLE_SEED_ACCEPTANCE.json','SINGLE_SEED_COMPARISON.csv','SINGLE_SEED_REFERENCES.json']:
    shutil.copy2(base/name,tables/name)
manifest_path=tables/'table_manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
filled=sum(row['value']!='' for row in rows)
manifest.update(title='ASRC: single-seed Table 2 and preserved experimental tables',date='2026-09-06',
    filled_cells=filled,placeholder_cells=len(rows)-filled,total_cells=len(rows),
    candidate_method='ASRC',candidate_seed=17,candidate_status=r['verdict'],
    candidate_raw_audit='SINGLE_SEED_ACCEPTANCE.json',
    task_scope='Table2 only; other tables preserved; no extra baseline or seed experiments')
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'updated_asrc_cells':12,'existing_comparison_cells_preserved':75,'verdict':r['verdict']},ensure_ascii=False))
