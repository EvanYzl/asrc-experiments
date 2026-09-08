"""Update only ASRC Table2 cells to validated three-seed mean and sample SD."""
import csv
import json
import re
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'reproduction/sota';phase=base/'three_seed';tables=ROOT/'outputs/kbs/_main/_tables'
report=json.loads((phase/'RESULTS.json').read_text());freeze=json.loads((phase/'freeze.json').read_text())
assert report['seeds']==[17,29,43] and report['test_result_files_verified']==12
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);return reader.fieldnames,list(reader)
def write_csv(path,fields,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
fields,rows=read_csv(tables/'cells_results.csv');preserved={r['cell_id']:dict(r) for r in rows if '.asrc.' not in r['cell_id']}
values={};updated=0
for row in rows:
    cid=row['cell_id']
    if not cid.startswith('T2.') or '.asrc.' not in cid:continue
    _,ds,_,metric=cid.split('.');ds={'dbp':'dbp5l','epkg':'depkg'}.get(ds,ds);d=report['datasets'][ds]
    row.update(placeholder_kind='PM',value=str(d['mean'][metric]),standard_deviation=str(d['sample_sd'][metric]),ci_lower='',ci_upper='',
        source_type='rerun',seed='17;29;43',run_id=';'.join(f'asrc_{ds}_s{s}' for s in report['seeds']),
        checkpoint=';'.join(freeze['entries'][f'{ds}_s{s}']['checkpoint'] for s in report['seeds']),
        aggregation='equal KG macro per seed; then mean and sample SD (ddof=1) across seeds17,29,43',
        selection_protocol='fixed val_select macro tail MRR, earliest maximum; checkpoints frozen before test',
        notes='ASRC three-seed confirmation. Seed17 locked test reused; seeds29/43 checkpoints reused and tested once. All raw ranks verified. Unknown/proxy baselines retained; no test-based tuning.')
    values[cid]=r'\('+f"{d['mean'][metric]*100:.2f}"+r'\mathbin{\pm}'+f"{d['sample_sd'][metric]*100:.2f}"+r'\)';updated+=1
assert updated==12 and preserved=={r['cell_id']:r for r in rows if '.asrc.' not in r['cell_id']}
write_csv(tables/'cells_results.csv',fields,rows)
fields,template=read_csv(tables/'cells_template.csv')
for row in template:
    if row['cell_id'] in values:row['placeholder_kind']='PM'
write_csv(tables/'cells_template.csv',fields,template)
vp=tables/'values.tex';text=vp.read_text(encoding='utf-8');changed=[]
def replace_value(m):
    cid=m.group(1)
    if cid not in values:return m.group(0)
    changed.append(cid);return '\\SetResult{'+cid+'}{'+values[cid]+'}'
text=re.sub(r'^\\SetResult\{([^}]+)\}\{(.*)\}$',replace_value,text,flags=re.M);assert set(changed)==set(values)
vp.write_text(text,encoding='utf-8')
tp=tables/'tables/t02.tex';text=tp.read_text(encoding='utf-8')
text=re.sub(r'\\V\{(T2\.[^.]+\.asrc\.[^}]+)\}',r'\\PM{\1}',text)
note=r'''\tnote{ASRC reports mean $\pm$ sample SD over seeds 17, 29 and 43. Each seed uses its fixed validation-selected checkpoint and one test; the previously locked seed-17 tests are reused. Core graph rows marked $\dagger$ remain single runs (seed 17); other existing filled baseline rows retain their three-seed mean $\pm$ sample SD. Full target candidates, tail prediction and ascending entity-ID ties are fixed. Core results use train+valid filtering and equal KG macro averages; WK3l evaluates FR with all filtering. User-supplied approximate references, unknown graph baselines and unmatched baseline repetitions remain, so the comparison is provisional. Pending cells are not treated as defeated.}'''
text=re.sub(r'\\tnote\{.*\}\s*\\end\{table\*\}',lambda _:note+'\n\\end{table*}',text,flags=re.S);tp.write_text(text,encoding='utf-8')
pp=tables/'preamble.tex';ps=pp.read_text(encoding='utf-8').replace('ASRC: Single-seed Table 2 and Preserved Experimental Tables','ASRC: Three-seed Table 2 and Preserved Experimental Tables');pp.write_text(ps,encoding='utf-8')
labels={'dbp5l':'DBP-5L','depkg':'E-PKG','dwy':'DWY','wk3l':'WK3l-15k'}
lines=['# ASRC 三种子结果','',f"{report['verdict']}：{report['strict_wins']}/12 项严格超过当前参考线。基线未知项、暂代值和重复次数差异仍保留，不宣称完整可比 SOTA。",'',
    '种子固定为 17、29、43。先对每个种子的 KG 指标等权宏平均，再对三个种子求均值及样本标准差（ddof=1）。单位为 %。','',
    '| 数据集 | MRR | H@1 | H@10 |','|---|---:|---:|---:|']
for ds,d in report['datasets'].items():lines.append('| '+labels[ds]+' | '+' | '.join(f"{d['mean'][k]*100:.2f} ± {d['sample_sd'][k]*100:.2f}" for k in ['mrr','h1','h10'])+' |')
lines+=['','## 逐种子结果','', '| 数据集 | 种子 | MRR | H@1 | H@10 | 最佳 epoch |','|---|---:|---:|---:|---:|---:|']
for ds,d in report['datasets'].items():
    for seed,s in d['seeds'].items():lines.append('| '+labels[ds]+' | '+seed+' | '+' | '.join(f"{s['macro'][k]*100:.6f}" for k in ['mrr','h1','h10'])+' | '+str(s['best_epoch'])+' |')
lines+=['','## 执行与证据','',
    '此次没有训练新模型。八个 29/43 检查点均来自已完成的验证训练，与原先测试前冻结的检查点哈希一致。冻结同一算法、超参数和选模规则后，各补一次测试；种子17的四套已锁定测试直接复用，未重测。',
    '共核验12份测试结果、45份各KG逐查询排名，以及数据/对齐/代码/检查点哈希。三种过滤排名、查询顺序和精确宏平均均已重算；全候选尾预测、固定过滤与实体ID同分规则沿用原协议。',
    '原始数据位于 reproduction/sota/three_seed/evaluation/，种子17原始数据保留在 reproduction/sota/single_seed/。冻结清单为 three_seed/freeze.json；完整精度的均值、样本标准差和逐种子结果在 three_seed/RESULTS.json；逐列比较在 three_seed/COMPARISON.csv。',
    'LaTeX 主文件：outputs/kbs/_main/_tables/main.tex。表格源：tables/t02.tex。数据：values.tex 与 cells_results.csv。编译产物：KBS_Main_Text_Tables.pdf。修改前版本保存在 reproduction/sota/history/before_three_seed/tables/。', '']
report_text='\n'.join(lines);(phase/'REPORT.zh.md').write_text(report_text,encoding='utf-8');(tables/'THREE_SEED_REPORT.zh.md').write_text(report_text,encoding='utf-8')
design='''# Table 2：ASRC 三种子实验设计与完成记录

本次按用户新指令将我们的方法从单种子补齐为 17、29、43 三种子。前一阶段的单种子文档保存在 reproduction/sota/history/before_three_seed/tables/；不再将其停止范围作为本次补种子的限制。

算法及超参数保持冻结：互逆 ComplEx、全候选交叉熵、N3=0.01、复数秩256、Adagrad学习率0.1、批量512、最多40 epoch、每5 epoch验证。DBP-5L/DWY/WK3l共享对齐实体，E-PKG采用独立表示；这些选择均来自先前验证集。所有种子统一依据固定 val_select 宏 MRR 选最优检查点，同分保留最早检查点。

不重新训练、不修改方案、不补基线。种子17测试已经锁定并直接复用；29/43已有训练结果与先前测试前的冻结记录一致，验证来源及哈希核验后各执行一次测试。八项独立评测按空闲显存分配至六张GPU；前后记录命令、PID/GPU、种子、输出和结论。

全部数据划分、输入、候选全集、尾预测和同分排序沿用原协议。核心三套数据使用train+valid过滤、分别对5/6/3个KG等权宏平均；WK3l评测FR并用all过滤。每个种子先取KG宏平均，再计算三种子均值和样本标准差（ddof=1），Table2不填写虚构的置信区间。原始查询与排名保存，可独立重算。

比较仍保留75个既有外部单元、用户暂代参考和未知图基线。部分核心图方法只有单种子，WK3l三种图方法未知；补齐ASRC三种子不等同于补齐全部基线。仅按保留的参考线报告暂定比较，不宣称完整可比SOTA或统计显著优势。

完成八项测试、原始结果核验、Table2填入均值±样本标准差、主LaTeX编译和双端同步后停止。全部数值、逐种子记录和证据路径见 THREE_SEED_REPORT.zh.md。
'''
(tables/'EXPERIMENT_DESIGN.zh.md').write_text(design,encoding='utf-8')
mp=tables/'table_manifest.json';manifest=json.loads(mp.read_text());manifest.update(title='ASRC: three-seed Table 2 and preserved experimental tables',
    candidate_seed=None,candidate_seeds=[17,29,43],candidate_status=report['verdict'],candidate_raw_audit='THREE_SEED_RESULTS.json',
    candidate_aggregation='three-seed mean and sample SD; ddof=1',task_scope='Complete only ASRC three seeds and compile Table2; existing baselines and other tables unchanged')
mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
shutil.copy2(phase/'RESULTS.json',tables/'THREE_SEED_RESULTS.json');shutil.copy2(phase/'COMPARISON.csv',tables/'THREE_SEED_COMPARISON.csv')
print(json.dumps({'asrc_cells_updated':12,'other_cells_unchanged':len(preserved),'seeds':[17,29,43],'format':'mean ± sample SD'},ensure_ascii=False))
