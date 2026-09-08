"""Export a standalone, seed-17 language/KG table from saved ranks; no model execution."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('G:/zhishitupui')
OUT = Path(__file__).resolve().parent
if __name__ == '__main__' and (OUT/'language_kg_seed17.json').exists():
    current = json.loads((OUT/'language_kg_seed17.json').read_text(encoding='utf-8'))
    if current.get('completion_batch') or current.get('partial_completion_batch'):
        import runpy
        runpy.run_path(str(ROOT/'reproduction/language_table_seed17/render_completed.py'), run_name='__main__')
        raise SystemExit(0)
SEED = 17
PROTOCOL = 'kbs-baselines-v1-20260905'
SPECS = {
    'dbp5l': ('DBP-5L', [('el', 'EL'), ('en', 'EN'), ('es', 'ES'), ('fr', 'FR'), ('ja', 'JA')]),
    'depkg': ('E-PKG', [('de', 'DE'), ('es', 'ES'), ('fr', 'FR'), ('it', 'IT'), ('jp', 'JP'), ('uk', 'UK')]),
    'dwy': ('DWY', [('db', 'DBpedia'), ('wk', 'Wikidata'), ('yg', 'YAGO')]),
    'wk3l': ('WK3l-15k', [('en_f', r'EN\_F (support)'), ('fr', 'FR (target)')]),
}
BASELINES = [('transe', 'TransE'), ('distmult', 'DistMult'), ('rotate', 'RotatE'),
             ('lsmga', 'LSMGA'), ('dmkgc', 'DMKGC'), ('imkgc', 'IMKGC')]
METRICS = ('h1', 'h10', 'mrr')
raw_sources = {}
query_reference = {}
records = []
method_rows = {}
max_metric_error = 0.0


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path, audit=True):
    data = path.read_bytes()
    if audit:
        raw_sources[path.relative_to(ROOT).as_posix()] = {'sha256': sha(data), 'bytes': len(data)}
    return json.loads(data.decode('utf-8-sig'))


def close(a, b, label):
    global max_metric_error
    error = abs(float(a) - float(b))
    max_metric_error = max(max_metric_error, error)
    assert error < 1e-12, (label, a, b)


main_results = read_json(ROOT/'reproduction/sota/three_seed/RESULTS.json')
queue = read_json(ROOT/'reproduction/runs/strict_baselines_20260905/queue_state.json', audit=False)


def extract(ds, key, label, source, accepted=None):
    expected = [kg for kg, _ in SPECS[ds][1] if kg != 'en_f']
    row = {'dataset': ds, 'method_key': key, 'method': label, 'seed': SEED,
           'source': source.relative_to(ROOT).as_posix(), 'per_kg': {}}
    if not source.exists():
        state = queue['jobs'].get(f'{key}_{ds}_s17', {})
        row['status'] = 'pending' if state.get('status') == 'running' else 'unavailable'
        for kg in expected + ['AVG']:
            records.append({'dataset': ds, 'method': label, 'seed': SEED, 'kg': kg,
                            'status': row['status'], 'n': '', 'primary_filter': '',
                            'h1': '', 'h10': '', 'mrr': '', 'source_result': '', 'source_rank': ''})
        return row
    d = read_json(source)
    assert d['status'] == 'completed' and d['seed'] == SEED and d['dataset'] == ds
    assert d['protocol'] == PROTOCOL
    if key in {x[0] for x in BASELINES} | {'atransn'}:
        assert d['method'].lower() == label.lower()
        queue_item = queue['jobs'][f'{key}_{ds}_s17']
        assert queue_item['status'] == 'completed'
        assert queue_item.get('exit_code') == 0 or (
            queue_item.get('exit_code') is None
            and queue_item.get('completion_evidence') == 'result artifact after adopted process exit'
            and queue_item.get('finished') is not None
        ), (key, ds, queue_item)
        row['queue_completion_receipt'] = {
            k: queue_item.get(k) for k in ['status', 'exit_code', 'finished', 'completion_evidence']}
    group = d.get('per_kg') or {kg: item['test'] for kg, item in d['domains'].items()}
    assert set(group) == set(expected), (source, list(group), expected)
    row['status'] = 'completed'
    row['checkpoint_sha256'] = d.get('checkpoint_sha256') or {
        kg: item['checkpoint_sha256'] for kg, item in d['domains'].items()}
    source_copy = OUT/'sources'/f'{ds}_{key}_s17_result.json'
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    source_copy.write_bytes(source.read_bytes())
    for kg in expected:
        filt = group[kg]['primary_filter']
        assert filt == ('all' if ds == 'wk3l' else 'train_valid')
        rank_path = source.parent/kg/'test_queries.npz'
        rank_bytes = rank_path.read_bytes()
        raw_sources[rank_path.relative_to(ROOT).as_posix()] = {'sha256': sha(rank_bytes), 'bytes': len(rank_bytes)}
        with np.load(rank_path, allow_pickle=False) as raw:
            rank = raw['rank_' + filt].astype(np.float64)
            triples, index = raw['triples'], raw['query_index']
            assert rank.ndim == 1 and len(rank) == group[kg]['count'] and np.all(rank >= 1)
            assert np.all(np.isfinite(rank)) and triples.shape == (len(rank), 3)
            assert np.array_equal(index, np.arange(len(rank)))
            query_hash = sha(triples.tobytes())
            query_reference.setdefault((ds, kg), (len(rank), query_hash))
            assert query_reference[(ds, kg)] == (len(rank), query_hash), (ds, kg, label)
            values = {'h1': float(np.mean(rank <= 1)), 'h10': float(np.mean(rank <= 10)),
                      'mrr': float(np.mean(1.0/rank))}
        for metric in METRICS:
            close(values[metric], group[kg]['metrics'][filt][metric], f'{ds}/{key}/{kg}/{metric}')
            if accepted is not None:
                close(values[metric], accepted['per_kg'][kg][metric], f'accepted/{ds}/{key}/{kg}/{metric}')
        row['per_kg'][kg] = {'n': len(rank), **values}
        records.append({'dataset': ds, 'method': label, 'seed': SEED, 'kg': kg, 'status': 'completed',
                        'n': len(rank), 'primary_filter': filt, **values,
                        'source_result': row['source'], 'source_rank': rank_path.relative_to(ROOT).as_posix()})
    row['macro'] = {metric: float(np.mean([row['per_kg'][kg][metric] for kg in expected])) for metric in METRICS}
    for metric in METRICS:
        close(row['macro'][metric], d['macro'][metric], f'macro/{ds}/{key}/{metric}')
        if accepted is not None:
            close(row['macro'][metric], accepted['macro'][metric], f'accepted/macro/{ds}/{key}/{metric}')
    if accepted is not None:
        assert row['checkpoint_sha256'] == accepted['checkpoint_sha256']
    records.append({'dataset': ds, 'method': label, 'seed': SEED, 'kg': 'AVG', 'status': 'completed',
                    'n': sum(row['per_kg'][kg]['n'] for kg in expected), 'primary_filter': filt,
                    **row['macro'], 'source_result': row['source'], 'source_rank': ''})
    return row


for ds in SPECS:
    order = BASELINES.copy()
    if ds == 'wk3l':
        order.insert(3, ('atransn', 'ATransN'))
    rows = []
    for key, label in order:
        rows.append(extract(ds, key, label,
                    ROOT/f'reproduction/runs/strict_baselines_20260905/jobs/{key}_{ds}_s17/result.json'))
    asrc = main_results['datasets'][ds]['seeds']['17']
    assert sha((ROOT/asrc['result_path']).read_bytes()) == asrc['result_sha256']
    rows.append(extract(ds, 'asrc', 'ASRC', ROOT/asrc['result_path'], asrc))
    method_rows[ds] = rows

supplement_path = OUT/'en_f_evaluation/RESULTS.json'
supplement = read_json(supplement_path) if supplement_path.exists() else None
if supplement is not None:
    assert supplement['status'] == 'completed' and supplement['seed'] == SEED
    assert supplement['original_fr_unchanged'] and supplement['original_avg_remains_fr_only']
    SPECS['wk3l'] = ('WK3l-15k', [('en_f', r'EN\_F (supplemental)'), ('fr', 'FR (primary)')])

    def add_english(row, result_path, rank_path, test):
        raw_sources[result_path.relative_to(ROOT).as_posix()] = {'sha256': sha(result_path.read_bytes()), 'bytes': result_path.stat().st_size}
        raw_sources[rank_path.relative_to(ROOT).as_posix()] = {'sha256': sha(rank_path.read_bytes()), 'bytes': rank_path.stat().st_size}
        with np.load(rank_path, allow_pickle=False) as z:
            rank = z['rank_all'].astype(np.float64)
            assert np.array_equal(z['query_index'], np.arange(len(rank)))
            identity = (len(rank), sha(z['triples'].tobytes()))
            query_reference.setdefault(('wk3l', 'en_f'), identity)
            assert query_reference[('wk3l', 'en_f')] == identity
            values = {'h1': float(np.mean(rank <= 1)), 'h10': float(np.mean(rank <= 10)), 'mrr': float(np.mean(1/rank))}
        assert len(rank) == 40700 and test['primary_filter'] == 'all'
        for metric in METRICS:
            close(values[metric], test['metrics']['all'][metric], 'EN_F/'+row['method']+'/'+metric)
        row['supplemental_kg_results'] = {'en_f': {'n': len(rank), **values}}
        records.append({'dataset': 'wk3l', 'method': row['method'], 'seed': SEED, 'kg': 'en_f', 'status': 'completed',
                        'n': len(rank), 'primary_filter': 'all', **values,
                        'source_result': result_path.relative_to(ROOT).as_posix(), 'source_rank': rank_path.relative_to(ROOT).as_posix()})

    asrc_row = next(row for row in method_rows['wk3l'] if row['method_key'] == 'asrc')
    assert asrc_row['checkpoint_sha256'] == supplement['asrc']['checkpoint_sha256']
    add_english(asrc_row, OUT/'en_f_evaluation/asrc_en_f_s17/result.json',
                OUT/'en_f_evaluation/asrc_en_f_s17/test_queries.npz', supplement['asrc']['test'])
    teacher = supplement['teacher']
    teacher_row = {'dataset': 'wk3l', 'method_key': 'en_teacher', 'method': 'TransE (EN teacher)',
                   'seed': SEED, 'status': 'completed', 'role': 'english_teacher_only', 'per_kg': {}, 'macro': None,
                   'source': teacher['source_result'], 'checkpoint_sha256': teacher['checkpoint_sha256']}
    add_english(teacher_row, ROOT/teacher['source_result'], ROOT/teacher['source_rank'], teacher['test'])
    method_rows['wk3l'].insert(-1, teacher_row)

timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
with (OUT/'language_kg_seed17.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
(OUT/'language_kg_seed17.json').write_text(json.dumps({
    'timestamp': timestamp, 'seed': SEED, 'units': 'fractions; PDF multiplies by 100',
    'protocol': PROTOCOL, 'rows': method_rows,
}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def column_value(row, kg, metric):
    if row['status'] != 'completed':
        return None
    if kg == 'AVG':
        values = row.get('macro')
    elif kg == 'en_f':
        values = row.get('supplemental_kg_results', {}).get(kg)
    else:
        values = row['per_kg'].get(kg)
    return None if values is None else values[metric]


def format_metric(row, kg, metric):
    value = column_value(row, kg, metric)
    levels = sorted({v for other in method_rows[row['dataset']]
                     if (v := column_value(other, kg, metric)) is not None}, reverse=True)
    text = f'{value*100:.2f}'
    if value == levels[0]:
        return r'\textbf{' + text + '}'
    if len(levels) > 1 and value == levels[1]:
        return r'\underline{' + text + '}'
    return text


body = [r'\begin{table}[!ht]', r'\centering',
        r'\caption{Single-seed results by language and knowledge graph across all datasets (\%, seed 17).}',
        r'\label{tab:all-language-seed17}', r'\fontsize{8.2}{9.2}\selectfont',
        r'\setlength{\tabcolsep}{2pt}', r'\renewcommand{\arraystretch}{1.0}']
numeric_cells = 0
for block, (ds, (title, display_groups)) in enumerate(SPECS.items()):
    if block:
        body.append(r'\par\vspace{2pt}')
    cols = 2 + 3 * len(display_groups)
    body.append(r'\begin{tabularx}{\linewidth}{@{}L{33mm}*{' + str(cols-1) + r'}{C}@{}}')
    body.append(r'\toprule')
    targets = [kg for kg, _ in display_groups if kg != 'en_f']
    block_note = ('supplemental EN\\_F; original primary target FR' if supplement is not None else 'EN\\_F support $\\rightarrow$ FR target') if ds == 'wk3l' else f'{len(targets)} target KGs'
    body.append(r'\multicolumn{' + str(cols) + r'}{@{}l}{\textbf{' + title + '} --- ' + block_note + r'} \\')
    avg_label = 'AVG (FR)' if ds == 'wk3l' and supplement is not None else 'AVG'
    body.append('Method & ' + ' & '.join(r'\multicolumn{3}{c}{' + label + '}' for _, label in display_groups) + ' & '+avg_label+r' \\')
    body.append(''.join(r'\cmidrule(lr){' + f'{2+3*j}-{4+3*j}' + '}' for j in range(len(display_groups)))
                + r'\cmidrule(l){' + f'{cols}-{cols}' + '}')
    body.append(' & ' + ' & '.join(['H@1', 'H@10', 'MRR'] * len(display_groups) + ['MRR']) + r' \\')
    body.append(r'\midrule')
    for row in method_rows[ds]:
        label = r'\textbf{ASRC}' if row['method_key'] == 'asrc' else row['method']
        cells = []
        for kg, _ in display_groups:
            if kg == 'en_f':
                values = row.get('supplemental_kg_results', {}).get('en_f')
                if values is not None:
                    cells.extend(format_metric(row, kg, m) for m in METRICS)
                    numeric_cells += 3
                else:
                    cells.extend((['NR'] if supplement is not None else [r'\textit{n/e}']) * 3)
            elif row.get('role') == 'english_teacher_only':
                cells.extend([r'\textit{n/a}'] * 3)
            elif row['status'] != 'completed':
                cells.extend(['P' if row['status'] == 'pending' else 'NR'] * 3)
            else:
                cells.extend(format_metric(row, kg, m) for m in METRICS)
                numeric_cells += 3
        if row.get('role') == 'english_teacher_only':
            cells.append(r'\textit{n/a}')
        elif row['status'] == 'completed':
            cells.append(format_metric(row, 'AVG', 'mrr'))
            numeric_cells += 1
        else:
            cells.append('P' if row['status'] == 'pending' else 'NR')
        body.append(label + ' & ' + ' & '.join(cells) + r' \\')
    body.extend([r'\bottomrule', r'\end{tabularx}'])
body += [r'\par\vspace{4pt}', r'\begin{minipage}{\linewidth}\fontsize{7.5}{8.5}\selectfont',
    r'\textit{Note.} Seed 17 only, with no SD. Full-candidate filtered tail ranking uses ascending entity-ID ties; filtering is train+valid for core datasets and all for WK3l. AVG is the equal-KG macro MRR computed before rounding. Single-seed values may differ from the main table\textquotesingle s three-seed means. Bold and underline mark the best and second-best available values per column, ranked before rounding.',
    (r'\par \textit{Coverage.} JP denotes Japanese in E-PKG. EN\_F scores are supplemental; WK3l AVG remains FR-only. ASRC reuses its FR-validation-selected checkpoint. TransE (EN teacher) is the reused ATransN source teacher (margin 4, batch 1024), with no FR/AVG entry; it is not an English-target ATransN result.' if supplement is not None else
     r'\par \textit{Coverage.} JP denotes Japanese in E-PKG. EN\_F is support-only (\textit{n/e}: not evaluated), so WK3l AVG equals FR. ASRC uses parameter sharing on DBP-5L, DWY and WK3l, and separate KG parameters on E-PKG.'),
    (r'\par \textit{Split audit.} The original EN\_F test has 40,700 queries: 31 occur in training and 15 in validation. Original-split scores are shown; diagnostics on the remaining 40,654 queries are supplied separately. The original split and filtering sets are unchanged.' if supplement is not None else ''),
    r'\par \textit{Availability.} P: seed-17 result pending; NR: no accepted result; \textit{n/a}: not applicable. All filled values were verified against saved ranks and source JSON. English ASRC inference was added using frozen weights; no model was retrained.' if supplement is not None else
    r'\par \textit{Availability.} P: seed-17 result pending; NR: no accepted result. Both denote missing values, not zeros. All filled values were verified against saved ranks and source JSON, without training or new test inference.',
    r'\end{minipage}', r'\end{table}']
(OUT/'table_language_kg_seed17.tex').write_text('\n'.join(body) + '\n', encoding='utf-8')
(OUT/'main.tex').write_text(r'''% !TEX program = xelatex
\documentclass[10pt,a4paper,landscape]{article}
\usepackage[left=10mm,right=10mm,top=7mm,bottom=7mm]{geometry}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\usepackage{booktabs,array,tabularx,caption}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={Single-seed language and KG results},pdfauthor={}}
\captionsetup[table]{font=small,labelfont=bf,labelsep=period,justification=raggedright,singlelinecheck=false,skip=4pt}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\textfloatsep}{0pt}
\begin{document}
\input{table_language_kg_seed17.tex}
\end{document}
''', encoding='utf-8')

for rel, item in raw_sources.items():
    assert sha((ROOT/rel).read_bytes()) == item['sha256'], f'Source changed during export: {rel}'
audit = {'timestamp': timestamp, 'status': 'passed', 'seed': SEED, 'protocol': PROTOCOL,
         'dataset_count': len(SPECS), 'target_kg_count': len(query_reference),
         'primary_target_kg_count': 15, 'supplemental_kg_count': int(supplement is not None),
         'completed_method_dataset_rows': sum(r['status'] == 'completed' for rows in method_rows.values() for r in rows),
         'unfilled_method_dataset_rows': [{'dataset': ds, 'method': r['method'], 'status': r['status']}
                                         for ds, rows in method_rows.items() for r in rows if r['status'] != 'completed'],
         'numeric_pdf_cells': numeric_cells, 'maximum_metric_reconstruction_error': max_metric_error,
         'same_test_queries_within_each_kg': True, 'macro_before_rounding': True,
        'no_training_or_model_inference': True, 'source_hashes': raw_sources}
audit['no_training_or_model_inference'] = supplement is None
audit['no_training'] = True
audit['supplemental_english_frozen_tests'] = 1 if supplement is not None else 0
(OUT/'DATA_AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(OUT/'README.zh.md').write_text('''# 全数据集语言／KG 单种子汇总表

固定使用 seed 17，不混用主表三种子均值。按照截图的结构，在一张横向表格中分四个数据集区块；每个语言／KG 显示 H@1、H@10、MRR，末列是 KG 等权宏平均 MRR，单位均为百分数。

每个数据集区块逐列比较已填的有效数值：最优加粗，次优加下划线。按未四舍五入的原值排序；原值并列时共享相同标记，次优取第二个不同的值。P、NR、n/a 不参与排序。

保留 Table 2 全部适用比较方法。本版按用户要求移除四个 Independent 展示行；修改前版本与原始对照结果仍保留。DWY 按知识图谱来源展示。EN_F 的追加评测属于补充结果，WK3l 的 AVG 继续仅统计 FR。

P 表示 seed 17 的实验尚未完成；NR 表示当前协议下没有已接受结果。均不按 0 填写。当前缺少 IMKGC/E-PKG，以及 WK3l 上 LSMGA、DMKGC、IMKGC。

所有已填数值与保存的测试排名逐项核对。原核心数据/FR 结果保持不变；ASRC 的 EN_F 结果来自 seed17 冻结权重的一次追加测试。TransE (EN teacher) 单独复用 ATransN 英文教师的已保存成绩，不把它当作 ATransN 的英语目标结果。没有重新训练模型。

EN_F 原始测试中有 31 条查询与训练集重叠，另有 15 条与验证集重叠。表内保留原划分成绩；en_f_evaluation/RESULTS.json 同时提供排除这 46 条后的 40,654 条查询诊断结果，过滤集合保持原样。原始划分成绩不应描述为无重叠结果。

文件：main.tex 为编译入口，table_language_kg_seed17.tex 为可移入论文的表格，language_kg_seed17.csv/JSON 保留未四舍五入的小数（0–1），sources 保存来源结果 JSON 副本，DATA_AUDIT.json 记录原始排名文件哈希与数值核验。此目录为独立交付，不修改主任务的表格或实验队列。
''', encoding='utf-8')
print(json.dumps({k: v for k, v in audit.items() if k != 'source_hashes'}, ensure_ascii=False))
