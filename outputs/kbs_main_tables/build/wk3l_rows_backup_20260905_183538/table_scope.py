"""Map every existing table slot to required baseline runs or proposed-method scope."""
import collections
import csv
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'reproduction/runs/strict_baselines_20260905'
DS = {'dbp':'dbp5l', 'dbp5l':'dbp5l', 'epkg':'depkg', 'dwy':'dwy', 'wk3l':'wk3l'}
SEEDS = [17, 29, 43]


def cell_scope(ident):
    parts = ident.split('.'); table, dataset = parts[:2]
    def required(prefix, note):
        return {'scope':'baseline_required', 'jobs':[f'{prefix}_s{s}' for s in SEEDS], 'reason':note}
    def excluded(note):
        return {'scope':'proposed_method_excluded', 'jobs':[], 'reason':note}
    if table == 'T1':
        return {'scope':'dataset_statistics', 'jobs':[], 'reason':'Recomputed from frozen public dataset manifests'}
    if table in ['T5', 'T6']:
        return excluded('QURA component ablation' if table == 'T5' else 'QURA threshold and risk certification')
    if table == 'T2':
        method = parts[2]; ds = DS[dataset]
        if method in ['qura','delta_mrr','mrr_ci']:
            return excluded('QURA score or paired effect relative to Uniform-All')
        if method == 'target':
            return required(f'internal_base_{ds}', 'Target-only shared 128-dimensional baseline')
        if method == 'uniform':
            return required(f'internal_teacher_{ds}', 'Uniform-All shared frozen fusion baseline')
        assert method in ['transe','distmult','rotate','atransn','lsmga','dmkgc','imkgc'], ident
        return required(f'{method}_{ds}', 'Full-domain external baseline; validation-only checkpoint selection')
    if table == 'T3':
        method = {'align':'alignkgc','ssaga':'ssaga'}[parts[2]]
        return required(f'{method}_{DS[dataset]}', 'Method-specific local rerun; three model seeds')
    if table == 'T4':
        method = parts[2]; ds = DS[dataset]
        if method in ['qura','delta_ntr','ntr_ci']:
            return excluded('QURA negative transfer or paired effect')
        if method == 'uniform':
            return required(f'internal_teacher_{ds}', 'Uniform-All paired query ranks and source-access records')
        assert method in ['global','random','attn'], ident
        return required(f'internal_controls_{ds}', 'Budget-prefix data saved; final budget must match QURA validation access' if method != 'attn' else 'Query Attention without LOSO supervision')
    if table == 'T7':
        condition, method = parts[2:4]
        if method == 'qura':
            return excluded('QURA robustness under data or source shift')
        assert method in ['uniform','random'], ident
        if dataset in ['el','ja']:
            return required('internal_controls_dbp5l', f'Frozen {dataset.upper()} checkpoint, {condition} visible-source condition')
        if condition in ['full','target20','align20']:
            suffix = '' if condition == 'full' else '_' + condition
            stage = 'teacher' if method == 'uniform' else 'controls'
            return required(f'internal_{stage}_dbp5l{suffix}', 'Matched Target-only reference; refit affected low-resource models')
        assert condition in ['corrupt10','corrupt20','corrupt40'], ident
        return required('internal_controls_dbp5l', 'Frozen checkpoint with deterministic alignment corruption')
    if table == 'T8':
        method = parts[2]
        if method in ['qura','post','lifecycle']:
            return excluded('QURA gate/post-fetch ablation or its complete lifecycle accounting')
        assert method in ['target','uniform'], ident
        return required(f'internal_benchmark_{DS[dataset]}', '5000 fixed queries, 3 warm-ups and 10 timed passes per seed')
    raise ValueError('Unmapped table cell: ' + ident)


def update_scope():
    tables = ROOT / 'outputs/kbs_main_tables'
    with (tables / 'cells_results.csv').open(encoding='utf-8-sig', newline='') as f:
        cells = list(csv.DictReader(f))
    manifest = json.loads((RUN / 'manifest.json').read_text(encoding='utf-8'))
    ids = {j['id'] for j in [*manifest['jobs'],*manifest.get('deferred_jobs',[])]}
    active_tables=manifest.get('active_tables',['T1','T2','T3'] if manifest.get('focus')=='tables_1_2_3_without_qura' else None)
    priority_only=active_tables is not None
    batch=manifest.get('method_batch')
    batch_jobs=set(batch['job_ids']) if batch else None
    by_table = collections.defaultdict(collections.Counter)
    rows, missing = [], []
    for cell in cells:
        ident = cell['cell_id']; spec = cell_scope(ident)
        unqueued = set(spec['jobs']) - ids
        if unqueued:
            missing.append({'cell_id':ident, 'missing_jobs':sorted(unqueued)})
        done = cell['value'] != ''
        state = 'excluded' if spec['scope'] == 'proposed_method_excluded' else ('filled' if done else 'pending')
        if priority_only and cell['table_id'] not in active_tables and state=='pending':
            state='deferred'
        if batch_jobs is not None and state=='pending' and spec['jobs'] and not batch_jobs.intersection(spec['jobs']):
            state='deferred'
        if state=='pending' and cell['table_id']=='T4' and ident.split('.')[2] in ['global','random']:
            state='waiting_qura_budget'
        if state == 'excluded':
            assert not done, 'Excluded proposed-method cell was populated: ' + ident
        by_table[cell['table_id']][state] += 1
        rows.append({'cell_id':ident, 'scope':spec['scope'], 'state':state,
                     'required_jobs':';'.join(spec['jobs']), 'reason':spec['reason']})
    assert not missing, 'Baseline cells have no queued execution: ' + repr(missing)
    counts = collections.Counter(r['state'] for r in rows)
    payload = {'updated_at':datetime.datetime.now().astimezone().isoformat(),
        'total_slots':len(rows), 'eligible_baseline_and_statistics_slots':counts['filled']+counts['pending']+counts['waiting_qura_budget'],
        'filled':counts['filled'], 'pending':counts['pending'], 'proposed_method_excluded':counts['excluded'],
        'deferred_baseline_slots':counts['deferred'], 'focus':manifest.get('focus','all_baseline_tables'),
        'active_tables':active_tables,
        'waiting_qura_budget':counts['waiting_qura_budget'],
        'all_baseline_and_statistics_slots':counts['filled']+counts['pending']+counts['deferred']+counts['waiting_qura_budget'],
        'all_required_jobs_queued':True, 'priority_tables_complete':counts['pending']==0 and counts['waiting_qura_budget']==0,
        'runnable_tables_complete':counts['pending']==0,
        'baseline_tables_complete':counts['pending']==0 and counts['deferred']==0 and counts['waiting_qura_budget']==0,
        'by_table':{k:dict(v) for k,v in sorted(by_table.items())}, 'missing_jobs':missing}
    if batch:
        selected=[r for r in rows if batch_jobs.intersection(r['required_jobs'].split(';'))]
        assert len(selected)==batch['expected_cells'], 'Selected batch cell coverage changed'
        payload['method_batch']={'methods':batch['methods'],'job_ids':batch['job_ids'],
            'cells':len(selected),'filled':sum(r['state']=='filled' for r in selected),
            'pending':sum(r['state']!='filled' for r in selected),
            'complete':all(r['state']=='filled' for r in selected),
            'stop_after_completion':True,
            'other_unfinished_cells_in_tables_1_3':sum(r['state']=='deferred' and r['cell_id'].split('.')[0] in ['T1','T2','T3'] for r in rows)}
    result = RUN / 'results'; result.mkdir(exist_ok=True)
    for path, body in [(result/'table_scope.json',json.dumps(payload,ensure_ascii=False,indent=2)+'\n')]:
        tmp=path.with_suffix('.tmp');tmp.write_text(body,encoding='utf-8');tmp.replace(path)
    with (result/'table_scope.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 全部正文表的基线执行范围','',
           (batch['instruction'] if batch else ('用户最新要求：只填写 '+ '、'.join(active_tables) +' 中不涉及 QURA 的结果。QURA 行、相关差值与区间留空；其他表格不填。' if priority_only else '用户要求：除我们提出的方法及其消融、认证、衍生差值外，完成全部现有表格中的基线实验。')),
           ('Random / Global-Utility 的预算匹配与 Table 4 实验暂缓；后续依据冻结的 QURA 验证集访问量确定预算。' if priority_only and 'T4' not in active_tables else 'Random / Global-Utility 保存完整预算前缀曲线，待 QURA 的验证集访问量冻结后确定匹配预算；此前不填其主表数值。'),'',
           '| 表 | 已填 | 当前待运行 | 等待 QURA 预算 | 暂缓基线 | 我们方法相关，不运行 |','|---|---:|---:|---:|---:|---:|']
    for table, c in sorted(by_table.items()):
        lines.append(f"| {table} | {c['filled']} | {c['pending']} | {c['waiting_qura_budget']} | {c['deferred']} | {c['excluded']} |")
    lines += ['',f"当前表格基线与统计共 {payload['eligible_baseline_and_statistics_slots']} 格：已填 {counts['filled']} 格，待运行 {counts['pending']} 格，另有 {counts['waiting_qura_budget']} 格等待 QURA 验证集预算。",
              '所有应做单元格均映射到现有队列任务。完整逐格映射：results/table_scope.csv。',
              'IMKGC-DWY 的公开划分不兼容，原表保留不适用标记，不作为待填实验槽位。']
    if batch:
        b=payload['method_batch']
        lines += ['',f"本批次只新增 {'、'.join(batch['methods'])} 的 {b['cells']} 格，已填 {b['filled']} 格；全部验收回填后停止。",
                  f"Table 1–3 中其余 {b['other_unfinished_cells_in_tables_1_3']} 个未完成单元格暂停，已有数值保留，等待用户进一步指令。"]
    text='\n'.join(lines)+'\n'
    destination=RUN/'TABLE_SCOPE.zh.md'
    if not destination.exists():
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        (ROOT/f'refine-logs/BASELINE_TABLE_SCOPE_{stamp}.md').write_text(text,encoding='utf-8')
    destination.write_text(text,encoding='utf-8')
    (ROOT/'refine-logs/BASELINE_TABLE_SCOPE.md').write_text(text,encoding='utf-8')
    return payload


if __name__ == '__main__':
    print(json.dumps(update_scope(),ensure_ascii=False))
