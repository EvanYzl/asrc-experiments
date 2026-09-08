"""One-time, archived migration for the user's reduced baseline scope."""
import csv
import datetime
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path('G:/zhishitupui')
SUITE = ROOT / 'reproduction/strict_baselines'
RUN = ROOT / 'reproduction/runs/strict_baselines_20260905'
TABLES = ROOT / 'outputs/kbs_main_tables'
ARCHIVE = Path(__file__).resolve().parent
changed = []
moves = []


def backup(path):
    path = Path(path)
    destination = ARCHIVE / 'before' / path.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)


def write(path, text):
    path = Path(path)
    if path.exists():
        backup(path)
    path.write_text(text, encoding='utf-8')
    changed.append(str(path.relative_to(ROOT)))


def edit(path, transform):
    write(path, transform(path.read_text(encoding='utf-8-sig')))


def replace(text, old, new):
    assert old in text, old
    return text.replace(old, new)


def dump(path, value):
    write(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def archive_move(path):
    path = Path(path).resolve()
    if path.exists():
        destination = ARCHIVE / 'retired' / path.relative_to(ROOT)
        moves.append({'source': str(path), 'destination': str(destination.resolve())})


for p in [RUN/'manifest.json', RUN/'queue_state.json', TABLES/'cells_results.csv',
          TABLES/'KBS_Main_Text_Tables.pdf', TABLES/'table_manifest.json',
          TABLES/'validation_report.json', RUN/'STATUS.json', RUN/'STATUS.zh.md',
          RUN/'TABLE_SCOPE.zh.md', RUN/'code_snapshot_manifest.json',
          ROOT/'refine-logs/BASELINE_EXPERIMENT_TRACKER.md',
          ROOT/'refine-logs/BASELINE_TABLE_SCOPE.md']:
    backup(p)

edit(SUITE/'table_scope.py', lambda t: replace(replace(t,
    "{'kens':'kens_v2','align':'alignkgc','ssaga':'ssaga'}",
    "{'align':'alignkgc','ssaga':'ssaga'}"),
    'KEnS 的标准全实体 MRR 和不兼容的 IMKGC-DWY 已在原表中明确不适用，不作为待填实验槽位。',
    'IMKGC-DWY 的公开划分不兼容，原表保留不适用标记，不作为待填实验槽位。'))


def clean_extend(t):
    t = replace(t, "KENS=str(ROOT/'reproduction/envs/kens-tf210/Scripts/python.exe')\n", '')
    start = t.index("    env={'PYTHONPATH':'','CUDA_VISIBLE_DEVICES':'-1'")
    end = t.index("    for short,script in [('alignkgc'", start)
    t = t[:start] + t[end:]
    return replace(t, "('kens','alignkgc','internal_')", "('alignkgc','internal_')")


edit(SUITE/'extend_queue.py', clean_extend)
edit(SUITE/'focus_tables.py', lambda t: replace(replace(replace(t,
    '# Prioritize corrected KEnS reruns and complete three-seed baseline groups.',
    '# Complete three-seed baseline groups before switching methods.'),
    "        if name.startswith('kens_v2_'):return (0,name)\n", ''),
    "        if name.startswith('kens_'):return (5,name)\n", ''))
edit(SUITE/'run_queue_lanes.py', lambda t: replace(t,
    "        if 'kens_v2_' in job['id']:\n            return result.get('training_revision')=='kens-transe-paper-loss-v2'\n", ''))


def clean_collector(t):
    start = t.index('def audited_kens(')
    end = t.index('def collect_benchmarks(', start)
    t = t[:start] + t[end:]
    t = replace(t, "    if method=='KEnS':\n        if obj.get('training_revision')!='kens-transe-paper-loss-v2':return None\n        return audited_kens(path,obj)\n", '')
    # Only the current manifest can authorize publication, including nested
    # outputs from internal controls. Archived directories can never re-enter.
    t = replace(t, "    for path in sorted((RUN/'jobs').rglob('result.json')):\n        try:result=audited_result(path)",
        "    active_job_ids={j['id'] for j in manifest['jobs']}\n    for path in sorted((RUN/'jobs').rglob('result.json')):\n        if path.relative_to(RUN/'jobs').parts[0] not in active_job_ids:continue\n        try:result=audited_result(path)")
    t = replace(t, "['KEnS','AlignKGC','SS-AGA']", "['AlignKGC','SS-AGA']")
    t = replace(t, "{'KEnS':'kens','AlignKGC':'align','SS-AGA':'ssaga'}", "{'AlignKGC':'align','SS-AGA':'ssaga'}")
    t = replace(t, "['h1','h10'] if method=='KEnS' else ['mrr','h1','h10']", "['mrr','h1','h10']")
    t = replace(t, "selection_protocol='fixed rounds; public-validation ensemble weights' if method=='KEnS' else 'val_select tail MRR, earliest maximum'", "selection_protocol='val_select tail MRR, earliest maximum'")
    t = replace(t, "candidate_scope='top-n nominations per model; weighted ensemble' if method=='KEnS' else 'all target-KG entities'", "candidate_scope='all target-KG entities'")
    t = replace(t, "notes=('KEnS corrected TransE loss/sampling/CSLS/boosting; paper batch256 and L2=.0001; fixed released rounds; nomination-specific Hits. See KENS_REPAIR.zh.md.'\n                        if method=='KEnS' else 'Raw query ranks and checkpoints validated; source implementation recorded in each config.json'))", "notes='Raw query ranks and checkpoints validated; source implementation recorded in each config.json')")
    assert 'kens' not in t.lower()
    return t


edit(SUITE/'collect_tables.py', clean_collector)
edit(SUITE/'archive_provenance.py', lambda t: replace(replace(t,
    "'ATransN','KEnS','SS-AGA'", "'ATransN','SS-AGA'"),
    ",('tensorflow',str(ROOT/'reproduction/envs/kens-tf210/Scripts/python.exe'))", ''))


def clean_t3(t):
    lines = [line for line in t.splitlines() if 'kens' not in line.lower()]
    pos = lines.index(r'\end{table*}')
    lines.insert(pos, r'\tnote{Values are local three-seed mean $\pm$ sample SD; method-specific settings are not ranked against Table~\ref{tab:2}. AlignKGC uses 20\% entity/relation alignment. SS-AGA retains supporter validation facts and uses regenerated mBERT label features. Unfilled cells await local reruns.}')
    return '\n'.join(lines) + '\n'


edit(TABLES/'tables/t03.tex', clean_t3)


def clean_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    kept = [r for r in rows if not any('kens' in str(v).lower() for v in r.values())]
    backup(path)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    changed.append(str(path.relative_to(ROOT)))
    return len(rows), len(kept)


for path in [TABLES/'cells_template.csv', TABLES/'cells_results.csv']:
    assert clean_csv(path) == (694, 688)
for path in [ROOT/'baselines/BASELINES_MANIFEST.csv', ROOT/'refine-logs/REPORTED_RESULTS.csv']:
    clean_csv(path)


def clean_table_readme(t):
    t = replace(t, '表2和表3仅使用本地三种子实测；KEnS 旧异常结果已撤下，正在修复重跑。', '表2和表3仅使用本地三种子实测。')
    t = t.replace('KEnS、AlignKGC、SS-AGA', 'AlignKGC、SS-AGA')
    t = t.replace('694 个', '688 个').replace('10 个', '9 个')
    return '\n'.join(line for line in t.splitlines() if 'kens' not in line.lower()) + '\n'


edit(TABLES/'README.zh.md', clean_table_readme)


def clean_execution(t):
    t = t.replace('KEnS、AlignKGC、SS-AGA', 'AlignKGC、SS-AGA')
    t = t.replace('单 GPU 串行，内存允许时与一个 KEnS CPU 任务并行，', '单 GPU 串行，')
    t = t.replace('队列优先运行修复后的 KEnS 并补齐已启动的三种子组。', '队列优先补齐已启动的三种子组。')
    t = t.replace('KEnS 固定训练轮数，其集成权重只用完整公开验证集拟合；KEnS/SS-AGA 按发布模型保留支持域验证事实。', 'SS-AGA 按发布模型保留支持域验证事实。')
    return '\n'.join(line for line in t.splitlines() if 'kens' not in line.lower()) + '\n'


edit(SUITE/'EXECUTION_PLAN.zh.md', clean_execution)
for path in [ROOT/'refine-logs/BASELINE_EXECUTION_PLAN.md', ROOT/'refine-logs/EXPERIMENT_PLAN.md']:
    write(path, (SUITE/'EXECUTION_PLAN.zh.md').read_text(encoding='utf-8'))
edit(SUITE/'ARTIFACT_SCHEMA.zh.md', lambda t: '\n'.join(line for line in
    t.replace('，含 PyG overlay 和独立 TF 环境', '，含 PyG overlay')
     .replace('资源审计；KEnS CPU 任务另由队列记录 RSS 和耗时', '资源审计；队列另行保存进程树 RSS 和耗时').splitlines()
    if 'kens' not in line.lower()) + '\n')
edit(RUN/'README.zh.md', lambda t: '\n'.join(
    ('`run_queue_lanes.py` 管理当前 GPU 任务，启动参数为 `--min-free-memory-gb 4`。每次只运行一个 GPU 训练任务，保存进程树资源记录。' if line.startswith('KEnS 旧结果') else line)
    for line in t.splitlines()) + '\n')
edit(RUN/'CONTINUATION.zh.md', lambda t: t.replace('133 格', '127 格').replace('一个 GPU 和一个 CPU 任务并行', '串行运行 GPU 任务'))

manifest = json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
removed_jobs = []
for key in ['jobs', 'deferred_jobs']:
    removed_jobs.extend(j['id'] for j in manifest[key] if 'kens' in j['id'])
    manifest[key] = [j for j in manifest[key] if 'kens' not in j['id']]
all_ids = {j['id'] for k in ['jobs', 'deferred_jobs'] for j in manifest[k]}
assert all(set(j.get('depends_on', [])) <= all_ids for k in ['jobs', 'deferred_jobs'] for j in manifest[k])
assert 'kens' not in json.dumps(manifest).lower()
dump(RUN/'manifest.json', manifest)
state = json.loads((RUN/'queue_state.json').read_text(encoding='utf-8'))
state['jobs'] = {name: value for name, value in state['jobs'].items() if 'kens' not in name}
state['current_jobs'] = [name for name in state.get('current_jobs', []) if 'kens' not in name]
state['current_job'] = next(iter(state['current_jobs']), None)
assert 'kens' not in json.dumps(state).lower()
dump(RUN/'queue_state.json', state)

# Older entry points also lose this method, so a legacy manifest cannot schedule it.
legacy = ROOT/'reproduction/queue'
legacy_manifest = json.loads((legacy/'baseline_full_manifest.json').read_text(encoding='utf-8'))
legacy_manifest['jobs'] = [j for j in legacy_manifest['jobs'] if 'kens' not in json.dumps(j).lower()]
index = 0
for job in legacy_manifest['jobs']:
    if not job.get('support_job'):
        index += 1
        job['baseline_index'] = index
dump(legacy/'baseline_full_manifest.json', legacy_manifest)


def clean_legacy_collector(t):
    start = t.index('def parse_kens(')
    end = t.index('def parse_alignkgc(', start)
    t = t[:start] + t[end:]
    return replace(t, '    if method == "KEnS (TransE)":\n        return parse_kens(text)\n', '')


edit(legacy/'collect_baseline_results.py', clean_legacy_collector)


def clean_legacy_verifier(t):
    t = t.replace('len(baseline_jobs) == 10', 'len(baseline_jobs) == 9').replace('exactly ten baselines', 'exactly nine baselines')
    start = t.index('                if job["method"] == "KEnS (TransE)":')
    end = t.index('        if job.get("artifact_capture"):', start)
    return t[:start] + '                for metric in ("mrr", "hits1", "hits10"):\n                    require(row.get(metric) is not None,\n                            f"{job_id}: missing {metric}", local_failures)\n' + t[end:]


edit(legacy/'verify_formal_results.py', clean_legacy_verifier)
edit(ROOT/'baselines/README.md', lambda t: '\n'.join(line for line in t.replace('10 个方法由 8 个', '9 个方法由 7 个').splitlines() if 'kens' not in line.lower()) + '\n')
edit(ROOT/'reproduction/ENVIRONMENT.md', lambda t: re.sub(r'## KEnS 隔离环境.*?(?=## 运行与缓存位置)', '', t, flags=re.S)
    .replace('runs\\baseline_reproduction_20260904', 'runs\\strict_baselines_20260905')
    .replace('queue\\baseline_full_manifest.json', 'runs\\strict_baselines_20260905\\manifest.json')
    .replace('详见 `queue\\verify_formal_results.py`', '详见 `strict_baselines\\collect_tables.py`'))
edit(ROOT/'reproduction/PATCHES.md', lambda t: '\n'.join(line for line in t.splitlines() if 'kens' not in line.lower()) + '\n')
edit(ROOT/'reproduction/BASELINE_REPRODUCTION_REPORT.md', lambda t: '\n'.join(line for line in t.splitlines() if 'kens' not in line.lower()).replace('10 个', '9 个') + '\n')
edit(ROOT/'reproduction/README.md', lambda t: '\n'.join(line for line in t.replace('KEnS / SS-AGA', 'SS-AGA').replace('10 个', '9 个').splitlines() if 'kens' not in line.lower()) + '\n')
write(ROOT/'reproduction/STATUS.md', '# 当前复现实验进度\n\n当前仅运行 Table 1–3 的非 QURA 项，使用三个随机种子。实时状态见 [STATUS.zh.md](runs/strict_baselines_20260905/STATUS.zh.md)，当前执行计划见 [EXECUTION_PLAN.zh.md](strict_baselines/EXECUTION_PLAN.zh.md)。\n')
edit(ROOT/'refine-logs/LITERATURE_COMPARISON_LEDGER.md', lambda t: '\n'.join(line for line in t.replace('SS-AGA 与 KEnS', 'SS-AGA').replace('10 个', '9 个').replace('10×4', '9×4').splitlines() if 'kens' not in line.lower()) + '\n')
smoke = json.loads((ROOT/'reproduction/smoke_results.json').read_text(encoding='utf-8'))
smoke['baselines'] = [b for b in smoke['baselines'] if 'kens' not in json.dumps(b).lower()]
smoke['summary']['baselines_passed'] = smoke['summary']['baselines_total'] = len(smoke['baselines'])
dump(ROOT/'reproduction/smoke_results.json', smoke)

# Move retired executable code and raw artifacts outside all active scans.
for parent in [SUITE, SUITE/'__pycache__', RUN, RUN/'jobs', RUN/'smoke', RUN/'logs', RUN/'results',
               ROOT/'reproduction/queue', ROOT/'reproduction/tests']:
    if parent.exists():
        for path in parent.iterdir():
            if 'kens' in path.name.lower():
                archive_move(path)
for path in [RUN/'results/active_scope_verification.json', RUN/'tensorflow_requirements_snapshot.txt',
             ROOT/'reproduction/KENS_KNN_AUDIT.md', ROOT/'outputs/paper_experiment_tables',
             ROOT/'outputs/KBS_Main_Text_Tables_LaTeX.zip']:
    archive_move(path)
for path in TABLES.glob('values_*.tex'):
    if 'kens' in path.read_text(encoding='utf-8').lower():
        archive_move(path)
for path in (ROOT/'refine-logs').iterdir():
    if path.is_file() and re.search(r'_20\d{6}_\d{6}\.', path.name) and path.suffix in ['.md', '.csv']:
        if 'kens' in path.read_text(encoding='utf-8-sig').lower():
            archive_move(path)
unique = {item['source']: item for item in moves}
(ARCHIVE/'move_plan.json').write_text(json.dumps(list(unique.values()), indent=2) + '\n', encoding='utf-8')
(ARCHIVE/'migration.json').write_text(json.dumps({'time': datetime.datetime.now().astimezone().isoformat(),
    'removed_jobs': removed_jobs, 'changed_files': changed, 'move_count': len(unique),
    'active_jobs': len(manifest['jobs']), 'deferred_jobs': len(manifest['deferred_jobs']),
    'table_slots': 688, 'active_table_slots': 127}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'changed_files': len(changed), 'moves_prepared': len(unique),
    'removed_jobs': len(removed_jobs), 'active_jobs': len(manifest['jobs']), 'deferred_jobs': len(manifest['deferred_jobs'])}))
