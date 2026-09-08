"""Compact process/validation status; does not train or evaluate anything."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / 'reproduction/sota/graph_three_seed'
spec = json.loads((PHASE / 'manifest.json').read_text())
state = json.loads((PHASE / 'state.json').read_text())
recipes = json.loads((PHASE / 'PREFLIGHT.json').read_text())['recipes']
budget = {(r['method'], r['dataset']): r['rounds'] for r in recipes}
rows = []
for job in spec['jobs']:
    s = state['jobs'].get(job['id'], {})
    out = ROOT / job['output']
    curve = out / 'learning_curve.jsonl'
    history = [json.loads(x) for x in curve.read_text().splitlines()] if curve.exists() else []
    rounds = budget[(job['method'], job['dataset'])]
    row = {'id': job['id'], 'status': s.get('status', 'pending'), 'gpu': s.get('gpu'), 'pid': s.get('pid'),
           'validations': len(history), 'rounds': rounds}
    if 'started' in s:
        row['elapsed_minutes'] = round(((s.get('finished') or time.time()) - s['started']) / 60, 1)
    if history:
        row['latest_val_mrr'] = history[-1]['val_select_macro']['mrr']
        row['best_val_mrr'] = max(x['val_select_macro']['mrr'] for x in history)
    if row['status'] == 'running':
        try:
            os.kill(row['pid'], 0)
            row['process_alive'] = True
        except ProcessLookupError:
            row['process_alive'] = False
    rows.append(row)
counts = {status: sum(r['status'] == status for r in rows) for status in ('pending', 'running', 'completed', 'failed')}
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
report = {'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'counts': counts,
          'scheduler_pid': state.get('scheduler_pid'), 'gpu': gpu, 'jobs': rows,
          'local_seed17': 'Owned by the existing local queue; no server duplication',
          'next': 'Finish registered runs, audit saved ranks and publish three-seed Table2; no test-based model or seed selection'}
tmp = PHASE / 'CURRENT_STATE.tmp'
tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
tmp.replace(PHASE / 'CURRENT_STATE.json')
lines = ['# 图基线补齐种子进度', '', f'更新时间：{report["observed_at"]}', '',
         f'新增 18 次运行：完成 {counts["completed"]}，运行 {counts["running"]}，待跑 {counts["pending"]}，失败 {counts["failed"]}。', '',
         '本地 IMKGC seed 17 由已有队列继续，本阶段不重复启动。', '',
         '|实验|状态|GPU|PID|验证轮数|', '|---|---|---|---|---|']
lines += [f'|{r["id"]}|{r["status"]}|{r["gpu"] if r["gpu"] is not None else ""}|{r["pid"] or ""}|{r["validations"]}/{r["rounds"]}|' for r in rows]
lines += ['', '完成后按 seed 17/29/43 汇总，核验原始排名并更新 Table 2。']
(PHASE / 'STATUS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'observed_at': report['observed_at'], 'counts': counts, 'gpu': gpu,
                  'active': [r for r in rows if r['status'] in ('running', 'failed')]}))
