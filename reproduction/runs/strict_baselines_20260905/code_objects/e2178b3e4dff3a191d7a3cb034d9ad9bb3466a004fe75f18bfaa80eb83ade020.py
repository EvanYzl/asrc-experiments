"""Summarize logical queue progress, retaining recovered attempt history."""
import collections
import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'


def update():
    manifest=json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
    state=json.loads((RUN/'queue_state.json').read_text(encoding='utf-8'))
    replaced={j['supersedes']:j['id'] for j in manifest['jobs'] if j.get('supersedes')}
    logical=[j for j in manifest['jobs'] if j['id'] not in replaced]
    rows=[];counts={'formal':collections.Counter(),'smoke':collections.Counter()}
    for job in logical:
        status=state['jobs'].get(job['id'],{}).get('status','pending')
        kind='smoke' if job['id'].startswith('smoke_') else 'formal';counts[kind][status]+=1
        rows.append({'job_id':job['id'],'kind':kind,'status':status,'result':job['result']})
    audit=json.loads((RUN/'results/table_fill_audit.json').read_text()) if (RUN/'results/table_fill_audit.json').exists() else {}
    unresolved=[j['job_id'] for j in rows if j['status']=='failed']
    recovered=[{'failed_attempt':old,'replacement':new,'replacement_status':state['jobs'].get(new,{}).get('status','pending')} for old,new in replaced.items()]
    complete=all(j['status']=='completed' for j in rows)
    scope_path=RUN/'results/table_scope.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8')) if scope_path.exists() else {}
    batch=manifest.get('method_batch')
    active=manifest.get('active_tables',['T1','T2','T3'])
    active_label='Table '+', '.join(t[1:] for t in active)
    current=state.get('current_jobs') or ([state['current_job']] if state.get('current_job') else [])
    payload={'updated_at':datetime.datetime.now().astimezone().isoformat(),'logical_status':'completed' if complete else ('needs_attention' if unresolved and not state.get('current_job') else 'running'),
        'current_job':state.get('current_job'),'current_jobs':current,'active_tables':active,'counts':{k:dict(v) for k,v in counts.items()},'filled_cells':audit.get('filled_cells',0),
        'completed_three_seed_groups':audit.get('three_seed_groups',[]),'unresolved_failures':unresolved,'recovered_attempts':recovered,
        'remaining_dependency':('Global/Random await a frozen QURA validation access budget' if 'T4' in active else 'None of the active baseline cells requires QURA; Table 4 and budget matching are deferred'),
        'table_scope':scope,
        'jobs':rows}
    if batch:
        batch_counts=collections.Counter(state['jobs'].get(name,{}).get('status','pending') for name in batch['job_ids'])
        payload['method_batch']={**scope.get('method_batch',{}),'job_counts':dict(batch_counts),
            'instruction':batch['instruction'],'stop_after_completion':True}
    temp=RUN/'STATUS.tmp';temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(RUN/'STATUS.json')
    lines=['# 基线复现实验进度','',f"更新时间：{payload['updated_at']}",'',
        f"当前任务：`{'、'.join(current) or '无'}`。正式任务完成 {counts['formal']['completed']}/{sum(counts['formal'].values())}，运行检查完成 {counts['smoke']['completed']}/{sum(counts['smoke'].values())}。正式任务数包含外部基线、共享骨干及教师。",
        f"已验收三种子组数：{len(payload['completed_three_seed_groups'])}；已填单元格：{payload['filled_cells']}（含数据集统计）。",'',
        f"{active_label} 当前执行范围（含保留的既有结果）共 {scope.get('filled',0)+scope.get('pending',0)} 格，已填 {scope.get('filled',0)} 格，待完成 {scope.get('pending',0)} 格。我们方法相关 {scope.get('proposed_method_excluded','待核对')} 格按要求不运行。",
        (f"Global/Random 的 {scope.get('waiting_qura_budget',0)} 格需等待 QURA 验证集访问量确定匹配预算。" if 'T4' in active else 'Table 4 的控制与预算实验暂缓，Table 4–8 留空。'),
        ('所有性能值来自本机运行，文献值不代填。'+(batch['instruction'] if batch else f'当前只填 {active_label} 的非 QURA 项，其余表格留空。')),'',
        '输出：`outputs/kbs_main_tables/KBS_Main_Text_Tables.pdf`、`outputs/kbs_main_tables/cells_results.csv`；原始数据、检查点、日志和图在本运行目录。','',
        '## 历史任务及替代运行','']
    if batch:
        lines[4:4]=[f"本批次：{'、'.join(batch['methods'])}；正式任务完成 {batch_counts['completed']}/{len(batch['job_ids'])}。全部验收回填后停止，等待用户指令。",'']
    for item in recovered:lines.append(f"- `{item['failed_attempt']}` → `{item['replacement']}`：{item['replacement_status']}。原始失败日志保留。")
    if unresolved:lines+=['','## 需要处理的失败','',*[f'- `{name}`' for name in unresolved]]
    lines+=['','## 执行清单','','| 任务 | 类型 | 状态 |','|---|---|---|']
    lines += [f"| `{r['job_id']}` | {r['kind']} | {r['status']} |" for r in rows]
    text='\n'.join(lines)+'\n';(RUN/'STATUS.zh.md').write_text(text,encoding='utf-8')
    (ROOT/'refine-logs/BASELINE_EXPERIMENT_TRACKER.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ['jobs','completed_three_seed_groups']},ensure_ascii=False))
    return payload


if __name__=='__main__':update()
