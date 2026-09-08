"""Add the three requested WK3l comparison rows without scheduling experiments."""
import ast
import csv
import datetime
import io
import json
import msvcrt
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('G:/zhishitupui')
TABLES = ROOT / 'outputs/kbs_main_tables'
SUITE = ROOT / 'reproduction/strict_baselines'
RUN = ROOT / 'reproduction/runs/strict_baselines_20260905'
METHODS = ['lsmga', 'dmkgc', 'imkgc']
NEW_IDS = [f'T2.wk3l.{method}.{metric}' for method in METHODS for metric in ['mrr', 'h1', 'h10']]
NOTE = 'WK3l adaptation and three-seed local reruns pending; no execution jobs configured yet'


def replace_once(text, before, after):
    assert text.count(before) == 1, f'Expected one edit anchor: {before!r}'
    return text.replace(before, after, 1)


def atomic_text(path, text):
    temporary = path.with_name(path.name + '.wk3l-edit.tmp')
    temporary.write_text(text, encoding='utf-8', newline='')
    os.replace(temporary, path)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def csv_text(fields, rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def edit_and_build():
    scope_path = SUITE / 'table_scope.py'
    scope = scope_path.read_text(encoding='utf-8')
    scope = replace_once(scope,
        "        return required(f'{method}_{ds}', 'Full-domain external baseline; validation-only checkpoint selection')",
        "        spec = required(f'{method}_{ds}', 'Full-domain external baseline; validation-only checkpoint selection')\n"
        "        if ds == 'wk3l' and method in ['lsmga','dmkgc','imkgc']:\n"
        "            spec['awaiting_adaptation'] = True\n"
        "            spec['reason'] = 'WK3l extension requires adapted local reruns; validation-only checkpoint selection'\n"
        "        return spec")
    scope = replace_once(scope, '    rows, missing = [], []',
                         '    rows, missing, awaiting_adaptation = [], [], []')
    scope = replace_once(scope,
        "            missing.append({'cell_id':ident, 'missing_jobs':sorted(unqueued)})",
        "            destination = awaiting_adaptation if spec.get('awaiting_adaptation') else missing\n"
        "            destination.append({'cell_id':ident, 'missing_jobs':sorted(unqueued)})")
    scope = replace_once(scope,
        "        if priority_only and cell['table_id'] not in active_tables and state=='pending':",
        "        if unqueued and spec.get('awaiting_adaptation') and state == 'pending':\n"
        "            state = 'deferred'\n"
        "        if priority_only and cell['table_id'] not in active_tables and state=='pending':")
    scope = replace_once(scope, "'all_required_jobs_queued':True,",
                         "'all_required_jobs_queued':not awaiting_adaptation,")
    scope = replace_once(scope, "'missing_jobs':missing}",
                         "'missing_jobs':missing + awaiting_adaptation,\n"
                         "        'awaiting_adaptation_slots':len(awaiting_adaptation)}")
    scope = replace_once(scope,
        "              '所有应做单元格均映射到现有队列任务。完整逐格映射：results/table_scope.csv。',",
        "              (f'新增 WK3l 的 LSMGA、DMKGC、IMKGC 共 {len(awaiting_adaptation)} 格等待适配与复现，尚未配置执行任务；其余应做格均已映射队列。完整逐格映射：results/table_scope.csv。' if awaiting_adaptation else '所有应做单元格均映射到现有队列任务。完整逐格映射：results/table_scope.csv。'),")
    ast.parse(scope)

    table_path = TABLES / 'tables/t02.tex'
    table = table_path.read_text(encoding='utf-8')
    assert all(ident not in table for ident in NEW_IDS), 'Requested rows already exist'
    added = ''.join(method.upper() + ' & ' + ' & '.join(
        r'\PM{' + f'T2.wk3l.{method}.{metric}' + '}' for metric in ['mrr', 'h1', 'h10'])
        + r' \\' + '\n' for method in METHODS)
    table = replace_once(table, 'Target-only & \\PM{T2.wk3l.target.mrr}',
                         added + 'Target-only & \\PM{T2.wk3l.target.mrr}')
    table = replace_once(table, 'Core results are KG macro averages; WK3l evaluates FR only.',
                         'Core results are KG macro averages; WK3l evaluates FR only. '
                         'WK3l LSMGA, DMKGC and IMKGC await adaptation and local reruns.')

    template_fields, template = read_csv(TABLES / 'cells_template.csv')
    result_fields, results = read_csv(TABLES / 'cells_results.csv')
    assert template_fields == result_fields
    before_results = {row['cell_id']: row.copy() for row in results}
    before_values = (TABLES / 'values.tex').read_bytes()
    for rows, is_result in [(template, False), (results, True)]:
        assert not any(row['cell_id'] in NEW_IDS for row in rows)
        insertion = next(i for i, row in enumerate(rows) if row['cell_id'] == 'T2.wk3l.target.mrr')
        new_rows = []
        for ident in NEW_IDS:
            row = dict.fromkeys(template_fields, '')
            row.update(cell_id=ident, table_id='T2', placeholder_kind='PM')
            if is_result:
                row.update(source_type='pending', notes=NOTE)
            new_rows.append(row)
        rows[insertion:insertion] = new_rows
    assert len(results) == len(before_results) + 9

    readme_path = TABLES / 'README.zh.md'
    readme = readme_path.read_text(encoding='utf-8')
    readme = replace_once(readme, f'{len(before_results)} 个空白结果槽位',
                         f'{len(results)} 个空白结果槽位')
    readme = replace_once(readme, '- WK3l 仅 EN_F→FR，是单源外部验证，不支撑多源排序结论。',
                         '- WK3l 仅 EN_F→FR，是单源外部验证，不支撑多源排序结论。\n'
                         '- WK3l 子表已补入 LSMGA、DMKGC、IMKGC 的 9 个指标槽位；等待 WK3l 适配和本地三种子复现，当前保留占位符，未新增训练任务。')
    changes = {
        scope_path: scope,
        table_path: table,
        TABLES / 'cells_template.csv': csv_text(template_fields, template),
        TABLES / 'cells_results.csv': csv_text(result_fields, results),
        readme_path: readme,
    }
    backup = TABLES / 'build' / ('wk3l_rows_backup_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup.mkdir()
    for path in [*changes, TABLES / 'table_manifest.json', TABLES / 'validation_report.json', TABLES / 'KBS_Main_Text_Tables.pdf']:
        shutil.copy2(path, backup / path.name)
    for path, text in changes.items():
        atomic_text(path, text)

    sys.path.insert(0, str(SUITE))
    from table_scope import update_scope
    scope_summary = update_scope()
    assert scope_summary['awaiting_adaptation_slots'] == 9
    manifest_path = TABLES / 'table_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    filled = sum(row['value'] != '' for row in results)
    manifest.update(total_cells=len(results), filled_cells=filled,
                    placeholder_cells=len(results) - filled, baseline_scope=scope_summary)
    atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')

    os.environ['PATH'] = 'G:/texlive/2026/bin/windows' + os.pathsep + os.environ.get('PATH', '')
    compiler = Path('C:/Users/evan/.codex/plugins/cache/openai-bundled/latex/0.2.6/scripts/compile_latex.py')
    built = subprocess.run([sys.executable, str(compiler), str(TABLES / 'main.tex'),
                            '--compiler', 'texlive', '--engine', 'xelatex',
                            '--output-directory', str(TABLES / 'build'), '--json'],
                           cwd=TABLES, capture_output=True, text=True, encoding='utf-8', errors='replace')
    atomic_text(TABLES / 'build/wk3l_rows_build.log', built.stdout + '\n' + built.stderr)
    assert built.returncode == 0, 'LaTeX compilation failed; see build/wk3l_rows_build.log'
    shutil.copy2(TABLES / 'build/main.pdf', TABLES / 'KBS_Main_Text_Tables.pdf')
    subprocess.run([sys.executable, str(TABLES / 'validate_current.py'), '--render'], check=True, cwd=TABLES)
    validation = json.loads((TABLES / 'validation_report.json').read_text(encoding='utf-8'))
    readme = re.sub(r'重新排版的 \d+ 页正文表格预览',
                    f"重新排版的 {validation['pages']} 页正文表格预览", readme)
    atomic_text(readme_path, readme)
    from package_tables import publish
    publish()

    _, after_rows = read_csv(TABLES / 'cells_results.csv')
    after = {row['cell_id']: row for row in after_rows}
    assert all(after[key] == value for key, value in before_results.items())
    assert all(after[key]['value'] == '' and after[key]['source_type'] == 'pending' for key in NEW_IDS)
    assert before_values == (TABLES / 'values.tex').read_bytes()
    report = {'added_rows': ['LSMGA', 'DMKGC', 'IMKGC'], 'dataset': 'WK3l-15k EN_F->FR',
              'new_pending_cells': NEW_IDS, 'total_cells': len(after), 'existing_cells_preserved': True,
              'filled_cells': filled, 'pages': validation['pages'], 'layout_passed': validation['passed'],
              'backup': str(backup), 'pdf': str(TABLES / 'KBS_Main_Text_Tables.pdf')}
    atomic_text(TABLES / 'build/wk3l_rows_edit_receipt.json', json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


with (RUN / 'results/collector.lock').open('r+b') as lock:
    deadline = time.monotonic() + 45
    while True:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError('Collector is busy; no table changes were applied')
            time.sleep(1)
    try:
        edit_and_build()
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
