"""Move the requested internal controls from Table 2 to unfilled Table 4."""
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
DATASETS = ['dbp', 'epkg', 'dwy', 'wk3l']


def replace_once(text, before, after):
    assert text.count(before) == 1, f'Edit anchor changed: {before!r}'
    return text.replace(before, after, 1)


def atomic_text(path, text):
    temporary = path.with_name(path.name + '.controls-edit.tmp')
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
    run_manifest = json.loads((RUN / 'manifest.json').read_text(encoding='utf-8'))
    assert 'T4' not in run_manifest.get('active_tables', []), 'Table 4 is currently being filled'
    table2_path = TABLES / 'tables/t02.tex'
    table4_path = TABLES / 'tables/t04.tex'
    table2 = table2_path.read_text(encoding='utf-8')
    table4 = table4_path.read_text(encoding='utf-8')
    removed_lines = [line for line in table2.splitlines(keepends=True)
                     if line.startswith(('Target-only &', 'Uniform-All &', '$\\Delta$MRR vs. Uniform-All &'))]
    assert len(removed_lines) == 6, 'Expected two control rows and one effect row in each Table 2 panel'
    removed_ids = {ident for line in removed_lines for _, ident in re.findall(r'\\(PM|V|CI)\{([^}]+)\}', line)}
    assert len(removed_ids) == 32
    table2 = ''.join(line for line in table2.splitlines(keepends=True) if line not in removed_lines)
    paired_note = r' Paired MRR effects use pointwise 95\% hierarchical-bootstrap CIs; non-inferiority requires the lower bound above $-0.5$ pp.'
    table2 = replace_once(table2, paired_note, '')
    assert 'Target-only' not in table2 and 'Uniform-All' not in table2
    for ds in DATASETS:
        target = 'Target-only & ' + r'\PM{' + f'T4.{ds}.target.mrr' + '}' + ' & {---}' * 5 + r' \\' + '\n'
        anchor = 'Uniform-All & ' + r'\PM{' + f'T4.{ds}.uniform.mrr' + '}'
        table4 = replace_once(table4, anchor, target + anchor)
        effect = (r'$\Delta$MRR vs. Uniform-All & \multicolumn{6}{l}{\V{'
                  + f'T4.{ds}.delta_mrr' + r'} pp; paired 95\% CI \CI{'
                  + f'T4.{ds}.mrr_ci' + r'}} \\' + '\n')
        ntr_anchor = r'$\Delta$NTR vs. Uniform-All & \multicolumn{6}{l}{\V{' + f'T4.{ds}.delta_ntr' + '}'
        table4 = replace_once(table4, ntr_anchor, effect + ntr_anchor)
    table4 = replace_once(table4,
        'Rates and mean harm are percentages; edges/query is a count.',
        'Rates and mean harm are percentages; edges/query is a count. '
        'Target-only is the accuracy reference; dashes mark source-transfer metrics not reported for this reference.')
    table4 = replace_once(table4,
        'Paired NTR CIs are pointwise; zero coverage is not evidence of useful transfer.',
        r'Paired MRR and NTR CIs are pointwise hierarchical-bootstrap intervals; MRR non-inferiority requires the lower bound above $-0.5$ pp. Zero coverage is not evidence of useful transfer.')

    scope_path = SUITE / 'table_scope.py'
    scope = scope_path.read_text(encoding='utf-8')
    scope = replace_once(scope,
        "        if method in ['qura','delta_mrr','mrr_ci']:\n"
        "            return excluded('QURA score or paired effect relative to Uniform-All')\n"
        "        if method == 'target':\n"
        "            return required(f'internal_base_{ds}', 'Target-only shared 128-dimensional baseline')\n"
        "        if method == 'uniform':\n"
        "            return required(f'internal_teacher_{ds}', 'Uniform-All shared frozen fusion baseline')\n",
        "        if method == 'qura':\n"
        "            return excluded('QURA completion score')\n")
    scope = replace_once(scope,
        "        if method in ['qura','delta_ntr','ntr_ci']:\n"
        "            return excluded('QURA negative transfer or paired effect')\n",
        "        if method in ['qura','delta_mrr','mrr_ci','delta_ntr','ntr_ci']:\n"
        "            return excluded('QURA score, negative transfer or paired effect')\n"
        "        if method == 'target':\n"
        "            return required(f'internal_base_{ds}', 'Target-only accuracy reference moved from Table 2; Table 4 remains unfilled')\n")
    ast.parse(scope)
    collector_path = SUITE / 'collect_tables.py'
    collector = collector_path.read_text(encoding='utf-8')
    collector = replace_once(collector,
        "        if method in ['AlignKGC','SS-AGA']:\n",
        "        if method in ['Target-only','Uniform-All']:\n"
        "            table='T4'\n"
        "        if method in ['AlignKGC','SS-AGA']:\n")
    ast.parse(collector)

    template_fields, before_template = read_csv(TABLES / 'cells_template.csv')
    result_fields, before_results = read_csv(TABLES / 'cells_results.csv')
    assert template_fields == result_fields
    before_by_id = {r['cell_id']: r for r in before_results}
    sources = {path: path.read_text(encoding='utf-8') for path in sorted((TABLES / 'tables').glob('*.tex'))}
    sources.update({table2_path: table2, table4_path: table4})
    slots = [(kind, ident) for text in sources.values() for kind, ident in re.findall(r'\\(PM|V|CI|N|TXT)\{([^}]+)\}', text)]
    assert len(slots) == len({ident for _, ident in slots})
    template_by_id = {r['cell_id']: r for r in before_template}
    all_ids = {ident for _, ident in slots}
    assert set(before_by_id) - all_ids == removed_ids
    added_ids = all_ids - set(before_by_id)
    assert len(added_ids) == 12
    template, results = [], []
    held_blank = set()
    for kind, ident in slots:
        base = dict.fromkeys(template_fields, '')
        base.update(cell_id=ident, table_id=ident.split('.')[0], placeholder_kind=kind)
        template.append(template_by_id.get(ident, base).copy())
        row = before_by_id.get(ident, base).copy()
        if ident.startswith('T4.') and ident.split('.')[2] in ['target', 'uniform', 'delta_mrr', 'mrr_ci']:
            row = base.copy()
            row.update(source_type='pending', notes='Moved internal controls and effects are intentionally unfilled by user request')
            held_blank.add(ident)
        elif ident in added_ids:
            row.update(source_type='pending')
        results.append(row)
    values_path = TABLES / 'values.tex'
    values = values_path.read_text(encoding='utf-8')
    retained_values = []
    for line in values.splitlines(keepends=True):
        match = re.match(r'\\SetResult\{([^}]+)\}', line)
        if not match or match.group(1) not in removed_ids | held_blank:
            retained_values.append(line)
    values = ''.join(retained_values)

    readme_path = TABLES / 'README.zh.md'
    readme = readme_path.read_text(encoding='utf-8')
    readme = replace_once(readme, f'{len(before_results)} 个空白结果槽位', f'{len(results)} 个空白结果槽位')
    readme = replace_once(readme,
        '| 2 | 四数据集严格补全主结果 | 安全门控是否保持精度？ |',
        '| 2 | 四数据集跨论文方法的严格补全主结果 | QURA 与外部基线在统一协议下表现如何？ |')
    readme = replace_once(readme,
        '统计检验已压缩：MRR 差值和置信区间放表2；NTR 差值和置信区间放表4。不再单列统计附表。',
        '统计检验已压缩：相对 Uniform-All 的 MRR、NTR 差值和置信区间统一放表4。不再单列统计附表。\n\n'
        'Table 2 只保留外部论文方法与 QURA；Target-only、Uniform-All 及其相关差值不再出现在表2。'
        '内部控制在 Table 4 中按 MRR 展示，暂不填写数值；已有逐查询结果和检查点仍保留在实验记录中。')
    changes = {table2_path: table2, table4_path: table4, scope_path: scope, collector_path: collector,
               TABLES / 'cells_template.csv': csv_text(template_fields, template),
               TABLES / 'cells_results.csv': csv_text(result_fields, results),
               values_path: values, readme_path: readme}
    backup = TABLES / 'build' / ('controls_move_backup_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup.mkdir()
    for path in [*changes, TABLES / 'table_manifest.json', TABLES / 'validation_report.json', TABLES / 'KBS_Main_Text_Tables.pdf']:
        shutil.copy2(path, backup / path.name)
    for path, text in changes.items():
        atomic_text(path, text)

    sys.path.insert(0, str(SUITE))
    from table_scope import update_scope
    scope_summary = update_scope()
    manifest_path = TABLES / 'table_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    filled = sum(r['value'] != '' for r in results)
    manifest.update(total_cells=len(results), filled_cells=filled, placeholder_cells=len(results) - filled,
                    baseline_scope=scope_summary)
    atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')

    os.environ['PATH'] = 'G:/texlive/2026/bin/windows' + os.pathsep + os.environ.get('PATH', '')
    compiler = Path('C:/Users/evan/.codex/plugins/cache/openai-bundled/latex/0.2.6/scripts/compile_latex.py')
    built = subprocess.run([sys.executable, str(compiler), str(TABLES / 'main.tex'),
                            '--compiler', 'texlive', '--engine', 'xelatex',
                            '--output-directory', str(TABLES / 'build'), '--json'],
                           cwd=TABLES, capture_output=True, text=True, encoding='utf-8', errors='replace')
    atomic_text(TABLES / 'build/controls_move_build.log', built.stdout + '\n' + built.stderr)
    assert built.returncode == 0, 'LaTeX compilation failed; see build/controls_move_build.log'
    shutil.copy2(TABLES / 'build/main.pdf', TABLES / 'KBS_Main_Text_Tables.pdf')
    subprocess.run([sys.executable, str(TABLES / 'validate_current.py'), '--render'], check=True, cwd=TABLES)
    validation = json.loads((TABLES / 'validation_report.json').read_text(encoding='utf-8'))
    readme = re.sub(r'重新排版的 \d+ 页正文表格预览',
                    f"重新排版的 {validation['pages']} 页正文表格预览", readme)
    atomic_text(readme_path, readme)
    from package_tables import publish
    publish()

    _, after_rows = read_csv(TABLES / 'cells_results.csv')
    after = {r['cell_id']: r for r in after_rows}
    assert all(after[ident] == row for ident, row in before_by_id.items() if ident not in removed_ids | held_blank)
    assert all(after[ident]['value'] == '' and after[ident]['source_type'] == 'pending' for ident in held_blank)
    assert not removed_ids.intersection(after)
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(TABLES / 'KBS_Main_Text_Tables.pdf') as document:
        texts = []
        for page in document:
            textpage = page.get_textpage()
            try:
                texts.append(textpage.get_text_bounded())
            finally:
                textpage.close()
            page.close()
    all_text = '\n'.join(texts)
    table2_text = re.split(r'Table\s+2\b', all_text, maxsplit=1)[1]
    table2_text = re.split(r'Table\s+3\b', table2_text, maxsplit=1)[0]
    assert 'Target-only' not in table2_text and 'Uniform-All' not in table2_text
    report = {'table2_internal_controls_removed': True, 'table4_controls_and_effects_unfilled': True,
              'unrelated_cells_preserved': True, 'total_cells': len(after), 'filled_cells': filled,
              'pages': validation['pages'], 'layout_passed': validation['passed'], 'backup': str(backup)}
    atomic_text(TABLES / 'build/controls_move_receipt.json', json.dumps(report, ensure_ascii=False, indent=2) + '\n')
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
                raise RuntimeError('Collector is busy; no changes applied')
            time.sleep(1)
    try:
        edit_and_build()
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
