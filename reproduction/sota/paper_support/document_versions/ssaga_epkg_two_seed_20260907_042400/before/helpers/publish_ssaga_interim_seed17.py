"""Publish explicitly provisional seed17 values without fabricating seed SD."""
import argparse
import csv
import datetime
import json
import shutil
from pathlib import Path
from audit_main_completion import BASE, ROOT, read, save, sha
from refresh_table2_highlights import refresh_highlights

PAPER = ROOT / 'reproduction/sota/paper_support'
TABLES = ROOT / 'outputs/kbs/_main/_tables'


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    revision = parser.parse_args().archive.resolve()
    assert revision.is_relative_to((PAPER / 'document_versions').resolve())
    assert sha(revision/'before/tables/cells_results.csv') == sha(TABLES/'cells_results.csv')
    summary = read(BASE/'INTERIM_SEED17_RESULTS.json')
    assert summary['plan_sha256'] == sha(BASE/'PLAN.json')
    assert summary['audit_script_sha256'] == sha(ROOT/'reproduction/sota/audit_main_completion.py')
    assert summary['interim_script_sha256'] == sha(ROOT/'reproduction/sota/audit_ssaga_interim_seed17.py')
    group = summary['groups'][0]
    assert group['status'] == 'interim_single_seed' and group['seeds'] == [17]
    assert group['planned_seeds'] == [17, 29, 43] and group['sample_sd'] is None
    for run in group['runs']:
        assert read(BASE/'publication_acceptance'/(run['run_id']+'.json')) == run and run['passed']
        result = ROOT/run['result_path']
        assert sha(result) == run['result_sha256']
        assert sha(result.with_name('config.json')) == run['config_sha256']
        assert sha(result.with_name('best.pt')) == run['checkpoint_sha256']
        for raw in run['raw_artifacts']:
            assert sha(ROOT/raw['path']) == raw['sha256']
    sources = revision/'source_results'
    sources.mkdir()
    for name in ['INTERIM_SEED17_RESULTS.json', 'PUBLICATION_SOURCE_RECEIPT.json']:
        shutil.copy2(BASE/name, sources/name)
    (sources/'acceptance').mkdir()
    for run in group['runs']:
        shutil.copy2(BASE/'publication_acceptance'/(run['run_id']+'.json'), sources/'acceptance'/(run['run_id']+'.json'))
    frozen = sources/'INTERIM_SEED17_RESULTS.json'
    fields, rows = csv_rows(TABLES/'cells_results.csv')
    original = {r['cell_id']: r.copy() for r in rows}
    updates = {}
    for row in rows:
        if not row['cell_id'].startswith('T3.epkg.ssaga.'):
            continue
        assert not row['value']
        metric = row['cell_id'].rsplit('.', 1)[1]
        is_label = metric == 'paper_table'
        row.update(value='17 (1/3)' if is_label else str(group['mean'][metric]), standard_deviation='',
                   seed='17', source_type='rerun', run_id=';'.join(r['run_id'] for r in group['runs']),
                   unit='seed identifier and completion count' if is_label else 'fraction displayed as percent',
                   dataset_hash=';'.join(read((ROOT/r['result_path']).with_name('config.json'))['dataset_hash'] for r in group['runs']),
                   checkpoint=';'.join(Path(r['result_path']).with_name('best.pt').as_posix() for r in group['runs']),
                   split='public test; validation-only checkpoint selection',
                   filter_protocol='train; method-specific SS-AGA protocol',
                   selection_protocol='earliest maximum val_select MRR; frozen final test reused',
                   candidate_scope='all target-KG entities; ascending entity-ID ties',
                   aggregation=group['aggregation'],
                   feature_policy='frozen regenerated mBERT label features; supporter train+public validation facts',
                   code_commit='main-completion-plan-sha256:'+summary['plan_sha256'],
                   paper_table_cell=row['cell_id'],
                   notes='INTERIM_SINGLE_SEED: seed17 is complete across all six target KGs. Seeds29/43 are pending; no sample SD or three-seed completion is claimed. Replace with the registered three-seed statistic when complete; preserve this version.')
        assert row['placeholder_kind'] == ('TXT' if is_label else 'V')
        updates[row['cell_id']] = (row, metric)
    assert len(updates) == 4
    assert all(row == original[row['cell_id']] for row in rows if row['cell_id'] not in updates)
    with (TABLES/'cells_results.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    fields, template = csv_rows(TABLES/'cells_template.csv')
    for row in template:
        if row['cell_id'] in updates:
            for key in ['unit', 'paper_table_cell']:
                row[key] = updates[row['cell_id']][0][key]
    with (TABLES/'cells_template.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(template)
    values = (TABLES/'values.tex').read_text(encoding='utf-8').splitlines()
    approved = read(PAPER/'APPROVED_BASELINE_UPDATES.json')
    for cid, (row, metric) in updates.items():
        shown = row['value'] if metric == 'paper_table' else r'\('+f"{float(row['value'])*100:.2f}"+r'\)'
        assert not any(v.startswith('\\SetResult{'+cid+'}') for v in values)
        values.append('\\SetResult{'+cid+'}{'+shown+'}')
        approved['cells'][cid] = row
        approved['statistics_sources'][cid] = {'path': frozen.relative_to(ROOT).as_posix(),
            'sha256': sha(frozen), 'method': 'SS-AGA', 'dataset': 'depkg', 'metric': metric,
            'status': 'interim_single_seed'}
    (TABLES/'values.tex').write_text('\n'.join(values)+'\n', encoding='utf-8')
    path = TABLES/'tables/t03.tex'
    text = path.read_text(encoding='utf-8')
    old = 'Values are local three-seed mean $\\pm$ sample SD; method-specific settings are not ranked against Table~\\ref{tab:2}.'
    new = 'Completed rows report three-seed mean $\\pm$ sample SD (17, 29, 43). SS-AGA on E-PKG reports seed 17 only (1/3); seeds 29/43 and the sample SD are pending. These method-specific settings are not ranked against Table~\\ref{tab:2}.'
    assert text.count(old) == 1
    text = text.replace(old, new).replace(' Unfilled cells await local reruns.', '')
    path.write_text(text, encoding='utf-8')
    (revision/'published').mkdir()
    shutil.copy2(path, revision/'published/t03.tex')
    approved['table_sources']['tables/t03.tex'] = {'sha256': sha(path),
        'archived_source': (revision/'published/t03.tex').relative_to(ROOT).as_posix(),
        'previous_sha256': sha(revision/'before/tables/tables/t03.tex'),
        'authorization': 'User requested checking and filling the latest returned results; single-seed status explicitly displayed',
        'changed_cells': list(updates), 'revision': revision.relative_to(ROOT).as_posix()}
    approved.setdefault('revisions', []).append(revision.relative_to(ROOT).as_posix())
    save(PAPER/'APPROVED_BASELINE_UPDATES.json', approved)
    previous_ranks = read(TABLES/'table2_highlights.json')['columns']
    assert refresh_highlights()['columns'] == previous_ranks
    manifest = read(TABLES/'table_manifest.json')
    manifest.update(filled_cells=301, placeholder_cells=0, interim_cells=list(updates),
                    pending_three_seed_cells=list(updates), all_registered_three_seed_groups_complete=False,
                    latest_baseline_revision=revision.relative_to(ROOT).as_posix(),
                    task_scope='Table2 and completed Table3 rows frozen; SS-AGA/E-PKG seed17 displayed as interim, awaiting seeds29/43')
    manifest['single_run_groups'] = [{'method': 'SS-AGA', 'dataset': 'depkg', 'condition': 'full', 'seeds': [17],
                                     'planned_seeds': [17, 29, 43], 'status': 'interim_single_seed'}]
    comparison = read(BASE/'CURRENT_COMPARISON.json')
    comparison['csv_sha256'] = sha(TABLES/'cells_results.csv')
    comparison['remaining_table3_cells'] = list(updates)
    comparison['remaining_table3_cells_reason'] = 'Displayed seed17 interim values; registered three-seed statistics are pending'
    comparison['latest_interim_results_sha256'] = sha(frozen)
    save(BASE/'CURRENT_COMPARISON.json', comparison)
    manifest['current_comparison_sha256'] = sha(BASE/'CURRENT_COMPARISON.json')
    save(TABLES/'table_manifest.json', manifest)
    old_status = 'Table 3 仅 SS-AGA/E-PKG 尚待原独立队列完成。'
    new_status = 'Table 3 的 SS-AGA/E-PKG 已完成 seed17 的六个 KG，单种子宏平均为 28.84 / 17.07 / 51.82（MRR / H@1 / H@10，%），当前明确标注为 1/3 的暂值，不提供跨种子标准差；正式三种子结果仍待原独立队列完成 seed29/43。'
    idea = ROOT/'work/ideaspark/_run/multidomain-kgc-local/_2/phase4'
    for doc in [idea/'idea.std.zh.md', idea/'idea.std.zh.tex', TABLES/'EXPERIMENT_DESIGN.zh.md']:
        text = doc.read_text(encoding='utf-8')
        assert text.count(old_status) == 1
        text = text.replace(old_status, new_status.replace('%', r'\%') if doc.suffix == '.tex' else new_status)
        if doc.name == 'EXPERIMENT_DESIGN.zh.md':
            text += '\n## 2026-09-07 SS-AGA/E-PKG 单种子进度回填\n\n27/39 项已完成并回传。新完成的 seed17 六个目标 KG 经过原始排名、验证选模、检查点、固定输入与代码哈希核验，按六个 KG 等权平均后报告 28.84 / 17.07 / 51.82（%）。该行是明确标注 1/3 的单种子暂值，不能当作三种子结果；seed29 六项运行、seed43 六项排队。现有 Table 2 和其余已发布数值保持不变，不重选配置、不重复测试。证据：reproduction/language_table_seed17/main_completion_20260907/INTERIM_SEED17_RESULTS.json 与 publication_acceptance/ssaga_depkg_*_s17.json。\n'
        doc.write_text(text, encoding='utf-8')
    legacy = ROOT/'work/ideaspark_run/multidomain-kgc-local_2/phase4'
    for ext in ['md', 'tex']:
        shutil.copy2(idea/f'idea.std.zh.{ext}', legacy/f'idea.std.zh.{ext}')
    audit = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'updated_cells': list(updates),
             'status': 'interim_single_seed', 'three_seed_complete': False, 'seeds': [17], 'pending_seeds': [29, 43],
             'all_prior_filled_rows_unchanged': True, 'table2_all_cells_and_ranks_unchanged': True,
             'new_training_or_inference': False, 'summary_path': frozen.relative_to(ROOT).as_posix(),
             'summary_sha256': sha(frozen), 'archive': revision.relative_to(ROOT).as_posix()}
    save(revision/'IMPORT_AUDIT.json', audit)
    save(PAPER/'LATEST_BASELINE_IMPORT.json', audit)
    print(json.dumps({'updated_cells': list(updates), 'display_percent': group['display_percent'], 'three_seed_complete': False}))


if __name__ == '__main__':
    main()
