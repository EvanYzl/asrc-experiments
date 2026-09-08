"""Publish accepted complete graph-baseline groups while preserving other cells."""
import argparse
import csv
import datetime
import json
import re
import shutil
from pathlib import Path

import numpy as np
from frozen_data import ROOT, atomic_json, sha256
from summarize_graph_repeats import accepted_seed
from refresh_table2_highlights import HIGHLIGHT_NOTE, refresh_highlights

BASE = ROOT / 'reproduction/sota'
PHASE = BASE / 'graph_three_seed'
PAPER = BASE / 'paper_support'
TABLES = ROOT / 'outputs/kbs/_main/_tables'
DATASET_KEYS = {'dbp5l': 'dbp', 'depkg': 'epkg', 'dwy': 'dwy'}
METRICS = ('mrr', 'h1', 'h10')


def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def writecsv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(row):
    return r'\(' + f"{100 * float(row['value']):.2f}" + r'\mathbin{\pm}' + f"{100 * float(row['standard_deviation']):.2f}" + r'\)'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    source = PHASE / ('PARTIAL_RESULTS.json' if args.allow_partial else 'RESULTS.json')
    summary = json.loads(source.read_text())
    assert summary['manifest_sha256'] == sha256(PHASE / 'manifest.json')
    assert summary['source_input_freeze_sha256'] == sha256(PHASE / 'SOURCE_INPUT_FREEZE.json')
    assert summary['summary_script_sha256'] == sha256(BASE / 'summarize_graph_repeats.py')
    if not args.allow_partial:
        assert summary['status'] == 'complete' and summary['complete_groups'] == 9
    groups = [g for g in summary['groups'] if g['status'] == 'complete']
    assert groups
    fields, original = readcsv(TABLES / 'cells_results.csv')
    lookup = {r['cell_id']: r.copy() for r in original}
    updates, selected = {}, []
    for group in groups:
        method, dataset = group['method'], group['dataset']
        assert method in ('LSMGA', 'DMKGC', 'IMKGC') and dataset in DATASET_KEYS
        assert group['seeds'] == [17, 29, 43]
        runs = [accepted_seed(method, dataset, seed) for seed in group['seeds']]
        for current, recorded in zip(runs, group['runs']):
            for key in ('run_id', 'macro', 'receipt_sha256', 'result_sha256', 'config_sha256', 'checkpoint_sha256'):
                assert current[key] == recorded[key], (current['run_id'], key)
        for metric in METRICS:
            assert float(np.mean([r['macro'][metric] for r in runs])) == group['mean'][metric]
            assert float(np.std([r['macro'][metric] for r in runs], ddof=1)) == group['sample_sd'][metric]
        ids = {metric: f'T2.{DATASET_KEYS[dataset]}.{method.lower()}.{metric}' for metric in METRICS}
        if all(lookup[cid]['placeholder_kind'] == 'PM' and lookup[cid]['seed'] == '17;29;43'
               and float(lookup[cid]['value']) == group['mean'][metric]
               and float(lookup[cid]['standard_deviation']) == group['sample_sd'][metric]
               for metric, cid in ids.items()):
            continue
        selected.append(group)
        for metric, cid in ids.items():
            row = lookup[cid].copy()
            row.update(placeholder_kind='PM', value=str(group['mean'][metric]),
                       standard_deviation=str(group['sample_sd'][metric]), seed='17;29;43',
                       unit='fraction displayed as percent', source_type='rerun',
                       run_id=';'.join(r['run_id'] for r in runs),
                       checkpoint=';'.join(str(Path(r['result_path']).with_name('best.pt')).replace('\\', '/') for r in runs),
                       code_commit='source-input-freeze-sha256:' + summary['source_input_freeze_sha256'],
                       aggregation='equal KG macro per seed; mean over seeds 17/29/43; sample SD ddof=1',
                       selection_protocol='val_select tail MRR, earliest maximum; one saved test per seed',
                       paper_table_cell=cid,
                       notes='Accepted full-query ranks and best/last checkpoints; local seed17 reused with frozen server repeats29/43. Full-precision statistics and source receipts archived with the approved baseline update.')
            updates[cid] = row
            lookup[cid] = row
    assert updates, 'All complete groups are already published'
    stamp = datetime.datetime.now(datetime.timezone.utc)
    revision = args.archive or PAPER / 'document_versions' / ('graph_three_seed_' + stamp.strftime('%Y%m%d_%H%M%S'))
    assert revision.resolve().is_relative_to((PAPER / 'document_versions').resolve())
    if not revision.exists():
        revision.mkdir(parents=True)
        shutil.copytree(TABLES, revision / 'before/tables', ignore=shutil.ignore_patterns('build'))
        (revision / 'before/helpers').mkdir()
        for name in ('publish_support_design.py', 'validate_support_documents.py', 'sync_support_documents.py'):
            shutil.copy2(BASE / name, revision / 'before/helpers' / name)
        for name in ('APPROVED_BASELINE_UPDATES.json', 'LATEST_BASELINE_IMPORT.json'):
            shutil.copy2(PAPER / name, revision / 'before' / name)
    assert (revision / 'before/tables/cells_results.csv').read_bytes() == (TABLES / 'cells_results.csv').read_bytes()
    source_dir = revision / 'source_results'
    source_dir.mkdir()
    frozen_summary = source_dir / source.name
    shutil.copy2(source, frozen_summary)
    for group in selected:
        for run in group['runs']:
            result = ROOT / run['result_path']
            for item, label in ((result, 'result'), (result.with_name('config.json'), 'config'),
                                (ROOT / run['receipt_path'], 'acceptance')):
                shutil.copy2(item, source_dir / (run['run_id'] + '.' + label + '.json'))
    assert {r['cell_id'] for r in original if r != lookup[r['cell_id']]} == set(updates)
    writecsv(TABLES / 'cells_results.csv', fields, [lookup[r['cell_id']] for r in original])
    template_fields, template_rows = readcsv(TABLES / 'cells_template.csv')
    template_cells = json.loads((PAPER / 'APPROVED_BASELINE_UPDATES.json').read_text())['cells'] | updates
    for row in template_rows:
        if row['cell_id'] in template_cells:
            for key in ('placeholder_kind', 'unit', 'paper_table_cell'):
                row[key] = template_cells[row['cell_id']][key]
    writecsv(TABLES / 'cells_template.csv', template_fields, template_rows)
    lines = (TABLES / 'values.tex').read_text(encoding='utf-8').splitlines()
    for cid, row in updates.items():
        lines = [line for line in lines if not line.startswith('\\SetResult{' + cid + '}')]
        lines.append('\\SetResult{' + cid + '}{' + render(row) + '}')
    (TABLES / 'values.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    highlights_enabled = (TABLES / 'table2_highlights.tex').exists()
    if highlights_enabled:
        refresh_highlights()
    table2 = TABLES / 'tables/t02.tex'
    text = table2.read_text(encoding='utf-8')
    for cid in updates:
        assert text.count('\\V{' + cid + '}') == 1
        text = text.replace('\\V{' + cid + '}', '\\PM{' + cid + '}')
    single_rows = []
    for method in ('LSMGA', 'DMKGC', 'IMKGC'):
        has_single = any(r['value'] and r['placeholder_kind'] == 'V'
                         for r in lookup.values()
                         if re.fullmatch(r'T2\.(dbp|epkg|dwy)\.' + method.lower() + r'\.(mrr|h1|h10)', r['cell_id']))
        if has_single:
            single_rows.append(method)
        else:
            text = text.replace(method + r'$^{\dagger}$ &', method + ' &')
    note = (r'Completed entries with $\pm$ report mean and sample SD over seeds 17, 29 and 43. '
            r'Each run uses its fixed validation-selected checkpoint and one test; already locked tests are reused. ')
    if highlights_enabled:
        note = HIGHLIGHT_NOTE + note
    if single_rows:
        note += r'In rows marked $\dagger$, filled entries without $\pm$ are single runs (seed 17). '
    note += (r'Full target candidates, tail prediction and ascending entity-ID ties are fixed. '
             r'Core results use train+valid filtering and equal KG macro averages; WK3l evaluates FR with all filtering. '
             r'User-supplied approximate references, unknown graph baselines and unmatched repetitions keep the comparison provisional. '
             r'Pending cells are not treated as defeated.')
    text, count = re.subn(r'(?m)^\\tnote\{.*\}$', lambda _: '\\tnote{' + note + '}', text)
    assert count == 1
    table2.write_text(text, encoding='utf-8')
    published = revision / 'published'
    published.mkdir()
    shutil.copy2(table2, published / 't02.tex')
    approved = json.loads((PAPER / 'APPROVED_BASELINE_UPDATES.json').read_text())
    approved['cells'].update(updates)
    approved.setdefault('revisions', []).append(revision.relative_to(ROOT).as_posix())
    approved.setdefault('table_sources', {})['tables/t02.tex'] = {
        'sha256': sha256(table2), 'archived_source': (published / 't02.tex').relative_to(ROOT).as_posix(),
        'previous_sha256': sha256(revision / 'before/tables/tables/t02.tex'),
        'authorization': 'User requested immediate publication of completed three-seed results',
        'changed_cells': sorted(updates), 'revision': revision.relative_to(ROOT).as_posix()}
    for group in selected:
        for metric in METRICS:
            cid = f"T2.{DATASET_KEYS[group['dataset']]}.{group['method'].lower()}.{metric}"
            approved.setdefault('statistics_sources', {})[cid] = {
                'path': frozen_summary.relative_to(ROOT).as_posix(), 'sha256': sha256(frozen_summary),
                'method': group['method'], 'dataset': group['dataset'], 'metric': metric}
    atomic_json(PAPER / 'APPROVED_BASELINE_UPDATES.json', approved)
    manifest = json.loads((TABLES / 'table_manifest.json').read_text())
    completed_keys = {(g['method'], g['dataset']) for g in groups}
    manifest['single_run_groups'] = [g for g in manifest.get('single_run_groups', [])
                                     if (g['method'], g['dataset']) not in completed_keys]
    for group in groups:
        entry = {'method': group['method'], 'dataset': group['dataset'], 'condition': 'full', 'seeds': [17, 29, 43]}
        if entry not in manifest.setdefault('three_seed_groups', []):
            manifest['three_seed_groups'].append(entry)
    manifest.update(total_cells=len(original), filled_cells=sum(bool(r['value']) for r in lookup.values()),
                    placeholder_cells=sum(not r['value'] for r in lookup.values()),
                    latest_baseline_revision=revision.relative_to(ROOT).as_posix(),
                    graph_baseline_complete_three_seed_groups=len(groups), graph_baseline_total_groups=9,
                    task_scope='Publish accepted complete graph-baseline groups; remaining registered repeats continue; ASRC and supporting results frozen')
    atomic_json(TABLES / 'table_manifest.json', manifest)
    audit = {'timestamp': stamp.isoformat(), 'purpose': 'User-requested immediate three-seed Table2 update',
             'updated_cells': sorted(updates), 'groups': [{k: g[k] for k in ('method', 'dataset', 'seeds', 'mean', 'sample_sd')} for g in selected],
             'all_other_rows_unchanged': True, 'full_precision_preserved': True,
             'saved_artifacts_reverified': True, 'new_training_or_inference': False,
             'summary_path': frozen_summary.relative_to(ROOT).as_posix(), 'summary_sha256': sha256(frozen_summary),
             'archive': revision.relative_to(ROOT).as_posix(), 'table2_source_sha256': sha256(table2)}
    atomic_json(revision / 'IMPORT_AUDIT.json', audit)
    atomic_json(PAPER / 'LATEST_BASELINE_IMPORT.json', audit)
    for name in ('cells_results.csv', 'cells_template.csv', 'values.tex', 'table_manifest.json'):
        shutil.copy2(TABLES / name, published / name)
    print(json.dumps({'updated_cells': sorted(updates), 'complete_groups': len(groups), 'archive': audit['archive']}))


if __name__ == '__main__':
    main()
