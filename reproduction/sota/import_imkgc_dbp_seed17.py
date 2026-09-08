"""Publish the requested interim single-seed values from audited saved results."""
import csv
import datetime
import json
import shutil
from pathlib import Path
from frozen_data import ROOT, atomic_json, sha256

base = ROOT / 'reproduction/sota'
tables = ROOT / 'outputs/kbs/_main/_tables'
phase = base / 'paper_support'
job = ROOT / 'reproduction/runs/strict_baselines_20260905/jobs/imkgc_dbp5l_s17'
stamp = datetime.datetime.now(datetime.timezone.utc)
revision = phase / 'document_versions' / ('imkgc_dbp_seed17_' + stamp.strftime('%Y%m%d_%H%M%S'))
revision.mkdir(parents=True)


def csvread(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


for name in ('values.tex', 'cells_results.csv', 'cells_template.csv', 'table_manifest.json',
             'validation_report.json', 'KBS_Main_Text_Tables.pdf'):
    shutil.copy2(tables / name, revision / name)
shutil.copy2(tables / 'tables/t02.tex', revision / 't02.tex')
for name in ('APPROVED_BASELINE_UPDATES.json', 'LATEST_BASELINE_IMPORT.json'):
    shutil.copy2(phase / name, revision / name)
shutil.copy2(base / 'sync_support_documents.py', revision / 'sync_support_documents.py')

acceptance = base / 'graph_three_seed/raw_acceptance/imkgc_dbp5l_s17.json'
verified = json.loads(acceptance.read_text())
result = json.loads((job / 'result.json').read_text())
assert verified['passed'] and verified['seed'] == result['seed'] == 17
assert result['status'] == 'completed' and result['purpose'] == 'formal'
assert sha256(job / 'best.pt') == verified['checkpoint_sha256'] == result['checkpoint_sha256']
assert sha256(job / 'config.json') == verified['config_sha256']
source_hashes = {acceptance.relative_to(ROOT).as_posix(): sha256(acceptance)}
source_dir = revision / 'source_results'
source_dir.mkdir()
for name in ('result.json', 'config.json'):
    shutil.copy2(job / name, source_dir / name)
    source_hashes[(job / name).relative_to(ROOT).as_posix()] = sha256(job / name)
shutil.copy2(acceptance, source_dir / 'raw_acceptance.json')
for item in verified['raw_artifacts']:
    path = job / item['kg'] / (item['split'] + '_queries.npz')
    assert sha256(path) == item['sha256']
    source_hashes[path.relative_to(ROOT).as_posix()] = item['sha256']
    if item['split'] == 'test':
        dest = source_dir / item['kg']
        dest.mkdir()
        shutil.copy2(path, dest / path.name)

fields, rows = csvread(tables / 'cells_results.csv')
source = ROOT / 'outputs/kbs_main_tables/cells_results.csv'
_, local_rows = csvread(source)
local = {row['cell_id']: row for row in local_rows}
lookup = {row['cell_id']: row.copy() for row in rows}
changed = []
for metric in ('mrr', 'h1', 'h10'):
    cid = 'T2.dbp.imkgc.' + metric
    assert not lookup[cid]['value'] and lookup[cid]['placeholder_kind'] == 'V'
    row = local[cid].copy()
    assert row['seed'] == '17' and row['standard_deviation'] == '' and row['placeholder_kind'] == 'V'
    assert row['unit'] == 'fraction displayed as percent' and row['filter_protocol'] == 'train+valid'
    assert float(row['value']) == result['macro'][metric] == verified['macro'][metric]
    row['notes'] = 'Audited saved ranks and checkpoints; interim seed17 result requested by user. Registered seeds29/43 are running; sample SD will be added after all three seeds complete.'
    lookup[cid] = row
    changed.append(cid)
assert {row['cell_id'] for row in rows if row != lookup[row['cell_id']]} == set(changed)
with (tables / 'cells_results.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(lookup[row['cell_id']] for row in rows)
values = (tables / 'values.tex').read_text(encoding='utf-8')
for cid in changed:
    assert '\\SetResult{' + cid + '}' not in values
    values += '\\SetResult{' + cid + '}{\\(' + f"{100 * float(lookup[cid]['value']):.2f}" + '\\)}\n'
(tables / 'values.tex').write_text(values, encoding='utf-8')

approved = json.loads((phase / 'APPROVED_BASELINE_UPDATES.json').read_text())
approved['cells'].update({cid: lookup[cid] for cid in changed})
approved['revisions'].append(revision.relative_to(ROOT).as_posix())
atomic_json(phase / 'APPROVED_BASELINE_UPDATES.json', approved)
manifest = json.loads((tables / 'table_manifest.json').read_text())
manifest.update(filled_cells=sum(bool(row['value']) for row in lookup.values()),
                placeholder_cells=sum(not row['value'] for row in lookup.values()),
                latest_baseline_revision=revision.relative_to(ROOT).as_posix(),
                task_scope='Graph baseline repetitions running; user-authorized interim seed17 Table2 import; ASRC and supporting experiment results frozen')
group = {'method': 'IMKGC', 'dataset': 'dbp5l', 'condition': 'full', 'seeds': [17]}
if group not in manifest['single_run_groups']:
    manifest['single_run_groups'].append(group)
atomic_json(tables / 'table_manifest.json', manifest)
audit = {'timestamp': stamp.isoformat(), 'purpose': 'User requested immediate interim PDF fill and compile',
         'newly_filled_cells': changed, 'source_csv': source.relative_to(ROOT).as_posix(),
         'source_hashes': source_hashes, 'raw_acceptance_reused': acceptance.relative_to(ROOT).as_posix(),
         'raw_artifact_hashes_reverified': True, 'macro': result['macro'], 'seed': 17,
         'single_seed_label_retained': True, 'standard_deviation_added': False,
         'unrounded_values_preserved': True, 'new_experiments': 0,
         'all_other_rows_unchanged': True, 'archive': revision.relative_to(ROOT).as_posix()}
atomic_json(revision / 'IMPORT_AUDIT.json', audit)
atomic_json(phase / 'LATEST_BASELINE_IMPORT.json', audit)
print(json.dumps({'filled': changed, 'archive': audit['archive'], 'filled_cells': manifest['filled_cells']}))
