"""Fill only newly completed and independently audited fixed-seed groups."""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import shutil
import numpy as np
from refresh_table2_highlights import refresh_highlights

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reproduction/language_table_seed17/main_completion_20260907'
PAPER = ROOT / 'reproduction/sota/paper_support'
TABLES = ROOT / 'outputs/kbs/_main/_tables'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def readcsv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    revision = args.archive.resolve()
    assert revision.is_relative_to((PAPER / 'document_versions').resolve())
    assert (revision / 'before/tables/cells_results.csv').read_bytes() == (TABLES / 'cells_results.csv').read_bytes()
    source = BASE / 'PUBLICATION_RESULTS.json'
    summary = read(source)
    assert summary['plan_sha256'] == sha(BASE / 'PLAN.json')
    assert summary['audit_script_sha256'] == sha(ROOT / 'reproduction/sota/audit_main_completion.py')
    groups = [g for g in summary['groups'] if g['status'] == 'complete']
    fields, rows = readcsv(TABLES / 'cells_results.csv')
    lookup = {r['cell_id']: r for r in rows}
    updates = {}
    interim_replacements = []
    for group in groups:
        assert group['seeds'] == [17, 29, 43]
        for run in group['runs']:
            receipt = read(BASE / 'publication_acceptance' / (run['run_id'] + '.json'))
            assert receipt == run and receipt['passed']
            assert receipt['audit_script_sha256'] == summary['audit_script_sha256']
            result = ROOT / run['result_path']
            for path, key in [(result, 'result_sha256'), (result.with_name('config.json'), 'config_sha256'), (result.with_name('best.pt'), 'checkpoint_sha256')]:
                assert sha(path) == run[key]
            for raw in run['raw_artifacts']:
                assert sha(ROOT / raw['path']) == raw['sha256']
        for metric in ('mrr', 'h1', 'h3', 'h10'):
            assert float(np.mean([group['per_seed'][str(s)][metric] for s in group['seeds']])) == group['mean'][metric]
            assert float(np.std([group['per_seed'][str(s)][metric] for s in group['seeds']], ddof=1)) == group['sample_sd'][metric]
        is_graph = group['dataset'] == 'wk3l'
        prefix = f"T2.wk3l.{group['method'].lower()}" if is_graph else f"T3.{'epkg' if group['dataset']=='depkg' else 'dbp5l'}.ssaga"
        metrics = ['mrr', 'h1', 'h10'] + ([] if is_graph else ['paper_table'])
        for metric in metrics:
            cid = prefix + '.' + metric
            old = lookup[cid]
            value = '17;29;43' if metric == 'paper_table' else str(group['mean'][metric])
            deviation = '' if metric == 'paper_table' else str(group['sample_sd'][metric])
            if old['value']:
                if old['seed']=='17' and old['notes'].startswith('INTERIM_SINGLE_SEED:'):
                    assert prefix=='T3.epkg.ssaga' and not old['standard_deviation']
                    previous=read(PAPER/'APPROVED_BASELINE_UPDATES.json')['statistics_sources'][cid]
                    previous_path=ROOT/previous['path'];assert sha(previous_path)==previous['sha256']
                    partial=next(g for g in read(previous_path)['groups'] if g['method']==group['method'] and g['dataset']==group['dataset'])
                    assert partial['status']=='interim_single_seed' and partial['seeds']==[17]
                    assert old['value']==('17 (1/3)' if metric=='paper_table' else str(partial['mean'][metric]))
                    if metric!='paper_table':assert partial['mean'][metric]==group['per_seed']['17'][metric]
                    interim_replacements.append(cid)
                else:
                    assert old['value'] == value and old['standard_deviation'] == deviation and old['seed'] == '17;29;43'
                    continue
            row = old.copy()
            row.update(value=value, standard_deviation=deviation, placeholder_kind='TXT' if metric == 'paper_table' else 'PM',
                       seed='17;29;43', unit='seed identifiers' if metric == 'paper_table' else 'fraction displayed as percent',
                       source_type='rerun', run_id=';'.join(r['run_id'] for r in group['runs']),
                       checkpoint=';'.join(str(Path(r['result_path']).with_name('best.pt')).replace('\\', '/') for r in group['runs']),
                       split='public test; validation-only selection', filter_protocol='all; FR target' if is_graph else 'train; method-specific SS-AGA protocol',
                       candidate_scope='all target-KG entities; ascending entity-ID ties',
                       aggregation=summary['aggregation'], selection_protocol='earliest maximum val_select MRR; one saved final test',
                       feature_policy='train-only structural graph and fixed supplied alignments' if is_graph else 'frozen regenerated mBERT label features; supporter train+public validation facts',
                       code_commit='main-completion-plan-sha256:' + summary['plan_sha256'], paper_table_cell=cid,
                       notes='Complete fixed seeds independently checked against raw test/selection ranks, validation-best checkpoints, source objects and returned archive hashes. SS-AGA results are reported only under Table3 method-specific settings.' if not is_graph else 'FR-only primary comparison; EN results retained as language supplement. Original seed17 reused; two server repeats use the same frozen recipe and host/path-only adaptations.')
            updates[cid] = (row, group, metric)
            lookup[cid] = row
    assert updates, 'No new complete group to publish'
    sources = revision / 'source_results'
    sources.mkdir()
    shutil.copy2(source, sources / 'PUBLICATION_RESULTS.json')
    shutil.copy2(BASE / 'PUBLICATION_SOURCE_RECEIPT.json', sources / 'PUBLICATION_SOURCE_RECEIPT.json')
    shutil.copytree(BASE / 'publication_acceptance', sources / 'acceptance')
    frozen_summary = sources / 'PUBLICATION_RESULTS.json'
    approved = read(PAPER / 'APPROVED_BASELINE_UPDATES.json')
    approved.setdefault('revisions', []).append(revision.relative_to(ROOT).as_posix())
    for cid, (row, group, metric) in updates.items():
        approved['cells'][cid] = row
        approved.setdefault('statistics_sources', {})[cid] = {'path': frozen_summary.relative_to(ROOT).as_posix(), 'sha256': sha(frozen_summary),
            'method': group['method'], 'dataset': group['dataset'], 'metric': metric}
    for name in ['cells_results.csv', 'cells_template.csv']:
        columns, content = readcsv(TABLES / name)
        for row in content:
            cid = row['cell_id']
            if cid not in updates:
                continue
            if name == 'cells_results.csv':
                row.update(updates[cid][0])
            else:
                for key in ('placeholder_kind', 'unit', 'paper_table_cell'):
                    row[key] = updates[cid][0][key]
        with (TABLES / name).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(content)
    values = (TABLES / 'values.tex').read_text(encoding='utf-8').splitlines()
    for cid, (row, _, metric) in updates.items():
        values = [line for line in values if not line.startswith('\\SetResult{' + cid + '}')]
        shown = row['value'] if metric == 'paper_table' else r'\(' + f"{100*float(row['value']):.2f}" + r'\mathbin{\pm}' + f"{100*float(row['standard_deviation']):.2f}" + r'\)'
        values.append('\\SetResult{' + cid + '}{' + shown + '}')
    (TABLES / 'values.tex').write_text('\n'.join(values) + '\n', encoding='utf-8')
    for number in (2, 3):
        path = TABLES / f'tables/t{number:02}.tex'
        text = path.read_text(encoding='utf-8')
        for cid, (_, _, metric) in updates.items():
            if metric != 'paper_table':
                text = text.replace('\\V{' + cid + '}', '\\PM{' + cid + '}')
        if number == 2:
            text = text.replace('The three unknown WK3l graph baselines keep the comparison provisional. Pending cells are not treated as defeated.',
                                'All listed Table 2 comparisons now use completed local three-seed runs under these fixed protocols. WK3l graph recipes and input adaptations were frozen before testing.')
        else:
            if any(cid.startswith('T3.epkg.ssaga.') for cid in updates):
                text=text.replace('Completed rows report three-seed mean $\\pm$ sample SD (17, 29, 43). SS-AGA on E-PKG reports seed 17 only (1/3); seeds 29/43 and the sample SD are pending.',
                                  'Values are local three-seed mean $\\pm$ sample SD (17, 29, 43).')
            text = text.replace(r'\centering\fontsize{8.5}{10.1}\selectfont', r'\centering\fontsize{8}{9.6}\selectfont' + '\n' + r'\setlength{\tabcolsep}{1.8pt}' + '\n' + r'\setlength{\medmuskip}{1mu}')
            text = text.replace(r'L{20mm}L{20mm}C C C L{28mm}L{22mm}L{16mm}', r'L{20mm}L{19mm}C C C L{24mm}L{22mm}L{18mm}')
        path.write_text(text, encoding='utf-8')
        archived = revision / 'published' / path.name
        archived.parent.mkdir(exist_ok=True)
        shutil.copy2(path, archived)
        approved['table_sources'][f'tables/t{number:02}.tex'] = {'sha256': sha(path), 'archived_source': archived.relative_to(ROOT).as_posix(),
            'previous_sha256': sha(revision / f'before/tables/tables/t{number:02}.tex'), 'authorization': 'User requested checking completed runs and filling PDF',
            'changed_cells': [cid for cid in updates if cid.startswith(f'T{number}.')], 'revision': revision.relative_to(ROOT).as_posix()}
    save(PAPER / 'APPROVED_BASELINE_UPDATES.json', approved)
    refresh_highlights()
    manifest = read(TABLES / 'table_manifest.json')
    for group in groups:
        entry = {'method': group['method'], 'dataset': group['dataset'], 'condition': 'full', 'seeds': [17, 29, 43]}
        if entry not in manifest['three_seed_groups']:
            manifest['three_seed_groups'].append(entry)
    manifest.update(filled_cells=sum(bool(x['value']) for x in lookup.values()), placeholder_cells=sum(not x['value'] for x in lookup.values()),
                    table2_complete=True, latest_baseline_revision=revision.relative_to(ROOT).as_posix(),
                    candidate_status='固定 Table2 全体比较项中三种子均值达到门槛',
                    task_scope='Publish complete audited groups; remaining SS-AGA/E-PKG jobs continue in their existing finite queue')
    if any(cid.startswith('T3.epkg.ssaga.') for cid in updates):
        manifest['single_run_groups']=[g for g in manifest.get('single_run_groups',[]) if (g['method'],g['dataset'])!=('SS-AGA','depkg')]
        manifest['interim_cells']=[];manifest['pending_three_seed_cells']=[]
        manifest['all_registered_three_seed_groups_complete']=True
    save(TABLES / 'table_manifest.json', manifest)
    audit = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'updated_cells': sorted(updates),
             'archive': revision.relative_to(ROOT).as_posix(), 'summary_path': frozen_summary.relative_to(ROOT).as_posix(),
             'summary_sha256': sha(frozen_summary), 'all_prior_numeric_results_unchanged': not bool(interim_replacements),
             'all_prior_final_results_unchanged': True, 'replaced_interim_cells': interim_replacements, 'new_training_or_inference': False}
    save(revision / 'IMPORT_AUDIT.json', audit)
    save(PAPER / 'LATEST_BASELINE_IMPORT.json', audit)
    print(json.dumps({'updated_cells': sorted(updates), 'filled_cells': manifest['filled_cells'], 'pending_cells': manifest['placeholder_cells'], 'archive': str(revision)}))


if __name__ == '__main__':
    main()
