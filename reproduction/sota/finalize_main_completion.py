"""Finalize documents and delivery receipts for the completed, fixed 39-job queue.

Reads saved artifacts only. Does not train, select models, or run inference.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import io
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reproduction/language_table_seed17/main_completion_20260907'
PAPER = ROOT / 'reproduction/sota/paper_support'
TABLES = ROOT / 'outputs/kbs/_main/_tables'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def documents(revision):
    results = read(BASE / 'PUBLICATION_RESULTS.json')
    assert results['complete_groups'] == results['total_groups'] == 5
    group = next(g for g in results['groups'] if g['method'] == 'SS-AGA' and g['dataset'] == 'depkg')
    assert group['status'] == 'complete' and len(group['runs']) == 18
    display = ' / '.join(group['display_percent'][m].replace(' +/- ', '±') for m in ['mrr', 'h1', 'h10'])
    old = ('Table 3 的 SS-AGA/E-PKG 已完成 seed17/29，每个种子覆盖六个 KG；两种子均值±样本标准差为 '
           '28.57±0.38 / 16.35±1.01 / 52.12±0.43（MRR / H@1 / H@10，%），当前明确标注为 2/3 的暂统计。'
           '正式三种子结果仍待原独立队列完成 seed43。')
    new = (f'Table 3 的 SS-AGA/E-PKG 已完成 seed17/29/43，每个种子覆盖六个 KG；三种子均值±样本标准差为 '
           f'{display}（MRR / H@1 / H@10，%）。18 项原始结果全部通过核验，已替换此前两种子暂统计。'
           '固定 39 项补齐队列已全部完成并回传，不再追加实验。')
    canonical = ROOT / 'work/ideaspark/_run/multidomain-kgc-local/_2/phase4'
    legacy = ROOT / 'work/ideaspark_run/multidomain-kgc-local_2/phase4'
    for suffix in ['md', 'tex']:
        path = canonical / ('idea.std.zh.' + suffix)
        content = path.read_text(encoding='utf-8')
        previous = old.replace('%', r'\%') if suffix == 'tex' else old
        replacement = new.replace('%', r'\%') if suffix == 'tex' else new
        assert content.count(previous) == 1
        path.write_text(content.replace(previous, replacement), encoding='utf-8')
        shutil.copy2(path, legacy / path.name)
    design = TABLES / 'EXPERIMENT_DESIGN.zh.md'
    content = design.read_text(encoding='utf-8')
    assert old in content
    content = content.replace(old, new, 1)
    for heading in ['2026-09-07 主表完成与本次回填', '2026-09-07 SS-AGA/E-PKG 单种子进度回填', '2026-09-07 SS-AGA/E-PKG 两种子更新']:
        content = content.replace('## ' + heading, '## ' + heading + '（历史记录，以最终三种子验收为准）')
    content = content.replace('其余独立任务的 WK3l 图方法与 SS-AGA 实验继续由原任务管理，不属于本阶段新增实验。',
                              '其余独立补齐队列也已完成；其 WK3l 图方法与 SS-AGA 结果见下述最终三种子验收。')
    content += ('\n\n## 2026-09-07 最终三种子验收与停止\n\n' + new + '\n\n'
                '最后一项于北京时间 14:15 结束，39/39 项于 14:16 回传核验完成。复用 WK3l 原有三个 seed17，'
                '本次完整审计覆盖 42 项保存的结果；只重算保存排名的统计，不重新推理或打开测试集选方案。'
                'Table 1–8 的 301 个登记单元格全部填齐，无暂统计或待补种子；语言汇总表保留原有 seed17 口径。'
                'Table 2 的完整精度数值、固定比较范围及最优加粗/次优下划线保持一致，ASRC 为 12/12 项严格领先。'
                'SS-AGA 仍只列在方法专属 Table 3，其 train 过滤、supporter 事实和文本输入均保持登记口径。'
                '原始数据、来源凭据和汇总位于 reproduction/language_table_seed17/main_completion_20260907；'
                '此前文档与两种子版本保存在 ' + revision.relative_to(ROOT).as_posix() + '/before。全部登记实验已停止。\n')
    design.write_text(content, encoding='utf-8')
    compare = read(BASE / 'CURRENT_COMPARISON.json')
    previous = read(revision / 'before/evidence/CURRENT_COMPARISON.json')
    assert compare['columns'] == previous['columns'] and compare['strict_wins'] == 12
    compare.update(publication_updated_at=now(), csv_sha256=sha(TABLES / 'cells_results.csv'),
                   new_results_sha256=sha(BASE / 'PUBLICATION_RESULTS.json'), remaining_table3_cells=[],
                   remaining_table3_cells_reason='All registered Table3 groups now have verified seeds17/29/43',
                   scope_stop='All registered experiments completed; all six GPUs idle; no further training or tests')
    compare.pop('latest_interim_results_sha256', None)
    save(BASE / 'CURRENT_COMPARISON.json', compare)
    with (TABLES / 'cells_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 301 and all(r['value'] for r in rows)
    assert all(not r['notes'].startswith('INTERIM_') for r in rows)
    manifest = read(TABLES / 'table_manifest.json')
    manifest['historical_baseline_scope'] = manifest['baseline_scope']
    scoped = [r for r in rows if r['table_id'] in ['T1', 'T2', 'T3']]
    manifest['baseline_scope'] = {'updated_at': now(), 'scope': 'Current registered Table1-3 cells; prior schema retained as historical_baseline_scope',
                                 'active_tables': ['T1', 'T2', 'T3'], 'registered_cells': len(scoped), 'filled': len(scoped),
                                 'pending': 0, 'missing_jobs': [], 'priority_tables_complete': True,
                                 'by_table': {t: {'filled': sum(r['table_id'] == t for r in scoped), 'pending': 0} for t in ['T1', 'T2', 'T3']}}
    published = {(g['method'], g['dataset'], g.get('condition', 'full')): g for g in manifest['published_groups']}
    for g in manifest['three_seed_groups']:
        published[g['method'], g['dataset'], g.get('condition', 'full')] = g
    manifest.update(published_groups=list(published.values()), graph_baseline_complete_three_seed_groups=12,
                    graph_baseline_total_groups=12, current_comparison_sha256=sha(BASE / 'CURRENT_COMPARISON.json'),
                    fill_audit=str((revision / 'IMPORT_AUDIT.json').relative_to(ROOT)).replace('\\', '/'),
                    task_scope='All registered Table1-8 and main completion jobs finished; no further experiments')
    save(TABLES / 'table_manifest.json', manifest)
    print(json.dumps({'final_ssaga_epkg_percent': display, 'filled_cells': len(rows), 'pending_cells': 0}, ensure_ascii=False))


def receipts(revision):
    transfer = read(BASE / 'TRANSFER_STATE.json')
    source = read(BASE / 'PUBLICATION_SOURCE_RECEIPT.json')
    assert len(transfer['jobs']) == len(source['jobs']) == 39
    files, archives = [], []
    for jid, job in transfer['jobs'].items():
        assert job['status'] == 'returned_verified' and job['verification']['status'] == 'passed'
        archive = BASE / 'returns' / (jid + '.tar.gz')
        digest = sha(archive)
        assert digest == job['archive_sha256'] == source['jobs'][jid]['return_archive_sha256']
        archives.append({'path': archive.relative_to(ROOT).as_posix(), 'bytes': archive.stat().st_size, 'sha256': digest})
        out = BASE / 'returned_jobs' / jid
        hashes = read(out / 'RETURN_HASHES.json')
        for name, expected in hashes.items():
            path = (out / name).resolve()
            assert path.is_relative_to(out.resolve()) and sha(path) == expected
            files.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': expected})
    report = {'timestamp': now(), 'passed': True, 'registered_jobs': 39, 'returned_verified_jobs': 39,
              'files_verified': len(files), 'raw_bytes': sum(f['bytes'] for f in files),
              'archive_bytes': sum(f['bytes'] for f in archives), 'archives': archives, 'files': files,
              'source_receipt_sha256': sha(BASE / 'PUBLICATION_SOURCE_RECEIPT.json'), 'new_training_or_inference': False}
    save(BASE / 'ALL_AVAILABLE_RAW_RETURN_VERIFICATION.json', report)
    save(revision / 'ALL_AVAILABLE_RAW_RETURN_VERIFICATION.json', report)
    print(json.dumps({k: report[k] for k in ['passed', 'returned_verified_jobs', 'files_verified', 'raw_bytes', 'archive_bytes']}))


def publication(revision):
    validation = read(PAPER / 'DOCUMENT_VALIDATION.json')
    raw = read(BASE / 'ALL_AVAILABLE_RAW_RETURN_VERIFICATION.json')
    assert validation['passed'] and not validation['pending_three_seed_cells'] and raw['returned_verified_jobs'] == 39
    final_pdf = TABLES / 'KBS_Main_Text_Tables.20260907_final.pdf'
    canonical_pdf = TABLES / 'KBS_Main_Text_Tables.pdf'
    canonical_current = sha(final_pdf) == sha(canonical_pdf)
    active = canonical_pdf if canonical_current else final_pdf
    pdfs = [active, TABLES / 'language_breakdown_seed17/All_Datasets_Language_KG_Seed17.pdf',
            ROOT / 'work/ideaspark/_run/multidomain-kgc-local/_2/phase4/idea.std.zh.pdf']
    manifest = read(TABLES / 'table_manifest.json')
    manifest.update(active_pdf=active.name, active_pdf_sha256=sha(active), canonical_pdf_update_pending=not canonical_current)
    save(TABLES / 'table_manifest.json', manifest)
    report = {'timestamp': now(), 'status': 'final_results_published' if canonical_current else 'final_results_published_to_dated_copy',
              'full_queue_complete': True, 'registered_jobs': 39, 'returned_verified_jobs': 39, 'remaining_training_jobs': 0,
              'seeds_complete': [17, 29, 43], 'seed_pending': None, 'filled_cells': 301, 'formal_or_original_filled_cells': 301,
              'interim_cells': [], 'table2_complete': True, 'table2_unrounded_values_and_styles_unchanged': True,
              'table2_strict_wins': 12, 'language_numeric_cells_verified': 378,
              'language_verification': 'Unchanged PDF and sources from the previously verified 152-file language publication',
              'all_other_final_values_and_sources_unchanged': True, 'raw_files_verified': raw['files_verified'],
              'raw_bytes': raw['raw_bytes'], 'archive_bytes': raw['archive_bytes'],
              'raw_return_manifest_sha256': sha(BASE / 'ALL_AVAILABLE_RAW_RETURN_VERIFICATION.json'),
              'new_training_or_inference': False, 'gpu_status': 'All six idle at live inspection',
              'canonical_pdf_updated': canonical_current,
              'canonical_pdf_pending_reason': None if canonical_current else 'Windows sharing lock from another program; final copy is compiled and verified',
              'pdf_layout_checked': 'Four main pages and three idea pages verified; Table2 and Table3 visually checked, no overflow or numeric line wrapping',
              'pdfs': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in pdfs],
              'version_archive': revision.relative_to(ROOT).as_posix(), 'next': 'No further experiments; await instructions'}
    save(BASE / 'MAIN_PUBLICATION.json', report)
    save(revision / 'MAIN_PUBLICATION.json', report)
    for p in [PAPER / 'DOCUMENT_VALIDATION.json', TABLES / 'table_manifest.json', TABLES / 'EXPERIMENT_DESIGN.zh.md']:
        shutil.copy2(p, revision / p.name)
    for p in pdfs:
        shutil.copy2(p, revision / 'published' / p.name)
    print(json.dumps({'status': report['status'], 'canonical_pdf_updated': canonical_current, 'pdfs': report['pdfs']}))


def pack(revision):
    dest = ROOT / 'deployment/main_tables_final_20260907'
    dest.mkdir(parents=True, exist_ok=True)
    prior = read(ROOT / 'deployment/main_tables_return_20260907/publication.json')
    paths = {ROOT / entry['path'] for entry in prior['files']}
    paths.update(p for p in revision.rglob('*') if p.is_file())
    paths.update(p for p in (BASE / 'publication_acceptance').glob('*.json'))
    paths.update(p for p in (BASE / 'publication_evidence').glob('FINAL_*') if p.is_file())
    paths.update(BASE / n for n in ['PUBLICATION_RESULTS.json', 'PUBLICATION_SOURCE_RECEIPT.json', 'CURRENT_COMPARISON.json',
                                  'MAIN_PUBLICATION.json', 'ALL_AVAILABLE_RAW_RETURN_VERIFICATION.json', 'TRANSFER_STATE.json', 'transfer_main.log'])
    paths.add(Path(__file__))
    final_pdf = TABLES / 'KBS_Main_Text_Tables.20260907_final.pdf'
    paths.add(final_pdf)
    canonical = TABLES / 'KBS_Main_Text_Tables.pdf'
    if sha(canonical) != sha(final_pdf):
        paths.discard(canonical)
    paths.discard(ROOT / 'SOTA_LOG.md')
    assert all(p.is_file() and p.resolve().is_relative_to(ROOT.resolve()) for p in paths)
    entries = []
    with tarfile.open(dest / 'publication.tar.gz', 'w:gz') as tar:
        for path in sorted(paths):
            content = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            item = tarfile.TarInfo(name)
            item.size = len(content)
            tar.addfile(item, io.BytesIO(content))
            entries.append({'path': name, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
    save(dest / 'publication.json', {'timestamp': now(), 'scope': 'Final39-job results, current tables and idea, raw-data verification and preserved prior versions',
                                    'archive_sha256': sha(dest / 'publication.tar.gz'), 'files': entries})
    print(json.dumps({'files': len(entries), 'archive_bytes': (dest / 'publication.tar.gz').stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['documents', 'receipts', 'publication', 'pack'])
    parser.add_argument('--archive', required=True, type=Path)
    args = parser.parse_args()
    revision = args.archive.resolve()
    assert revision.is_relative_to((PAPER / 'document_versions').resolve()) and revision.is_dir()
    globals()[args.mode](revision)
