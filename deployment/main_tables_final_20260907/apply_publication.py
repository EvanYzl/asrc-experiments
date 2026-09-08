"""Verify and install the final document bundle, preserving changed server files."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


manifest = json.loads((DEST / 'publication.json').read_text())
assert sha(DEST / 'publication.tar.gz') == manifest['archive_sha256']
expected = {entry['path']: entry for entry in manifest['files']}
assert len(expected) == len(manifest['files']) and 'SOTA_LOG.md' not in expected
with tarfile.open(DEST / 'publication.tar.gz') as archive:
    members = archive.getmembers()
    assert len(members) == len(expected) and {m.name for m in members} == set(expected)
    for member in members:
        path = (ROOT / member.name).resolve()
        assert member.isfile() and path.is_relative_to(ROOT.resolve())
        data = archive.extractfile(member).read()
        entry = expected[member.name]
        assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
    for member in members:
        path = ROOT / member.name
        entry = expected[member.name]
        if path.exists() and sha(path) != entry['sha256']:
            before = DEST / 'server_before' / member.name
            if not before.exists():
                before.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, before)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.extractfile(member).read())
        assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']

base = ROOT / 'reproduction/language_table_seed17/main_completion_20260907'
report = json.loads((base / 'MAIN_PUBLICATION.json').read_text())
state = json.loads((ROOT / 'reproduction/main_tables_completion_20260907/QUEUE_STATE.json').read_text())
assert state['counts'] == {'completed': 39}
assert all(j['status'] == 'completed' for j in state['jobs'].values())
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader'], text=True)
compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
with (ROOT / 'SOTA_LOG.md').open('a', encoding='utf-8') as log:
    log.write('\n## ' + stamp + ' — 最终三种子回填与交付核验（AFTER）\n')
    log.write('- 目的：确认全部训练结束，并完成此前授权的所有表格回填、文档编译及数据回传。\n')
    log.write('- 队列：固定39/39项全部完成，最后SS-AGA/E-PKG/UK seed43于北京时间14:15结束，39项于14:16本地回传核验完成。\n')
    log.write('- 配置与种子：沿用原登记输入、划分、25轮SS-AGA预算和seed17/29/43；WK3l三项原seed17复用。CPU审计42项已有输出，只从保存排名汇总，未启动训练、推理或重新选模。\n')
    log.write('- 命令：audit_main_completion.py；publish_available_main_tables.py --archive reproduction/sota/paper_support/document_versions/ssaga_epkg_three_seed_20260907_083100；finalize_main_completion.py；XeLaTeX编译；validate_support_documents.py；apply_publication.py。\n')
    log.write('- Table3：SS-AGA/E-PKG最终三种子MRR / H@1 / H@10为28.27±0.58 /16.17±0.79 /51.25±1.54（%）。按每种子六KG等权宏平均，再求种子均值与样本标准差；仍为方法专属train过滤。此前两种子暂统计与全部修改前版本完整保留。\n')
    log.write('- 表格：301个登记单元格全部填齐；Table2完整精度数值及加粗/下划线未变，固定范围12/12领先；语言表378值及原seed17口径保持不变。主表4页、idea3页，编译及数值/布局检查通过；idea两套路径与实验设计更新。\n')
    log.write('- 回传：39个归档与615个原始文件全量SHA256核验，原始文件9,788,874,301字节，归档7,283,885,212字节。路径G:/zhishitupui/reproduction/language_table_seed17/main_completion_20260907/returned_jobs；证据ALL_AVAILABLE_RAW_RETURN_VERIFICATION.json与PUBLICATION_RESULTS.json。\n')
    log.write('- 文档同步：' + str(len(expected)) + '个文档、源文件、统计及历史文件逐个SHA256验证。原名PDF被本地阅读程序占用，Windows拒绝覆盖；最终版已另存为KBS_Main_Text_Tables.20260907_final.pdf，SHA256=647cd6c8d09e0beff31513852569f238ab178d05a4d0579675f60f2d1cd2bf2f，并完成同步。MAIN_PUBLICATION.json明确记录原名替换待关闭占用程序。\n')
    log.write('- 进程/GPU：原调度队列已结束；本轮GPU分配为无，当前GPU快照：' + gpu.replace('\n', '; ') + '。计算进程：' + (compute or '无') + '。\n')
    log.write('- 结论与下一步：全部登记实验完成并停止，不再改善或提高门槛；交付已核验最终版本，等待用户指令及解除原名PDF占用。\n')
receipt = {'status': 'passed', 'timestamp': stamp, 'files_verified': len(expected), 'files': manifest['files'],
           'archive_sha256': manifest['archive_sha256'], 'manifest_sha256': sha(DEST / 'publication.json'),
           'queue_counts': state['counts'], 'gpu_snapshot': gpu, 'compute_processes': compute,
           'canonical_pdf_updated': report['canonical_pdf_updated'], 'final_pdf': report['pdfs'][0],
           'raw_files_verified': report['raw_files_verified'], 'raw_returned_jobs': report['returned_verified_jobs'],
           'sota_log_sha256': sha(ROOT / 'SOTA_LOG.md'), 'new_training_or_inference': False}
(DEST / 'FINAL_DELIVERY_VERIFICATION.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: receipt[k] for k in ['status', 'files_verified', 'queue_counts', 'raw_returned_jobs', 'canonical_pdf_updated']}))
