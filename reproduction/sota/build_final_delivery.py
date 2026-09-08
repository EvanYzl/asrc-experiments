"""Bundle documents and raw records, and hash large unchanged checkpoints/inputs."""
import datetime
import hashlib
import json
import tarfile
from pathlib import Path
from frozen_data import ROOT,atomic_json,sha256

base=ROOT/'reproduction/sota';dest=ROOT/'deployment/sota_final';dest.mkdir(parents=True,exist_ok=True)
accept=json.loads((base/'SINGLE_SEED_ACCEPTANCE.json').read_text());assert accept['threshold_met']
assert json.loads((base/'DOCUMENT_DELIVERY_VALIDATION.json').read_text())['passed']
previous_log_sha=sha256(ROOT/'SOTA_LOG.md')
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## Final local delivery validation — '+stamp+'\n')
    f.write('Purpose: deliver only the accepted seed17 Table2 results; no experiments.\n')
    f.write('Changes: Table2 ASRC row, exact reference comparison, Chinese idea source/PDF and experimental design updated; pre-change versions retained.\n')
    f.write('Commands/config: verify_and_aggregate_single.py, publish_single_seed.py, LaTeX builds, validate_current.py and validate_delivery.py; seed17 results only. PID/GPU: local CPU document/integrity work, no GPU job.\n')
    f.write('Results: local raw-rank recomputation matches all four locked results; all four downloaded checkpoints and original data/alignment/code hashes pass. Table PDF schema/layout and 12 displayed numbers pass; 75 external comparison cells retained.\n')
    f.write('Artifacts: SINGLE_SEED_ACCEPTANCE.json, DOCUMENT_DELIVERY_VALIDATION.json, Table_2_SOTA_single_seed.pdf, idea.std.zh.pdf, final delivery receipt.\n')
    f.write('Conclusion: 单种子达到暂定门槛, 12/12 strict wins. No full comparable SOTA claim. Next: server byte-for-byte bundle verification, return final receipt/log, then await user.\n')
bundle=set([ROOT/'SOTA_LOG.md'])
extensions={'.py','.json','.jsonl','.md','.csv','.npz','.log','.tex','.pdf','.ps1'}
excluded={'FINAL_DELIVERY_RECEIPT.json','FINAL_SYNC_VERIFICATION.json','FINAL_LOCAL_VERIFICATION.json'}
for path in base.rglob('*'):
    if path.is_file() and path.suffix in extensions and '__pycache__' not in path.parts and path.name not in excluded:bundle.add(path)
tables=ROOT/'outputs/kbs/_main/_tables'
for path in tables.rglob('*'):
    if path.is_file() and path.suffix in extensions and 'build' not in path.parts:bundle.add(path)
for folder in [ROOT/'work/ideaspark/_run/multidomain-kgc-local/_2/phase4',ROOT/'work/ideaspark_run/multidomain-kgc-local_2/phase4']:
    for ext in ['md','tex','pdf']:bundle.add(folder/f'idea.std.zh.{ext}')
unchanged=set()
for ds in accept['datasets']:
    freeze=json.loads((base/'single_seed'/ds/'freeze.json').read_text());cp=ROOT/freeze['checkpoint'];unchanged.add(cp)
    config=json.loads(cp.with_name('config.json').read_text())
    for kg in config['manifests']:
        mp=ROOT/'reproduction/strict_baselines/data_manifests'/f'{ds}_{kg}.json';unchanged.add(mp)
        unchanged.update(ROOT/name.replace('\\','/') for name in json.loads(mp.read_text())['files'])
    unchanged.update(ROOT/name for name in config['alignment']['alignment_files'])
entries=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha256(p),'bytes':p.stat().st_size,'in_bundle':p in bundle} for p in sorted(bundle|unchanged)]
receipt={'timestamp':stamp,'status':'prepared_local_verified','method':'ASRC','seed':17,'verdict':accept['verdict'],'strict_wins':12,
    'local_root':'G:/zhishitupui','remote_root':'/root/zhishitupui','files':entries,
    'large_checkpoints_transferred_separately':True,'all_raw_test_rank_files_included':True,
    'unknown_and_provisional_comparisons_retained':True,'further_experiments_authorized':False}
receipt_path=base/'FINAL_DELIVERY_RECEIPT.json';atomic_json(receipt_path,receipt)
archive=dest/'delivery.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for path in sorted(bundle|{receipt_path}):tar.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
atomic_json(dest/'package.json',{'archive_sha256':sha256(archive),'receipt_sha256':sha256(receipt_path),
    'remote_log_before_sha256':previous_log_sha,'files_in_bundle':len(bundle)+1,'verification_files':len(entries),'archive_bytes':archive.stat().st_size})
print(json.dumps({'bundle_files':len(bundle)+1,'verified_files':len(entries),'archive_bytes':archive.stat().st_size}))
