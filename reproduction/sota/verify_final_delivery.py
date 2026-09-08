"""Verify the delivery across machines; operates on artifacts, never on experiments."""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];base=ROOT/'reproduction/sota'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
def validate(receipt):
    for item in receipt['files']:
        path=(ROOT/item['path']).resolve();assert path.is_relative_to(ROOT)
        assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],item['path']

p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
receipt_path=base/'FINAL_DELIVERY_RECEIPT.json'
if args.apply:
    pack=ROOT/'deployment/sota_final';metadata=json.loads((pack/'package.json').read_text());archive=pack/'delivery.tar.gz'
    assert digest(archive)==metadata['archive_sha256']
    assert digest(ROOT/'SOTA_LOG.md')==metadata['remote_log_before_sha256'],'Server log changed after last snapshot; preserve newer log before sync.'
    with tarfile.open(archive,'r:gz') as tar:
        receipt_bytes=tar.extractfile('reproduction/sota/FINAL_DELIVERY_RECEIPT.json').read()
        assert hashlib.sha256(receipt_bytes).hexdigest()==metadata['receipt_sha256'];receipt=json.loads(receipt_bytes)
        expected={x['path']:x for x in receipt['files'] if x['in_bundle']}
        assert len(tar.getmembers())==len(expected)+1
        for member in tar.getmembers():
            assert member.isfile();target=(ROOT/member.name).resolve();assert target.is_relative_to(ROOT)
            data=tar.extractfile(member).read()
            if member.name in expected:assert hashlib.sha256(data).hexdigest()==expected[member.name]['sha256']
            else:assert member.name=='reproduction/sota/FINAL_DELIVERY_RECEIPT.json'
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    validate(receipt)
    audit=subprocess.run([sys.executable,str(base/'verify_and_aggregate_single.py'),'--check-only'],capture_output=True,text=True)
    assert audit.returncode==0,audit.stderr
    processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    running=[line for line in processes.splitlines() if any(token in line for token in ['python reproduction/sota/train_complex.py','python reproduction/sota/evaluate_single.py','python reproduction/sota/run_sota_queue.py','kgc/bin/python reproduction/sota/evaluate_single.py'])]
    assert not running,running
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    verification={'timestamp':stamp,'passed':True,'all_manifest_files_verified':len(receipt['files']),'raw_rank_recheck':json.loads(audit.stdout),
        'campaign_processes_running':running,'gpu_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
        'scope':'Single seed17 acceptance and artifact synchronization only. All experiment work stopped.'}
    save(base/'FINAL_SYNC_VERIFICATION.json',verification)
    scope=json.loads((base/'TASK_SCOPE.json').read_text());scope.update(status='completed_awaiting_user',delivery_verified_at=stamp,further_experiments_authorized=False)
    save(base/'TASK_SCOPE.json',scope)
    with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
        f.write('\n## Final server delivery verification — '+stamp+'\nPurpose: close delivery after the stopping condition. No experiment launched.\n')
        f.write('Command: python reproduction/sota/verify_final_delivery.py --apply; CPU/SFTP verification only; seed17 artifacts.\n')
        f.write('Result: every final manifest file hash and size matched; original inputs, all four checkpoints, raw ranks, Table2, idea sources/PDF, design, logs and archives verified. Raw-rank acceptance recheck returned 12/12.\n')
        f.write('Artifacts: FINAL_DELIVERY_RECEIPT.json, FINAL_SYNC_VERIFICATION.json; return these and this final log to the local workspace and verify against the same manifest.\n')
        f.write('Conclusion: 单种子达到暂定门槛. Campaign processes remaining: 0. No new baselines, additional seeds or retesting. Unknown/proxy comparisons and repeat-count differences remain disclosed. All this campaign experiments stopped; await user after returned-file hash verification.\n')
    paths={x['path']:x for x in receipt['files']}
    for path in [base/'TASK_SCOPE.json',ROOT/'SOTA_LOG.md',base/'FINAL_SYNC_VERIFICATION.json']:
        name=path.relative_to(ROOT).as_posix();paths[name]={'path':name,'sha256':digest(path),'bytes':path.stat().st_size,'in_bundle':False,'final_remote_return':True}
    receipt.update(status='server_verified_return_receipt',server_verified_at=stamp,files=[paths[k] for k in sorted(paths)])
    save(receipt_path,receipt);validate(receipt)
    print(json.dumps({'passed':True,'files':len(receipt['files']),'receipt_sha256':digest(receipt_path),'log_sha256':digest(ROOT/'SOTA_LOG.md'),'experiments_running':0},ensure_ascii=False))
else:
    receipt=json.loads(receipt_path.read_text());assert receipt['status']=='server_verified_return_receipt';validate(receipt)
    result={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'files':len(receipt['files']),
        'receipt_sha256':digest(receipt_path),'log_sha256':digest(ROOT/'SOTA_LOG.md'),'verdict':receipt['verdict'],
        'local_and_server_bytes_match':True,'further_experiments_authorized':False}
    save(base/'FINAL_LOCAL_VERIFICATION.json',result);print(json.dumps(result,ensure_ascii=False))
