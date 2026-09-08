"""Pack and verify the bounded three-seed delivery without launching experiments."""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];base=ROOT/'reproduction/sota';phase=base/'three_seed';pack=ROOT/'deployment/sota_three_seed'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)
def verify(manifest):
    for item in manifest['files']:
        path=(ROOT/item['path']).resolve();assert path.is_relative_to(ROOT)
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path']
def entry(path,bundled):return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size,'bundled':bundled}

p=argparse.ArgumentParser();p.add_argument('mode',choices=['pack','apply','verify']);args=p.parse_args();mp=phase/'DELIVERY_MANIFEST.json'
if args.mode=='pack':
    report=json.loads((phase/'RESULTS.json').read_text());assert json.loads((phase/'TABLE_VALIDATION.json').read_text())['passed']
    before=sha(ROOT/'SOTA_LOG.md');stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
        f.write('\n## Three-seed Table2 compiled and locally verified — '+stamp+'\nPurpose: fill our method and compile the requested PDF. No new experiment.\n')
        f.write('Changes: 12 ASRC cells now report seeds17/29/43 mean and sample SD; 668 other cells and seven other table sources unchanged. Old table/PDF version retained in history/before_three_seed/tables.\n')
        f.write('Command/config: publish_three_seed.py, compile_latex.py main.tex --compiler texlive --engine xelatex --output-directory build, validate_current.py, validate_three_seed_table.py, aggregate_three_seed.py --check-only. PID/GPU: local CPU document/hash work only.\n')
        f.write('Results: four-page KBS_Main_Text_Tables.pdf compiled successfully; numeric text, table schema/layout and raw three-seed mean/sample SD verified. All twelve checkpoints and raw queries available locally and hash-verified.\n')
        f.write('Transfer note: the initial SFTP checkpoint copy was interrupted to use a larger SSH receive window; three already-complete files passed hashes, and only the five missing/incomplete files were resumed. This affected no server experiment or result.\n')
        f.write('Artifacts: outputs/kbs/_main/_tables/main.tex, tables/t02.tex, values.tex, KBS_Main_Text_Tables.pdf; three_seed/RESULTS.json and TABLE_VALIDATION.json.\n')
        f.write('Conclusion: '+report['verdict']+'; all eight requested new tests complete. Next: synchronize final files, verify both machines, and stop.\n')
    files={ROOT/'SOTA_LOG.md',base/'PLAN.md',base/'TASK_SCOPE.json',base/'SINGLE_SEED_REFERENCES.json',base/'runtime/server_common.py'}
    for name in ['prepare_three_seed.py','evaluate_three_seed.py','aggregate_three_seed.py','publish_three_seed.py','validate_three_seed_table.py','close_three_seed_evaluations.py','sync_three_seed.py','train_complex.py','frozen_data.py','evaluate_single.py','run_sota_queue.py']:files.add(base/name)
    suffixes={'.json','.csv','.md','.jsonl','.npz','.log','.tex','.pdf','.py','.ps1'}
    exclude={'DELIVERY_MANIFEST.json','REMOTE_VERIFICATION.json','LOCAL_VERIFICATION.json'}
    for folder in [phase,base/'history/before_three_seed']:
        files.update(x for x in folder.rglob('*') if x.is_file() and x.suffix in suffixes and x.name not in exclude)
    tables=ROOT/'outputs/kbs/_main/_tables'
    names=['main.tex','preamble.tex','values.tex','cells_results.csv','cells_template.csv','KBS_Main_Text_Tables.pdf','EXPERIMENT_DESIGN.zh.md','table_manifest.json','validation_report.json','THREE_SEED_REPORT.zh.md','THREE_SEED_RESULTS.json','THREE_SEED_COMPARISON.csv','build.ps1','validate_current.py']
    files.update(tables/name for name in names);files.update((tables/'tables').glob('*.tex'))
    large=set();freeze=json.loads((phase/'freeze.json').read_text())
    for e in freeze['entries'].values():
        cp=ROOT/e['checkpoint'];large.add(cp);cfg=json.loads(cp.with_name('config.json').read_text())
        files.update(x for x in cp.parent.iterdir() if x.is_file() and x.suffix in suffixes)
        out=(ROOT/e['evaluation_result']).parent
        files.update(x for x in out.rglob('*') if x.is_file() and x.suffix in suffixes)
        if e['seed']==17:files.update([out.parent/'freeze.json',out.parent/'LOCKED_RESULT.json'])
        for kg in cfg['manifests']:
            m=ROOT/'reproduction/strict_baselines/data_manifests'/f"{e['dataset']}_{kg}.json";large.add(m)
            large.update(ROOT/name.replace('\\','/') for name in json.loads(m.read_text())['files'])
        large.update(ROOT/name for name in cfg['alignment']['alignment_files'])
    manifest={'timestamp':stamp,'phase':'ASRC_three_seed_completion','seeds':[17,29,43],'status':'local_prepared',
        'files':[entry(x,x in files) for x in sorted(files|large)],'verdict':report['verdict'],'new_training':0,'new_baselines':0,'new_tests':8,'seed17_retested':False}
    save(mp,manifest);pack.mkdir(parents=True,exist_ok=True)
    with tarfile.open(pack/'delivery.tar.gz','w:gz') as tar:
        for path in sorted(files|{mp}):tar.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    save(pack/'package.json',{'archive_sha256':sha(pack/'delivery.tar.gz'),'manifest_sha256':sha(mp),'remote_log_before_sha256':before})
    print(json.dumps({'bundle_files':len(files)+1,'verified_files':len(manifest['files']),'archive_bytes':(pack/'delivery.tar.gz').stat().st_size}))
elif args.mode=='apply':
    metadata=json.loads((pack/'package.json').read_text());assert sha(pack/'delivery.tar.gz')==metadata['archive_sha256']
    assert sha(ROOT/'SOTA_LOG.md')==metadata['remote_log_before_sha256'],'Server log changed; preserve newer entries before syncing.'
    with tarfile.open(pack/'delivery.tar.gz') as tar:
        b=tar.extractfile(mp.relative_to(ROOT).as_posix()).read();assert hashlib.sha256(b).hexdigest()==metadata['manifest_sha256'];manifest=json.loads(b)
        expected={x['path']:x for x in manifest['files'] if x['bundled']};assert len(tar.getmembers())==len(expected)+1
        for member in tar.getmembers():
            path=(ROOT/member.name).resolve();assert path.is_relative_to(ROOT) and member.isfile();content=tar.extractfile(member).read()
            if member.name in expected:assert hashlib.sha256(content).hexdigest()==expected[member.name]['sha256']
            else:assert member.name==mp.relative_to(ROOT).as_posix()
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    verify(manifest)
    audit=subprocess.run([sys.executable,str(base/'aggregate_three_seed.py'),'--check-only'],capture_output=True,text=True);assert audit.returncode==0,audit.stderr
    running=[line for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if any(token in line for token in ['python reproduction/sota/evaluate_three_seed.py','kgc/bin/python reproduction/sota/evaluate_three_seed.py','python reproduction/sota/run_sota_queue.py'])];assert not running,running
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();save(phase/'REMOTE_VERIFICATION.json',{'timestamp':stamp,'passed':True,'files':len(manifest['files']),'raw_rank_check':json.loads(audit.stdout),'campaign_processes_running':running})
    scope=json.loads((base/'TASK_SCOPE.json').read_text());scope.update(status='completed_awaiting_user',completed_at=stamp,further_experiments_authorized=False);save(base/'TASK_SCOPE.json',scope)
    with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
        f.write('\n## Three-seed final delivery verified on server — '+stamp+'\nPurpose: close the authorized three-seed completion. Command: sync_three_seed.py apply; CPU/file integrity only, no GPU experiment.\n')
        f.write('Results: all delivery files, twelve checkpoints, input and raw rank hashes matched; three-seed aggregation independently rechecked; Table2 mean±sample SD and main PDF synchronized.\n')
        f.write('Artifacts: three_seed/DELIVERY_MANIFEST.json and REMOTE_VERIFICATION.json; main.tex and KBS_Main_Text_Tables.pdf at the requested canonical directory.\n')
        f.write('Conclusion: completed; no campaign process remains. No extra training, baseline or retest beyond the eight missing tests. Unknown/proxy baseline limitations retained. Return final log and receipt, verify local bytes, then wait for user.\n')
    items={x['path']:x for x in manifest['files']}
    for path in [ROOT/'SOTA_LOG.md',base/'TASK_SCOPE.json',phase/'REMOTE_VERIFICATION.json']:items[path.relative_to(ROOT).as_posix()]=entry(path,False)
    manifest.update(status='server_verified',server_verified_at=stamp,files=[items[k] for k in sorted(items)]);save(mp,manifest);verify(manifest)
    print(json.dumps({'passed':True,'files':len(manifest['files']),'manifest_sha256':sha(mp),'log_sha256':sha(ROOT/'SOTA_LOG.md'),'campaign_processes_running':0}))
else:
    manifest=json.loads(mp.read_text());assert manifest['status']=='server_verified';verify(manifest)
    result={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'files':len(manifest['files']),'manifest_sha256':sha(mp),
        'log_sha256':sha(ROOT/'SOTA_LOG.md'),'local_and_remote_bytes_match':True,'seeds':[17,29,43]}
    save(phase/'LOCAL_VERIFICATION.json',result);print(json.dumps(result))
