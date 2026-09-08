"""Hashed local document bundle; safe server apply without touching experiment logs."""
import argparse
import datetime
import io
import json
import tarfile
from pathlib import Path
from frozen_data import ROOT,sha256,atomic_json
p=argparse.ArgumentParser();p.add_argument('mode',choices=['pack','apply']);mode=p.parse_args().mode
base=ROOT/'reproduction/sota';phase=base/'paper_support';dest=ROOT/'deployment/paper_support';dest.mkdir(parents=True,exist_ok=True)
if mode=='pack':
    plan=json.loads((phase/'PLAN.json').read_text());archive=ROOT/plan['old_version_archive'];files=set()
    folders=[ROOT/'outputs/kbs/_main/_tables',ROOT/'work/ideaspark/_run/multidomain-kgc-local/_2/phase4',
        ROOT/'work/ideaspark_run/multidomain-kgc-local_2/phase4',archive,phase/'document_versions']
    for folder in folders:
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in ['.md','.tex','.pdf','.json','.csv','.py','.ps1','.png','.npz'] and 'build' not in path.relative_to(folder).parts:files.add(path)
    files.update(path for path in base.glob('*support*.py'))
    for name in ['import_table2_screenshot.py','import_imkgc_dbp_seed17.py']:
        if (base/name).exists():files.add(base/name)
    files.update(path for path in (ROOT/'refine-logs').glob('EXPERIMENT_*') if path.suffix=='.md')
    files.add(ROOT/'MANIFEST.md')
    for name in ['TABLE_SCHEMA.json','DOCUMENT_VALIDATION.json','APPROVED_BASELINE_UPDATES.json','LATEST_BASELINE_IMPORT.json']:
        if (phase/name).exists():files.add(phase/name)
    items=[]
    with tarfile.open(dest/'documents.tar.gz','w:gz') as tar:
        for path in sorted(files):
            content=path.read_bytes();name=path.relative_to(ROOT).as_posix();info=tarfile.TarInfo(name);info.size=len(content);tar.addfile(info,io.BytesIO(content))
            items.append({'path':name,'sha256':sha256(path),'bytes':len(content)})
    atomic_json(dest/'documents.json',{'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_sha256':sha256(dest/'documents.tar.gz'),'files':items})
    print(json.dumps({'files':len(files),'archive_bytes':(dest/'documents.tar.gz').stat().st_size}))
else:
    manifest=json.loads((dest/'documents.json').read_text());assert sha256(dest/'documents.tar.gz')==manifest['archive_sha256'];items={x['path']:x for x in manifest['files']}
    with tarfile.open(dest/'documents.tar.gz','r:gz') as tar:
        assert set(tar.getnames())==set(items)
        for member in tar.getmembers():
            target=ROOT/member.name;assert member.isfile() and target.resolve().is_relative_to(ROOT.resolve())
            content=tar.extractfile(member).read();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
            assert target.stat().st_size==items[member.name]['bytes'] and sha256(target)==items[member.name]['sha256']
    atomic_json(dest/'documents_server_receipt.json',{'status':'verified','files':manifest['files'],'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_sha256':manifest['archive_sha256']})
    print(json.dumps({'server_document_files_verified':len(items)}))
