import hashlib
import json
import pathlib
import shutil
import tarfile
from datetime import datetime, timezone

root = pathlib.Path(__file__).resolve().parents[1]
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
run = root / 'reproduction/sota'
run.mkdir(parents=True, exist_ok=True)
archive = run / 'history' / stamp
archive.mkdir(parents=True, exist_ok=True)
idea = root / 'work/ideaspark_run/multidomain-kgc-local_2/phase4'
tables = root / 'outputs/kbs_main_tables'
canonical_idea = root / 'work/ideaspark/_run/multidomain-kgc-local/_2/phase4'
canonical_tables = root / 'outputs/kbs/_main/_tables'
for src, dst in [(idea, canonical_idea), (tables, canonical_tables)]:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob('*'):
        if p.is_file() and not any(x in {'build', 'history'} for x in p.relative_to(src).parts):
            relative = p.relative_to(src)
            saved = archive / src.name / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, saved)
            target = dst / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(p, target)
files = []
for name in ['LSMGA', 'DMKGC', 'IMKGC']:
    source = root / 'reproduction/sources' / name
    files.extend(p for p in source.rglob('*') if p.is_file() and p.suffix in {'.py', '.md', '.txt', '.json', '.yaml', '.yml'} and not any(x in {'.git', '__pycache__', 'data', 'dataset', 'datasets', 'wandb'} for x in p.relative_to(source).parts))
files.extend(p for p in (root/'reproduction/strict_baselines').iterdir() if p.is_file() and p.suffix in {'.py','.md','.json'})
files.extend(p for p in canonical_idea.rglob('*') if p.is_file())
files.extend(p for p in canonical_tables.rglob('*') if p.is_file())
files.extend(p for p in run.iterdir() if p.is_file() and p.suffix in {'.py','.json','.md'})
receipt = []
path = root/'deployment/sota_code_bundle.tar.gz'
with tarfile.open(path, 'w:gz') as tar:
    for p in sorted(set(files)):
        tar.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
        receipt.append({'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(root/'deployment/sota_bundle_receipt.json').write_text(json.dumps({'timestamp':stamp,'archive':str(archive),'files':receipt}, indent=2), encoding='utf-8')
print(json.dumps({'files':len(files),'bundle_bytes':path.stat().st_size,'archive':str(archive)}))
