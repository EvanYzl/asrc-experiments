"""Create an immutable, hashed lightweight sync snapshot while jobs run."""
import datetime
import json
import tarfile
from pathlib import Path
from frozen_data import ROOT,sha256,atomic_json
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
dest=ROOT/'deployment/sota_snapshots';dest.mkdir(parents=True,exist_ok=True)
files=[ROOT/'SOTA_LOG.md']
for p in (ROOT/'reproduction/sota').rglob('*'):
    if p.is_file() and p.suffix in {'.py','.json','.jsonl','.md','.csv','.npz','.log'} and '__pycache__' not in p.parts:files.append(p)
receipt=[]
with tarfile.open(dest/'latest.tar.gz','w:gz') as tar:
    for p in sorted(files):
        # A single byte snapshot avoids inconsistent hash/length if live logs grow.
        import io,hashlib
        content=p.read_bytes();name=p.relative_to(ROOT).as_posix()
        info=tarfile.TarInfo(name);info.size=len(content)
        tar.addfile(info,io.BytesIO(content))
        receipt.append({'path':name,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
atomic_json(dest/'latest_receipt.json',{'timestamp':stamp,'archive_sha256':sha256(dest/'latest.tar.gz'),'files':receipt})
print(json.dumps({'timestamp':stamp,'files':len(receipt),'bytes':(dest/'latest.tar.gz').stat().st_size}))
