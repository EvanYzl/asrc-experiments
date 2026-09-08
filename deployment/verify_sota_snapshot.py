import hashlib
import json
import pathlib
import tarfile
root=pathlib.Path(__file__).resolve().parents[1]
dest=root/'deployment/sota_snapshots'
receipt=json.loads((dest/'latest_receipt.json').read_text(encoding='utf-8'))
assert hashlib.sha256((dest/'latest.tar.gz').read_bytes()).hexdigest()==receipt['archive_sha256']
with tarfile.open(dest/'latest.tar.gz') as tar:
    for item in receipt['files']:
        target=(root/item['path']).resolve()
        assert target.is_relative_to(root.resolve())
        content=tar.extractfile(item['path']).read()
        assert len(content)==item['bytes'] and hashlib.sha256(content).hexdigest()==item['sha256']
        # Remote scripts and results are authoritative only inside this campaign.
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
print(json.dumps({'verified_and_synced':len(receipt['files']),'timestamp':receipt['timestamp']}))
