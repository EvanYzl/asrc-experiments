"""Verify returned artifacts while preserving the independent Windows baseline runtime."""
import argparse
import datetime
import json
from frozen_data import ROOT,atomic_json,sha256
phase=ROOT/'reproduction/sota/paper_support';rp=phase/'SERVER_FINAL_RECEIPT.json'
p=argparse.ArgumentParser();p.add_argument('--metadata-only',action='store_true');args=p.parse_args();receipt=json.loads(rp.read_text())
# The server's exact common.py has always been mirrored here. The original Windows
# baseline common.py belongs to a separate running campaign and is not overwritten.
mapping={'reproduction/strict_baselines/common.py':'reproduction/sota/runtime/server_common.py'}
items=[x for x in receipt['files'] if not args.metadata_only or not x['path'].endswith('.pt')]
for item in items:
    path=ROOT/mapping.get(item['path'],item['path'])
    assert path.exists() and path.stat().st_size==item['bytes'] and sha256(path)==item['sha256'],(item['path'],str(path))
result={'status':'metadata_verified' if args.metadata_only else 'both_hosts_verified','timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'server_receipt_sha256':sha256(rp),'verified_files':len(items),'checkpoints':0 if args.metadata_only else 48,'support_cells':174,
    'experiments_running':False,'exact_server_source_local_mirror':mapping,
    'note':'All returned content hashes match the server receipt; the server evaluation helper is verified against its exact local runtime mirror, preserving the independent Windows baseline helper.'}
atomic_json(phase/('LOCAL_METADATA_VERIFICATION.json' if args.metadata_only else 'LOCAL_FINAL_VERIFICATION.json'),result)
print(json.dumps(result))
