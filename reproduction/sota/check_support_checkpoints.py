"""Incremental local verification during large immutable checkpoint transfers."""
import json
from frozen_data import ROOT,sha256,atomic_json
phase=ROOT/'reproduction/sota/paper_support';freeze=json.loads((phase/'EVALUATION_FREEZE.json').read_text());path=phase/'CHECKPOINT_TRANSFER_PROGRESS.json'
cache=json.loads(path.read_text()).get('files',{}) if path.exists() else {};valid={};pending=[]
for ident,e in freeze['models'].items():
    name=e['training_path']+'/best.pt';cp=ROOT/name
    if not cp.exists():pending.append(ident);continue
    stat=cp.stat();old=cache.get(name,{})
    if old.get('sha256')==e['checkpoint_sha256'] and old.get('bytes')==stat.st_size and old.get('mtime_ns')==stat.st_mtime_ns:valid[name]=old;continue
    if stat.st_size<e['parameters']*8:pending.append(ident);continue
    if sha256(cp)==e['checkpoint_sha256']:valid[name]={'sha256':e['checkpoint_sha256'],'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    else:pending.append(ident)
atomic_json(path,{'verified':len(valid),'required':48,'verified_bytes':sum(v['bytes'] for v in valid.values()),'pending':pending,'files':valid})
print(json.dumps({'verified_checkpoints':len(valid),'required':48,'verified_GiB':round(sum(v['bytes'] for v in valid.values())/2**30,2),'pending_count':len(pending)}))
