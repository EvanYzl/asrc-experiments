"""Verify frozen inputs and prepare deterministic graph caches before parallel targets."""
from pathlib import Path
import datetime as dt
import gc
import hashlib
import json
import os
import sys
import traceback

BASE=Path(__file__).resolve().parent
ROOT=Path('/root/zhishitupui')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)
def main():
    import numpy as np
    import torch
    import torch_geometric
    import torch_sparse
    import torch_scatter
    import scipy
    import sklearn
    import psutil
    plan=json.loads((BASE/'PLAN.json').read_text())
    for rel,digest in json.loads((BASE/'BUNDLE_HASHES.json').read_text()).items():
        assert sha(BASE/rel)==digest,('bundle',rel)
    for rel,digest in plan['input_sha256'].items():assert sha(ROOT/rel)==digest,('raw',rel)
    for ds,files in plan['frozen_feature_sha256'].items():
        for rel,digest in files.items():assert sha(BASE/'runtime_ssaga/ssaga_data'/f'dataset{ds}'/rel)==digest,('features',ds,rel)
    sys.path.insert(0,str(BASE/'runtime_ssaga'))
    from common import load_kg,DOMAINS
    from run_ssaga import prepare
    for ds in ['dbp5l','depkg']:
        prepare(ds)
        for kg in DOMAINS[ds]:load_kg(ds,kg)
    assert torch.__version__=='2.10.0+cu128' and torch.cuda.device_count()==6
    assert torch_geometric.__version__=='2.7.0'
    sys.path.insert(0,str(BASE/'sources/SS-AGA'))
    from src.utils import get_subgraph_list
    caches={}
    for ds in ['dbp5l','depkg']:
        data=BASE/'runtime_ssaga/ssaga_data'/f'dataset{ds}'
        relation_count=sum(1 for _ in (data/'relations.txt').open(encoding='utf-8'))+1
        node_base=relation_base=0
        for kg in DOMAINS[ds]:
            n=sum(1 for _ in (data/'entity'/f'{kg}.tsv').open(encoding='utf-8'))
            for role in [True,False]:
                graphs=get_subgraph_list(str(data),kg,role,n,2,10,node_base,relation_base)
                assert len(graphs)==n,(ds,kg,role,len(graphs),n)
                file=data/'ssaga_cache'/f"{kg}_{'target' if role else 'support'}_h2_k10.graph"
                caches[file.relative_to(BASE).as_posix()]=sha(file)
                print(f"Verified SS-AGA cache {ds}/{kg}/{'target' if role else 'support'}",flush=True)
                del graphs;gc.collect()
            node_base+=n;relation_base+=relation_count
    assert len(caches)==22
    receipt={'status':'completed','finished_at':dt.datetime.now(dt.timezone.utc).isoformat(),'raw_files':len(plan['input_sha256']),
        'manifest_count':11,'cache_sha256':caches,'features':plan['frozen_feature_sha256'],'torch':torch.__version__,
        'pyg':torch_geometric.__version__,'numpy':np.__version__,'gpu_count':torch.cuda.device_count(),
        'memory_available_gb':psutil.virtual_memory().available/2**30,'cache_generation':'Exact frozen deterministic CSR construction; training RNG is untouched'}
    save(BASE/'PREFLIGHT.json',receipt)
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        save(BASE/'PREFLIGHT_FAILED.json',{'time':dt.datetime.now(dt.timezone.utc).isoformat(),'error':traceback.format_exc()})
        raise
