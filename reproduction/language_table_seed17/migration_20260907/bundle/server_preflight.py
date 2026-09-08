"""Read-only verification of uploaded code, input splits, and resumable state."""
from pathlib import Path
import hashlib
import json
import importlib.util
import sys
import torch
import torch_geometric
import numba

BASE=Path(__file__).resolve().parent
ROOT=Path('/root/zhishitupui')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    d=json.loads((BASE/'DEPLOYMENT.json').read_text())
    for rel,digest in json.loads((BASE/'BUNDLE_HASHES.json').read_text()).items():assert sha(BASE/rel)==digest,rel
    for rel,digest in d['input_sha256'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(BASE/'teacher_fr/checkpoint_valid.pt')==d['teacher_sha256']
    assert torch.__version__=='2.10.0+cu128' and torch_geometric.__version__=='2.7.0'
    out={'status':'passed','torch':torch.__version__,'pyg':torch_geometric.__version__,'numba':numba.__version__,'raw_inputs_verified':len(d['input_sha256'])}
    verified={}
    for runtime,ds,kgs in [('runtime_transfer','wk3l',['en','fr']),('runtime_epkg','depkg',['de','es','fr','it','jp','uk'])]:
        spec=importlib.util.spec_from_file_location(runtime+'_common',BASE/runtime/'common.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for kg in kgs:
            frozen=module.load_kg(ds,kg)
            verified[ds+'/'+kg]=frozen['manifest']['dataset_hash']
    out['frozen_manifests_verified']=verified
    if '--checkpoint' in sys.argv:
        cp=BASE/'jobs/imkgc_depkg_s17'
        r=json.loads((cp/'MIGRATION_CHECKPOINT.json').read_text())
        for rel,digest in r['sha256'].items():assert sha(cp/rel)==digest,rel
        last=torch.load(cp/'last.pt',map_location='cpu',weights_only=False)
        best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False)
        assert last['format']=='multidomain_full_state_v1'
        required=['model_state_dict','optimizer_state_dict','scheduler_state_dict','python_rng_state','numpy_rng_state','torch_rng_state','cuda_rng_state_all']
        assert all(k in last and last[k] is not None for k in required)
        assert last['completed_round']+1==r['completed_rounds']<50
        assert set(last['ordered_domains'])=={'de','es','fr','it','jp','uk'}
        assert best['evaluation_index']<=r['completed_rounds']+1
        out.update(completed_rounds=r['completed_rounds'],best_evaluation=best['evaluation_index'],full_state_verified=True)
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
