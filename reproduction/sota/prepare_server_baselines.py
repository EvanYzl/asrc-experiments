"""One-time isolated server port; preserve every original byte and patch receipt."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
assert str(ROOT)=='/root/zhishitupui','Server-only adaptation; never modify the live Windows baseline queue'
archive=ROOT/'reproduction/sota/history/server_adapter01'
archive.mkdir(parents=True,exist_ok=True)
receipt=[]
def change(path,fn):
    original=path.read_text(encoding='utf-8');new=fn(original)
    if original==new:return
    target=archive/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists(),'Already adapted; inspect before repeating'
    shutil.copy2(path,target);path.write_text(new,encoding='utf-8')
    receipt.append({'path':str(path.relative_to(ROOT)),'before_sha256':hashlib.sha256(original.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest()})

def common_patch(s):
    old="    files = {str(p.relative_to(ROOT)): sha256(p) for p in [*paths.values(), entity_path, relation_path]}"
    assert old in s
    return s.replace(old,"""    frozen_path = SUITE / 'data_manifests' / f'{dataset}_{kg}.json'
    frozen_keys = {}
    if frozen_path.exists():
        frozen_keys = {k.replace('\\\\','/'): k for k in json.loads(frozen_path.read_text(encoding='utf-8'))['files']}
    files = {frozen_keys.get(p.relative_to(ROOT).as_posix(), p.relative_to(ROOT).as_posix()): sha256(p) for p in [*paths.values(), entity_path, relation_path]}""")
change(ROOT/'reproduction/strict_baselines/common.py',common_patch)
for method in ['LSMGA','DMKGC','IMKGC']:
    def patch(s):
        s=s.replace("os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'", "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')")
        s=s.replace("os.environ['CUDA_VISIBLE_DEVICES'] = '0'", "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')")
        if method=='DMKGC':
            s=s.replace("    else:\n        raise ValueError(f'Unsupported dataset: {args.dataset}')", "    elif args.dataset == 'wk3l':\n        all_langs = ['fr','en']\n    else:\n        raise ValueError(f'Unsupported dataset: {args.dataset}')")
        return s
    change(ROOT/f'reproduction/sources/{method}/run_model.py',patch)

source=ROOT/'reproduction/strict_baselines/run_graph_baseline.py'
s=source.read_text(encoding='utf-8')
s=s.replace('from common import *', "sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'strict_baselines'))\nfrom common import *")
s=s.replace('    source=ROOT/f\'data/raw/dmkgc/dataset{dataset}\'', "    if dataset=='wk3l':\n        from wk3l_graph_data import prepare_wk3l\n        return prepare_wk3l(profile)\n    source=ROOT/f'data/raw/dmkgc/dataset{dataset}'",1)
s=s.replace("choices=['dbp5l','depkg','dwy']", "choices=['dbp5l','depkg','dwy','wk3l']")
s=s.replace("data_path=prepare_data(opts.dataset, 'train_only_k10_h2')", "data_path=prepare_data(opts.dataset, f'train_only_k10_h2_{opts.method}_s{opts.seed}')")
s=s.replace("module.get_language_list=lambda directory:list(DOMAINS[opts.dataset])", "module.get_language_list=lambda directory:['fr','en'] if opts.dataset=='wk3l' else list(DOMAINS[opts.dataset])")
s=s.replace("{'dbp5l':80,'depkg':50,'dwy':50}", "{'dbp5l':80,'depkg':50,'dwy':50,'wk3l':50}")
s=s.replace("opts.dataset if opts.method!='LSMGA' or opts.dataset!='dwy' else 'dbp5l'", "'dbp5l' if opts.method=='LSMGA' and opts.dataset in ['dwy','wk3l'] else opts.dataset")
s=s.replace("                ids=torch.as_tensor(batch.copy(),device=self.device)", "                ids=torch.as_tensor(batch.copy(),device=self.device)\n                if opts.dataset=='wk3l' and kgname=='en':ids[:,1]+=load_kg('wk3l','fr')['relations']")
s=s.replace("            for kgindex,kgname in enumerate(self.kg_objects_dict):", "            for kgindex,kgname in enumerate(self.kg_objects_dict):\n                if split=='test' and kgname not in DOMAINS[opts.dataset]:continue")
s=s.replace("mm=macro_metrics({k:v['metrics']['select'] for k,v in results.items()})", "mm=macro_metrics({k:v['metrics']['select'] for k,v in results.items() if k in DOMAINS[opts.dataset]})")
s=s.replace("test=tester.evaluate_split('test',save=True)", "test=tester.evaluate_split('val_select' if opts.smoke else 'test',save=True)")
s=s.replace("for kg,result in test.items()}", "for kg,result in test.items() if kg in DOMAINS[opts.dataset]}")
s=s.replace("'domain_order':DOMAINS[opts.dataset],", "'domain_order':DOMAINS[opts.dataset],\n            'wk3l_adaptation':'FR target and EN_F source; original entity IDs; FR relation IDs unchanged, EN relations offset by FR dictionary size; train-only graph; all supplied alignments; 50 LSMGA rounds / 30 DMKGC or IMKGC rounds; baseline recipe frozen before test' if opts.dataset=='wk3l' else None,")
target=ROOT/'reproduction/sota/run_graph_sota.py'
target.write_text(s,encoding='utf-8')
receipt.append({'new_adapter':str(target.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sha256':hashlib.sha256(s.encode()).hexdigest()})
(archive/'patch_receipt.json').write_text(json.dumps(receipt,indent=2))
sys.path.insert(0,str(ROOT/'reproduction/strict_baselines'))
from common import load_kg,DOMAINS
for ds,kgs in DOMAINS.items():
    for kg in kgs:load_kg(ds,kg)
load_kg('wk3l','en')
print(json.dumps({'frozen_manifests_verified':16,'patches':len(receipt)}))
