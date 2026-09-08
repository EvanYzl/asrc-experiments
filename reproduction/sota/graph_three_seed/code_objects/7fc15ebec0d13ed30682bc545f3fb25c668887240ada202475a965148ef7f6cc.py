"""Shared, versioned data and full-candidate evaluation for the KBS baseline runs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / 'reproduction' / 'strict_baselines'
DOMAINS = {'dbp5l': ['el', 'en', 'es', 'fr', 'ja'],
           'depkg': ['de', 'es', 'fr', 'it', 'jp', 'uk'],
           'dwy': ['db', 'wk', 'yg'], 'wk3l': ['fr']}
PROTOCOL_VERSION = 'kbs-baselines-v1-20260905'
SPLIT_SEED = 20260905


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def version_info():
    return {'protocol': PROTOCOL_VERSION, 'python': __import__('sys').version,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'numpy': np.__version__, 'device': torch.cuda.get_device_name(0)}


def validation_partition(triples):
    """Relation-stratified 50/50 split; identical triples cannot cross folds."""
    groups = {}
    for i, triple in enumerate(triples):
        groups.setdefault(tuple(map(int, triple)), []).append(i)
    by_relation = {}
    for triple in groups:
        by_relation.setdefault(triple[1], []).append(triple)
    select, cert = [], []
    total_select = total_cert = 0
    for relation, keys in sorted(by_relation.items()):
        keys.sort(key=lambda k: hashlib.sha256(f'{SPLIT_SEED}:{k}'.encode()).digest())
        a = b = 0
        for key in keys:
            indices = groups[key]
            choose_select = a < b or (a == b and total_select <= total_cert)
            if choose_select:
                select.extend(indices); a += len(indices); total_select += len(indices)
            else:
                cert.extend(indices); b += len(indices); total_cert += len(indices)
    return np.array(sorted(select), dtype=np.int64), np.array(sorted(cert), dtype=np.int64)


def load_kg(dataset, kg):
    if dataset == 'wk3l':
        folder = ROOT / 'data/raw/atransn' / ('WK3l-15k_EN_F' if kg == 'en' else 'WK3l-15k_FR')
        paths = {s: folder / f'{v}_triple_id.txt' for s, v in [('train','train'),('valid','valid'),('test','test')]}
        entity_path, relation_path = folder / 'entity_dict.txt', folder / 'relation_dict.txt'
    else:
        folder = ROOT / f'data/raw/dmkgc/dataset{dataset}'
        paths = {s: folder / 'kg' / f'{kg}-{v}.tsv' for s, v in [('train','train'),('valid','val'),('test','test')]}
        entity_path, relation_path = folder / 'entity' / f'{kg}.tsv', folder / 'relations.txt'
    arrays = {s: np.loadtxt(p, dtype=np.int64, delimiter='\t', ndmin=2) for s,p in paths.items()}
    nentities = sum(1 for _ in entity_path.open(encoding='utf-8'))
    nrelations = sum(1 for _ in relation_path.open(encoding='utf-8'))
    for name, a in arrays.items():
        assert a.shape[1] == 3 and a.min() >= 0
        assert a[:,[0,2]].max() < nentities and a[:,1].max() < nrelations
    select, cert = validation_partition(arrays['valid'])
    arrays['val_select'], arrays['val_cert'] = arrays['valid'][select], arrays['valid'][cert]
    frozen_path = SUITE / 'data_manifests' / f'{dataset}_{kg}.json'
    frozen_keys = {}
    if frozen_path.exists():
        frozen_keys = {k.replace('\\','/'): k for k in json.loads(frozen_path.read_text(encoding='utf-8'))['files']}
    files = {frozen_keys.get(p.relative_to(ROOT).as_posix(), p.relative_to(ROOT).as_posix()): sha256(p) for p in [*paths.values(), entity_path, relation_path]}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {'dataset': dataset, 'kg': kg, 'entities': nentities, 'relations_dictionary': nrelations,
                'relations_observed': len(np.unique(np.concatenate([a[:,1] for a in [arrays['train'], arrays['valid'], arrays['test']]]))),
                'counts': {s: len(a) for s,a in arrays.items()}, 'files': files, 'dataset_hash': digest,
                'split_seed': SPLIT_SEED, 'val_select_indices': select.tolist(), 'val_cert_indices': cert.tolist(),
                'split_rule': 'relation-stratified; identical validation triples grouped; balanced greedy hashed order',
                'selection_filter': 'train+val_select; no val_cert or test labels',
                'test_filter': 'all' if dataset == 'wk3l' else 'train+valid'}
    manifest_path = SUITE / 'data_manifests' / f'{dataset}_{kg}.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding='utf-8')) == manifest, 'Frozen data manifest changed'
    else:
        atomic_json(manifest_path, manifest)
    return {'dataset':dataset,'kg':kg,'entities':nentities,'relations':nrelations,
            'arrays':arrays,'manifest':manifest,'valid_indices':select,'cert_indices':cert}


def tail_map(arrays):
    result = {}
    for a in arrays:
        for h,r,t in a:
            result.setdefault((int(h),int(r)),set()).add(int(t))
    return result


def filtered_ranks(scores, triples, known):
    """Higher is better; retain current gold; deterministic ascending-ID ties."""
    device = scores.device
    gold = torch.as_tensor(triples[:,2].copy(), device=device)
    gold_scores = scores.gather(1, gold[:,None]).squeeze(1)
    ids = torch.arange(scores.shape[1], device=device)
    ahead = (scores > gold_scores[:,None]) | ((scores == gold_scores[:,None]) & (ids[None,:] < gold[:,None]))
    rows, cols = [], []
    for i,(h,r,t) in enumerate(triples):
        tails = known.get((int(h),int(r)), ())
        rows.extend([i]*len(tails)); cols.extend(tails)
    if rows:
        ahead[torch.tensor(rows,device=device),torch.tensor(cols,device=device)] = False
    ahead.scatter_(1,gold[:,None],False)
    return ahead.sum(1).cpu().numpy().astype(np.int32)+1


def metrics(ranks):
    r = np.asarray(ranks, dtype=np.float64)
    assert len(r) and np.isfinite(r).all() and (r >= 1).all()
    return {'mrr':float((1/r).mean()), 'h1':float((r<=1).mean()),
            'h3':float((r<=3).mean()),'h10':float((r<=10).mean()),'n':len(r)}


def macro_metrics(per_kg):
    names = list(per_kg)
    return {k: float(np.mean([per_kg[n][k] for n in names])) for k in ('mrr','h1','h3','h10')}


@torch.no_grad()
def evaluate(data, score_fn, split='test', batch_size=64, output=None, limit=None):
    """Save full-precision ranks/scores/top-10 without storing query x entity tensors."""
    triples = data['arrays'][split]
    if limit:
        triples = triples[:limit]
    a = data['arrays']
    if split == 'val_select':
        filters = {'select':tail_map([a['train'], a['val_select']])}
        primary = 'select'
    else:
        filters = {'train':tail_map([a['train']]), 'train_valid':tail_map([a['train'],a['valid']]),
                   'all':tail_map([a['train'],a['valid'],a['test']])}
        primary = 'all' if data['dataset']=='wk3l' else 'train_valid'
    ranks = {k:[] for k in filters}
    gold_scores, top_ids, top_scores, timings = [], [], [], []
    for start in range(0,len(triples),batch_size):
        batch = triples[start:start+batch_size]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        scores = score_fn(batch)
        assert scores.shape == (len(batch), data['entities'])
        assert torch.isfinite(scores).all(), 'nonfinite ranking score'
        for protocol, known in filters.items():
            ranks[protocol].append(filtered_ranks(scores,batch,known))
        if output:
            gold = torch.tensor(batch[:,2].copy(),device=scores.device)
            gold_scores.append(scores.gather(1,gold[:,None]).cpu().numpy().ravel())
            filtered = scores.clone()
            for i,(h,r,t) in enumerate(batch):
                ids = [e for e in filters[primary].get((int(h),int(r)),()) if e!=int(t)]
                if ids: filtered[i,ids] = -torch.inf
            # Stable sorting is intentional: exported cases share the rank tie rule.
            top = torch.argsort(filtered,dim=1,descending=True,stable=True)[:,:10]
            top_ids.append(top.cpu().numpy().astype(np.int32))
            top_scores.append(filtered.gather(1,top).cpu().numpy())
        torch.cuda.synchronize()
        timings.append([start,len(batch),time.perf_counter()-t0])
    rank_arrays = {k:np.concatenate(v) for k,v in ranks.items()}
    result = {'primary_filter': primary, 'metrics':{k:metrics(v) for k,v in rank_arrays.items()},
              'count':len(triples),'batch_seconds':float(sum(x[2] for x in timings))}
    if output:
        output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, triples=triples.astype(np.int32), query_index=np.arange(len(triples)),
                            gold_score=np.concatenate(gold_scores),top10_ids=np.concatenate(top_ids),
                            top10_scores=np.concatenate(top_scores),batch_timings=np.asarray(timings),
                            **{f'rank_{k}':v for k,v in rank_arrays.items()})
        atomic_json(output.with_suffix('.json'),result)
    return result


class ResourceTrace:
    def __init__(self, path):
        self.path = Path(path); self.stop_event = threading.Event(); self.peak_rss = 0
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        def run():
            proc=psutil.Process()
            with self.path.open('w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f); writer.writerow(['elapsed_s','rss_bytes','system_available_bytes'])
                while not self.stop_event.is_set():
                    rss=proc.memory_info().rss; self.peak_rss=max(self.peak_rss,rss)
                    writer.writerow([time.perf_counter()-self.start,rss,psutil.virtual_memory().available])
                    self.stop_event.wait(.1)
        self.thread=threading.Thread(target=run,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.stop_event.set(); self.thread.join()
        self.summary={'wall_seconds':time.perf_counter()-self.start,'peak_rss_bytes':self.peak_rss,
                      'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                      'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()}
        atomic_json(self.path.with_suffix('.json'),self.summary)


def atomic_checkpoint(path,payload):
    path=Path(path); tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); os.replace(tmp,path)


def source_hashes(paths):
    result={};objects=ROOT/'reproduction/runs/strict_baselines_20260905/code_objects';objects.mkdir(exist_ok=True)
    for p in paths:
        p=Path(p);content=p.read_bytes();digest=hashlib.sha256(content).hexdigest()
        artifact=objects/(digest+p.suffix)
        if not artifact.exists():artifact.write_bytes(content)
        result[str(p.relative_to(ROOT))]=digest
    return result
