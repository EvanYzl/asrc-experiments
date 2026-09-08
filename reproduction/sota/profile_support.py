"""Isolated, fixed-validation-query filtered ranking and top-10 benchmark."""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch
from frozen_data import ROOT,DOMAINS,atomic_json,sha256
from support_data import build_inputs,array_hash
from train_complex import Complex
sys.path.insert(0,str(ROOT/'reproduction/strict_baselines'))
from common import tail_map,filtered_ranks,ResourceTrace

def other_gpu_processes():
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return [int(x.strip()) for x in rows if x.strip().isdigit() and int(x.strip())!=os.getpid()]

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);ident=p.parse_args().id
    phase=ROOT/'reproduction/sota/paper_support';freeze=json.loads((phase/'EVALUATION_FREEZE.json').read_text());e=freeze['models'][ident];spec=freeze['profiling']
    assert e['variant'] in ['independent','shared'] and not other_gpu_processes(),'Profiling requires all GPUs otherwise idle'
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(spec['gpu'])
    for path,digest in freeze['source_hashes'].items():assert sha256(ROOT/path)==digest
    cp=ROOT/e['training_path']/'best.pt';assert sha256(cp)==e['checkpoint_sha256']
    out=phase/'profiling'/ident;out.mkdir(parents=True,exist_ok=True);assert not (out/'result.json').exists()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data,maps,roff,n,nr,_,_,_=build_inputs(e['recipe']);maps={k:torch.as_tensor(v,device='cuda') for k,v in maps.items()}
    saved=torch.load(cp,map_location='cpu',weights_only=False);model=Complex(n,nr,e['recipe']['rank']).cuda();model.load_state_dict(saved['model']);del saved;gc.collect();model.eval()
    atomic_json(out/'config.json',{'id':ident,'spec':spec,'checkpoint_sha256':e['checkpoint_sha256'],'freeze_sha256':sha256(phase/'EVALUATION_FREEZE.json'),
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'pid':os.getpid(),'test_access':False,'optimizer_loaded_to_gpu':False})
    per={}
    for kg in DOMAINS[e['dataset']]:
        d=data[kg];queries=d['arrays']['val_select'][:spec['queries_per_kg']];mapping=maps[kg]
        known=tail_map([d['arrays']['train'],d['arrays']['val_select']]);kgout=out/kg;kgout.mkdir(exist_ok=True)
        def operation(batch):
            b=torch.as_tensor(batch.copy(),device='cuda');scores=model.scores(mapping[b[:,0]],b[:,1]+roff[kg],mapping)
            assert torch.isfinite(scores).all()
            ranks=filtered_ranks(scores,batch,known)
            gold=scores.gather(1,b[:,2,None]).cpu().numpy().ravel();filtered=scores.clone()
            for i,(h,r,t) in enumerate(batch):
                ids=[v for v in known.get((int(h),int(r)),()) if v!=int(t)]
                if ids:filtered[i,ids]=-torch.inf
            top=torch.argsort(filtered,dim=1,descending=True,stable=True)[:,:10]
            return ranks,gold,top.cpu().numpy().astype(np.int32),filtered.gather(1,top).cpu().numpy()
        traces={};sample=None
        with ResourceTrace(kgout/'resources.csv') as trace:
            for bs in [spec['batch_latency'],spec['batch_throughput']]:
                rows=[]
                for repetition in range(-spec['warmup_passes'],spec['measured_passes']):
                    for start in range(0,len(queries),bs):
                        batch=queries[start:start+bs];torch.cuda.synchronize();t0=time.perf_counter()
                        sample=operation(batch);torch.cuda.synchronize();elapsed=time.perf_counter()-t0
                        if repetition>=0:rows.append([repetition,start,len(batch),elapsed])
                traces[bs]=np.asarray(rows,dtype=np.float64)
        assert not other_gpu_processes(),'Concurrent GPU work invalidated this timing run'
        lat=traces[1][:,3]*1000;throughput=traces[256][:,2].sum()/traces[256][:,3].sum()
        np.savez_compressed(kgout/'timings.npz',triples=queries,latency=traces[1],throughput=traces[256],
            sample_rank=sample[0],sample_gold=sample[1],sample_top10_ids=sample[2],sample_top10_scores=sample[3])
        per[kg]={'queries':len(queries),'query_sha256':array_hash(queries),'p50_ms':float(np.quantile(lat,.5)),
                 'p95_ms':float(np.quantile(lat,.95)),'qps':float(throughput),**trace.summary}
    macro={k:float(np.mean([v[k] for v in per.values()])) for k in ['p50_ms','p95_ms','qps']}
    macro.update(vram_mib=max(v['peak_cuda_allocated_bytes'] for v in per.values())/2**20,
                 rss_mib=max(v['peak_rss_bytes'] for v in per.values())/2**20,params_m=e['parameters']/1e6)
    result={'status':'completed','purpose':'isolated_validation_query_profiling','test_access':False,'id':ident,'seed':e['seed'],
            'macro':macro,'per_kg':per,'parameters':sum(x.numel() for x in model.parameters()),'spec':spec,'hardware':torch.cuda.get_device_name()}
    assert result['parameters']==e['parameters'];atomic_json(out/'result.json',result);print(json.dumps({'id':ident,'macro':macro}),flush=True)

if __name__=='__main__':main()
