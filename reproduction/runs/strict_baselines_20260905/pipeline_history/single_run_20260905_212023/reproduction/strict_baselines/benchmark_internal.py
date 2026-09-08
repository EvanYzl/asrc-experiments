"""Measured end-to-end Target-only/Uniform-All cost on a frozen 5000-query set."""
import argparse
import gc
from pathlib import Path

from common import *
from internal_data import INTERNAL_VERSION,load_setting
from internal_models import load_target
from run_internal_controls import load_teacher


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DOMAINS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--teacher-root',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    opts=p.parse_args();seed_all(opts.seed);out=Path(opts.output).resolve();out.mkdir(parents=True,exist_ok=True)
    domains=DOMAINS[opts.dataset];data={kg:load_setting(opts.dataset,kg) for kg in domains}
    amount=50 if opts.smoke else 5000;warmups=1 if opts.smoke else 3;passes=1 if opts.smoke else 10
    rng=np.random.default_rng(20260905);workload=[]
    for k,kg in enumerate(domains):
        n=amount//len(domains)+int(k<amount%len(domains));assert len(data[kg]['arrays']['test'])>=n
        for q in rng.choice(len(data[kg]['arrays']['test']),n,replace=False):workload.append((k,int(q)))
    workload=np.asarray(workload,dtype=np.int32);workload=workload[rng.permutation(len(workload))]
    np.save(out/'fixed_workload.npy',workload)
    known={kg:tail_map([d['arrays']['train'],d['arrays']['valid']]+([d['arrays']['test']] if opts.dataset=='wk3l' else [])) for kg,d in data.items()}
    config=json.loads((Path(opts.teacher_root)/domains[0]/'config.json').read_text())['recipe']
    base_root=config['base_root'];summaries={}
    atomic_json(out/'config.json',{'version':INTERNAL_VERSION,'dataset':opts.dataset,'seed':opts.seed,'teacher_root':opts.teacher_root,
        'base_root':base_root,'queries':amount,'warmup_passes':warmups,'measured_passes':passes,'latency_batch_size':1,'throughput_batch_size':256,
        'query_selection':'fixed seed 20260905; equal KG quotas then shuffle, no replacement','cache_state':'warm OS page cache and warmed CUDA kernels; CSR remains read-only mmap',
        'scope':'score all target entities, source CSR fetch and message encoding when used, filter and rank current gold, CUDA synchronization',
        'environment':version_info(),'source_hashes':source_hashes([Path(__file__),SUITE/'internal_models.py',SUITE/'internal_data.py',SUITE/'common.py']),
        'workload_sha256':sha256(out/'fixed_workload.npy')})
    for method in ['Target-only','Uniform-All']:
        name='target' if method=='Target-only' else 'uniform';dest=out/name;dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():summaries[name]=json.loads((dest/'result.json').read_text());continue
        shared={kg:load_target(opts.dataset,kg,base_root).cuda().eval() for kg in (['en','fr'] if opts.dataset=='wk3l' else domains)}
        for base in shared.values():
            for parameter in base.parameters():parameter.requires_grad_(False)
        models={kg:shared[kg] for kg in domains}
        if name=='uniform':
            models={kg:load_teacher(opts,kg) for kg in domains}
            # Each frozen KG encoder is physically shared across target models.
            for kg,model in models.items():
                assert torch.equal(model.base.entity.weight,shared[kg].entity.weight)
                model.base=shared[kg]
                for s in model.sources:
                    assert torch.equal(model.source_models[s].entity.weight,shared[s].entity.weight)
                    model.source_models[s]=shared[s]
        elif opts.dataset=='wk3l':
            # EN is unnecessary for the standalone FR baseline.
            del shared['en']
        parameters={id(p):p for m in models.values() for p in m.parameters()}
        count=sum(p.numel() for p in parameters.values())
        torch.cuda.empty_cache();events=[];throughput=[]
        def scores(kg,batch):
            if name=='target':return models[kg].all_scores(torch.as_tensor(batch.copy(),device='cuda'))
            return models[kg].all_scores_numpy(batch)
        with ResourceTrace(dest/'rss_trace.csv') as trace,torch.no_grad():
            for repeat in range(-warmups,passes):
                for ki,qi in workload:
                    kg=domains[int(ki)];batch=data[kg]['arrays']['test'][int(qi):int(qi)+1]
                    torch.cuda.synchronize();started=time.perf_counter()
                    rank=filtered_ranks(scores(kg,batch),batch,known[kg])
                    torch.cuda.synchronize();seconds=time.perf_counter()-started
                    if name=='uniform':
                        aligned,available,edges=models[kg].availability(batch[:,0]);edge_count=int(edges.sum());source_count=int(available.sum())
                    else:edge_count=source_count=0
                    events.append((repeat,int(ki),int(qi),seconds,int(rank[0]),edge_count,source_count))
                print(f'benchmark {opts.dataset} {method} seed={opts.seed} pass={repeat}',flush=True)
            for repeat in range(passes):
                torch.cuda.synchronize();started=time.perf_counter()
                for ki,kg in enumerate(domains):
                    selected=workload[workload[:,0]==ki,1]
                    for start in range(0,len(selected),256):
                        batch=data[kg]['arrays']['test'][selected[start:start+256]]
                        filtered_ranks(scores(kg,batch),batch,known[kg])
                torch.cuda.synchronize();elapsed=time.perf_counter()-started
                throughput.append((repeat,len(workload),elapsed,len(workload)/elapsed))
        events=np.asarray(events,dtype=np.float64);throughput=np.asarray(throughput,dtype=np.float64)
        np.savez_compressed(dest/'timing_events.npz',events=events,event_columns=np.array(['pass','kg_index','query_index','seconds','rank','edges','sources']),
            throughput=throughput,throughput_columns=np.array(['pass','queries','seconds','qps']),languages=np.array(domains))
        measured=events[events[:,0]>=0]
        # Identical query ranks across measured passes are a correctness guard.
        assert np.all(measured[:,4].reshape(passes,len(workload))==measured[:len(workload),4])
        summary={'status':'completed','method':method,'dataset':opts.dataset,'seed':opts.seed,'parameter_count':count,
            'query_count':len(workload),'measured_passes':passes,'warmup_passes':warmups,'resources':trace.summary,
            'p50_ms':float(np.quantile(measured[:,3],.5)*1000),'p95_ms':float(np.quantile(measured[:,3],.95)*1000),
            'qps':float(throughput[:,1].sum()/throughput[:,2].sum()),'edges':float(measured[:,5].mean()),'sources':float(measured[:,6].mean()),
            'timing_sha256':sha256(dest/'timing_events.npz'),'workload_sha256':sha256(out/'fixed_workload.npy')}
        atomic_json(dest/'result.json',summary);summaries[name]=summary
        del models,shared,parameters;gc.collect();torch.cuda.empty_cache()
    atomic_json(out/'result.json',{'status':'completed','publication':False,'full_data':not opts.smoke,'kind':'internal-benchmark',
        'dataset':opts.dataset,'seed':opts.seed,'summaries':summaries,'protocol':INTERNAL_VERSION})


if __name__=='__main__':main()
