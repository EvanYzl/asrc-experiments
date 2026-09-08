"""Fill only provenance-verified, full-domain, three-seed baseline cells."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import os
import sys

import numpy as np
from table_scope import cell_scope, update_scope

ROOT=Path(__file__).resolve().parents[2]
SUITE=ROOT/'reproduction/strict_baselines'
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'
TABLES=ROOT/'outputs/kbs_main_tables'
DS={'dbp5l':'dbp','depkg':'epkg','dwy':'dwy','wk3l':'wk3l'}
KGS={'dbp5l':['el','en','es','fr','ja'],'depkg':['de','es','fr','it','jp','uk'],'dwy':['db','wk','yg'],'wk3l':['fr']}
SEEDS=[17,29,43]
HASH_CACHE={}


def digest(p):
    p=Path(p);stat=p.stat();key=(str(p.resolve()),stat.st_size,stat.st_mtime_ns)
    if key not in HASH_CACHE:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        HASH_CACHE[key]=h.hexdigest()
    return HASH_CACHE[key]
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)


def input_triples(ds,kg):
    if ds=='wk3l':p=ROOT/'data/raw/atransn/WK3l-15k_FR/test_triple_id.txt'
    else:p=ROOT/f'data/raw/dmkgc/dataset{ds}/kg/{kg}-test.tsv'
    return np.loadtxt(p,dtype=np.int64,delimiter='\t',ndmin=2)


def audited_result(path):
    obj=json.loads(path.read_text(encoding='utf-8'))
    if obj.get('status')!='completed' or not obj.get('full_data') or obj.get('teacher') or obj.get('publication') is False or 'kg' in obj:return None
    method,ds,seed=obj['method'],obj['dataset'],obj['seed']
    if seed not in SEEDS or ds not in DS:return None
    domains=obj.get('domains',obj.get('per_kg',{}))
    condition=obj.get('condition','full')
    expected_kgs=[condition.split('_')[0]] if condition.startswith(('el_','ja_')) else KGS[ds]
    if sorted(domains)!=sorted(expected_kgs):raise ValueError(f'Incomplete domains: {path}')
    if method=='KEnS':return audited_kens(path,obj)
    if method in ['Global-Utility','Random']:return None # Await a frozen QURA validation-budget match.
    per_kg={};rows=[];checkpoints=[];hashes=[];paired={};data_hashes=set();code_hashes=set();sensitivity=[]
    for kg in expected_kgs:
        qpath=path.parent/kg/'test_queries.npz'
        with np.load(qpath) as q:
            truth=input_triples(ds,kg)
            assert np.array_equal(q['triples'],truth),f'Query mismatch: {qpath}'
            assert np.array_equal(q['query_index'],np.arange(len(truth)))
            key='rank_train' if method=='SS-AGA' else ('rank_all' if ds=='wk3l' else 'rank_train_valid')
            rank=q[key].astype(np.float64)
            assert len(rank)==len(truth) and np.isfinite(rank).all() and (rank>=1).all()
            mm={'mrr':float((1/rank).mean()),'h1':float((rank<=1).mean()),'h3':float((rank<=3).mean()),'h10':float((rank<=10).mean())}
            per_kg[kg]=mm
            for metric,value in mm.items():rows.append({'run_id':str(path.parent.relative_to(RUN/'jobs')),'method':method,'dataset':ds,'condition':condition,'kg':kg,'seed':seed,'metric':metric,'value':value,'queries':len(rank),'query_artifact':str(qpath),'query_sha256':digest(qpath)})
            mask_path=RUN/'results/overlap_masks'/f'{ds}_{kg}.npz'
            if mask_path.exists():
                with np.load(mask_path) as mask:
                    for filter_name in ['test_seen_train','test_seen_train_or_valid']:
                        kept=~mask[filter_name];r=rank[kept]
                        if len(r):sensitivity.append({'run_id':str(path.parent.relative_to(RUN/'jobs')),'method':method,'dataset':ds,'kg':kg,'condition':condition,'seed':seed,
                            'excluded_queries':filter_name,'n':len(r),'removed_n':int((~kept).sum()),'mrr':float((1/r).mean()),'h1':float((r<=1).mean()),'h10':float((r<=10).mean()),
                            'note':'same selected model and original filtering; evaluation query subset only, no retraining'})
        ckpt=Path(domains[kg]['checkpoint']) if 'domains' in obj else Path(obj['checkpoint'])
        expected=domains[kg]['checkpoint_sha256'] if 'domains' in obj else obj['checkpoint_sha256']
        assert ckpt.exists() and digest(ckpt)==expected,f'Checkpoint hash mismatch: {ckpt}'
        checkpoints.append(str(ckpt));hashes.append(expected)
        config_path=Path(domains[kg]['config']) if domains[kg].get('config') else path.parent/kg/'config.json'
        if not config_path.exists():
            for directory in [path.parent,*path.parent.parents]:
                if directory==RUN/'jobs':break
                if (directory/'config.json').exists():config_path=directory/'config.json';break
        if config_path.exists():
            config=json.loads(config_path.read_text(encoding='utf-8'))
            source=config.get('source_hashes',{})
            if source:
                code_hashes.add(hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest())
                for source_name,checksum in source.items():
                    artifact=RUN/'code_objects'/(checksum+Path(source_name).suffix)
                    assert artifact.exists() and digest(artifact)==checksum,f'Missing exact code snapshot: {source_name}'
            if config.get('dataset_hash'):data_hashes.add(config['dataset_hash'])
            elif config.get('data_manifest'):data_hashes.add(hashlib.sha256(json.dumps(config['data_manifest'],sort_keys=True).encode()).hexdigest())
        frozen=SUITE/'data_manifests'/f'{ds}_{kg}.json'
        if frozen.exists():data_hashes.add(json.loads(frozen.read_text())['dataset_hash'])
        if method in ['Uniform-All','Query Attention']:
            with np.load(path.parent/kg/'target_reference_queries.npz') as baseline,np.load(path.parent/kg/'source_trace.npz') as sources:
                assert np.array_equal(baseline['triples'],truth)
                r0=baseline[key].astype(np.float64);available=sources['available'];edges=sources['edges']
                assert available.shape==edges.shape and len(available)==len(truth)
                assert ((edges>=0)&(edges<=32)).all() and (edges[~available]==0).all()
                if 'weights' in sources:
                    weights=sources['weights'];assert weights.shape==available.shape and np.isfinite(weights).all() and (weights>=0).all()
                    assert np.allclose(weights.sum(1),available.any(1),atol=1e-6)
                paired[kg]={'mrr':mm['mrr'],'ntr_all':float((rank>r0).mean()),'ptr':float((rank<r0).mean()),
                    'mean_harm':float(np.maximum(1/r0-1/rank,0).mean()),'coverage':float(available.any(1).mean()),'edges':float(edges.sum(1).mean())}
    for dep in obj.get('dependencies',[]):assert digest(dep['path'])==dep['sha256']
    macro={key:float(np.mean([v[key] for v in per_kg.values()])) for key in ['mrr','h1','h3','h10']}
    for key in macro:assert abs(macro[key]-obj['macro'][key])<1e-10,(path,key)
    paired_macro={k:float(np.mean([v[k] for v in paired.values()])) for k in next(iter(paired.values()),{})}
    for key,value in paired_macro.items():assert abs(value-obj['paired_macro'][key])<1e-10,(path,key)
    return {'method':method,'dataset':ds,'seed':seed,'condition':condition,'macro':macro,'rows':rows,'run_id':str(path.parent.relative_to(RUN/'jobs')),
            'checkpoints':checkpoints,'checkpoint_hashes':hashes,'path':str(path),'paired_macro':paired_macro,
            'dataset_hashes':sorted(data_hashes),'source_manifest_hashes':sorted(code_hashes),'sensitivity_rows':sensitivity}


def audited_kens(path,obj):
    ds=obj['dataset'];rows=[];per_kg={};checkpoints=[];hashes=[];data_hashes=set();code_hashes=set()
    for kg,info in obj['per_kg'].items():
        qpath=path.parent/kg/'test_queries.npz';truth=input_triples(ds,kg)
        assert digest(qpath)==info['query_sha256']
        with np.load(qpath) as q:
            assert np.array_equal(q['triples'],truth)
            assert np.array_equal(q['query_index'],np.arange(len(truth)))
            mm={}
            for n in [1,3,10]:
                actual=(q[f'final_top{n}']==truth[:,2,None]).any(axis=1)
                assert np.array_equal(actual,q[f'hit_{n}'])
                mm[f'h{n}']=float(actual.mean())
                assert abs(mm[f'h{n}']-info['metrics'][f'h{n}'])<1e-10
                rows.append({'run_id':path.parent.name,'method':'KEnS','dataset':ds,'condition':'full','kg':kg,'seed':obj['seed'],
                    'metric':f'h{n}','value':mm[f'h{n}'],'queries':len(truth),'query_artifact':str(qpath),'query_sha256':digest(qpath)})
            per_kg[kg]=mm
        for rel,expected in info['checkpoints'].items():
            ckpt=path.parent/kg/rel;assert digest(ckpt)==expected
            checkpoints.append(str(ckpt));hashes.append(expected)
        assert digest(path.parent/kg/'ensemble_weights.npz')==info['weight_sha256']
        config=json.loads((path.parent/kg/'config.json').read_text(encoding='utf-8'))
        data_hashes.add(hashlib.sha256(json.dumps(config['raw_file_sha256'],sort_keys=True).encode()).hexdigest())
        code_hashes.add(hashlib.sha256(json.dumps(config['source_hashes'],sort_keys=True).encode()).hexdigest())
        for source,checksum in config['source_hashes'].items():assert digest(RUN/'code_objects'/(checksum+Path(source).suffix))==checksum
    macro={m:float(np.mean([v[m] for v in per_kg.values()])) for m in ['h1','h3','h10']}
    for key in macro:assert abs(macro[key]-obj['macro'][key])<1e-10
    return {'method':'KEnS','dataset':ds,'seed':obj['seed'],'condition':'full','macro':macro,'rows':rows,'run_id':path.parent.name,
        'checkpoints':checkpoints,'checkpoint_hashes':hashes,'path':str(path),'dataset_hashes':sorted(data_hashes),'source_manifest_hashes':sorted(code_hashes)}


def collect_benchmarks(fill):
    groups={}
    for path in (RUN/'jobs').glob('internal_benchmark_*/result.json'):
        obj=json.loads(path.read_text(encoding='utf-8'))
        if obj.get('status')!='completed' or not obj.get('full_data'):continue
        groups.setdefault(obj['dataset'],{})[obj['seed']]=(path,obj)
    for ds,by_seed in groups.items():
        if sorted(by_seed)!=SEEDS:continue
        for short in ['target','uniform']:
            latencies=[];throughput=[];summary=[];access=[];workload_hashes=[]
            for seed,(path,obj) in sorted(by_seed.items()):
                info=obj['summaries'][short];timing=path.parent/short/'timing_events.npz'
                assert digest(timing)==info['timing_sha256'] and info['query_count']==5000 and info['measured_passes']==10 and info['warmup_passes']==3
                assert digest(path.parent/'fixed_workload.npy')==info['workload_sha256']
                workload_hashes.append(info['workload_sha256'])
                with np.load(timing) as records:
                    measured=records['events'][records['events'][:,0]>=0]
                    assert len(measured)==50000 and np.isfinite(measured).all() and (measured[:,3]>0).all()
                    latencies.append(measured[:,3]);access.append(measured[:,5:7]);throughput.append(records['throughput'])
                summary.append(info)
            assert len(set(workload_hashes))==1,'Benchmark workload differs between seeds'
            times=np.concatenate(latencies);speed=np.concatenate(throughput);counts=np.concatenate(access)
            metrics={'params_m':max(s['parameter_count'] for s in summary)/1e6,
                'vram_gb':max(s['resources']['peak_cuda_allocated_bytes'] for s in summary)/1e9,
                'rss_gb':max(s['resources']['peak_rss_bytes'] for s in summary)/1e9,
                'p50_ms':float(np.quantile(times,.5)*1000),'p95_ms':float(np.quantile(times,.95)*1000),
                'qps':float(speed[:,1].sum()/speed[:,2].sum()),'edges':float(counts[:,0].mean()),'sources':float(counts[:,1].mean())}
            for metric,value in metrics.items():
                fill(f'T8.{DS[ds]}.{short}.{metric}',value,f'{value:.2f}','rerun',seed='17;29;43',
                     run_id=';'.join(path.parent.name for path,obj in by_seed.values()),
                     notes='5000 identical fixed queries; 3 warmup and 10 measured passes per seed; raw timing events verified')


def collect_impl(build=True):
    manifest=json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
    active_tables=manifest.get('active_tables',['T1','T2','T3'] if manifest.get('focus')=='tables_1_2_3_without_qura' else ['T1','T2','T3','T4','T7','T8'])
    with (TABLES/'cells_template.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;cells={r['cell_id']:r for r in reader}
    values={};audit=[];metrics_rows=[];sensitivity_rows=[]
    for ident,cell in cells.items():
        cell['source_type']='pending'
        spec=cell_scope(ident)
        cell['notes']=spec['reason'] if spec['scope']=='proposed_method_excluded' else 'Awaiting complete audited local reruns: '+spec['reason']
        if spec['scope']!='proposed_method_excluded' and cell['table_id'] not in active_tables:
            cell['notes']='Deferred by user: only '+','.join(active_tables)+' are filled in the current phase'
    def fill(ident,value,display,source_type,**meta):
        assert ident in cells,ident
        if cells[ident]['table_id'] not in active_tables:return
        cells[ident].update(value=str(value),source_type=source_type,**{k:str(v) for k,v in meta.items() if k in fields})
        values[ident]=display
    # Dataset statistics are recomputed from all currently frozen file statistics.
    stats=list(csv.DictReader((ROOT/'data/manifests/DATASET_STATS.csv').open(encoding='utf-8-sig')))
    align=list(csv.DictReader((ROOT/'data/manifests/ALIGNMENT_STATS.csv').open(encoding='utf-8-sig')))
    for ds,key in DS.items():
        if ds=='wk3l':
            s=list(csv.DictReader((ROOT/'data/manifests/WK3L15K_STATS.csv').open(encoding='utf-8-sig')))
            s=[r for r in s if r['role']=='target_fr']
            totals={k:sum(int(r[k]) for r in s) for k in ['entities','relations','train','validation','test']};totals['alignment_links']=2496
            src=ROOT/'data/manifests/WK3L15K_STATS.csv'
        else:
            s=[r for r in stats if r['dataset']=='dataset'+ds]
            totals={k:sum(int(r[k]) for r in s) for k in ['entities','train','validation','test']}
            totals['relations']=sum(int(r['relation_ids_observed']) for r in s)
            totals['alignment_links']=sum(int(r['rows']) for r in align if r['dataset']=='dataset'+ds)
            src=ROOT/'data/manifests/DATASET_STATS.csv'
        for k,v in totals.items():fill(f'T1.{key}.{k}',v,f'{v:,}','derived',unit='count',run_id='dataset-manifest-audit',dataset_hash=digest(src),notes=str(src))
    # User correction: Table 3 must also contain local reruns. No literature
    # values are filled into any performance cell in this deliverable.
    groups={}
    failures=[]
    for path in sorted((RUN/'jobs').rglob('result.json')):
        try:result=audited_result(path)
        except Exception as e:failures.append({'path':str(path),'error':repr(e)});continue
        if result is None:continue
        groups.setdefault((result['method'],result['dataset'],result['condition']),{})[result['seed']]=result
        audit.append({k:v for k,v in result.items() if k not in ['rows','sensitivity_rows']});metrics_rows.extend(result['rows']);sensitivity_rows.extend(result.get('sensitivity_rows',[]))
    published=[]
    for (method,ds,condition),by_seed in sorted(groups.items()):
        if sorted(by_seed)!=SEEDS:continue
        short={'Target-only':'target','Uniform-All':'uniform','Query Attention':'attn','Global-Utility':'global','Random':'random'}.get(method,method.lower());key=DS[ds];table='T2'
        if method in ['KEnS','AlignKGC','SS-AGA']:
            table='T3';key='dbp5l' if ds=='dbp5l' else 'epkg'
            short={'KEnS':'kens','AlignKGC':'align','SS-AGA':'ssaga'}[method]
        for metric in ['h1','h10'] if method=='KEnS' else ['mrr','h1','h10']:
            if condition!='full':continue
            ident=f'{table}.{key}.{short}.{metric}'
            if ident not in cells:continue
            data=np.array([by_seed[s]['macro'][metric] for s in SEEDS]);mean=float(data.mean());sd=float(data.std(ddof=1))
            fill(ident,mean,rf'\({mean*100:.2f}\mathbin{{\pm}}{sd*100:.2f}\)','rerun',standard_deviation=sd,
                 run_id=';'.join(by_seed[s]['run_id'] for s in SEEDS),seed='17;29;43',unit='fraction displayed as percent',
                 dataset_hash=';'.join(sorted({h for s in SEEDS for h in by_seed[s].get('dataset_hashes',[])})),
                 code_commit='source-manifest-sha256:'+(';'.join(sorted({h for s in SEEDS for h in by_seed[s].get('source_manifest_hashes',[])}))),
                 split='public test; validation-only selection',filter_protocol='method-specific; see run config' if table=='T3' else ('all' if ds=='wk3l' else 'train+valid'),
                 selection_protocol='fixed rounds; public-validation ensemble weights' if method=='KEnS' else 'val_select tail MRR, earliest maximum',
                 candidate_scope='top-n nominations per model; weighted ensemble' if method=='KEnS' else 'all target-KG entities',
                 aggregation='FR only' if ds=='wk3l' else 'equal KG macro; then three-seed mean and sample SD',
                 feature_policy='regenerated mBERT CLS from URI labels' if method=='SS-AGA' else 'structural',
                 checkpoint=';'.join(c for s in SEEDS for c in by_seed[s]['checkpoints']),
                 notes='Raw query ranks and checkpoints validated; source implementation recorded in each config.json')
        if table=='T3':
            fill(f'T3.{key}.{short}.paper_table','17;29;43','3 seeds','rerun',seed='17;29;43',
                 run_id=';'.join(by_seed[s]['run_id'] for s in SEEDS),notes='Local reruns; method-specific protocol, no literature substitution')
        if method in ['Uniform-All','Query Attention','Global-Utility','Random']:
            for metric in ['mrr','ntr_all','mean_harm','ptr','coverage','edges']:
                identifiers=[]
                if condition=='full':identifiers.append(f'T4.{key}.{short}.{metric}')
                if ds=='dbp5l' and method in ['Uniform-All','Random'] and metric in ['mrr','ntr_all']:
                    scope='.'.join(condition.split('_')) if condition.startswith(('el_','ja_')) else f'dbp.{condition}'
                    identifiers.append(f'T7.{scope}.{short}.{metric}')
                data=np.array([by_seed[s]['paired_macro'][metric] for s in SEEDS]);mean=float(data.mean());sd=float(data.std(ddof=1));scale=1 if metric=='edges' else 100
                for ident in identifiers:
                    fill(ident,mean,rf'\({mean*scale:.2f}\mathbin{{\pm}}{sd*scale:.2f}\)','rerun',standard_deviation=sd,
                        run_id=';'.join(by_seed[s]['run_id'] for s in SEEDS),seed='17;29;43',unit='edges/query' if metric=='edges' else 'fraction displayed as percent',
                        aggregation='repeat average within seed; equal KG macro then seed mean and sample SD',
                        checkpoint=';'.join(c for s in SEEDS for c in by_seed[s]['checkpoints']),
                        dataset_hash=';'.join(sorted({h for s in SEEDS for h in by_seed[s].get('dataset_hashes',[])})),
                        notes='Paired raw ranks and source records revalidated; '+('budget matched to frozen QURA validation access' if method in ['Global-Utility','Random'] else 'all available sources'))
        published.append({'method':method,'dataset':ds,'condition':condition,'seeds':SEEDS})
    if 'T8' in active_tables:collect_benchmarks(fill)
    value_content='% Generated from audited result artifacts. Unavailable cells retain explicit placeholders.\n'
    value_content+='\n'.join(r'\SetResult{'+k+'}{'+v+'}' for k,v in sorted(values.items()))+'\n'
    changed=not (TABLES/'values.tex').exists() or (TABLES/'values.tex').read_text(encoding='utf-8')!=value_content
    result_dir=RUN/'results';result_dir.mkdir(exist_ok=True)
    dump(result_dir/'table_fill_audit.json',{'filled_cells':len(values),'three_seed_groups':published,'accepted_runs':audit,'rejected_runs':failures,'proposed_method_excluded':True})
    if metrics_rows:
        with (result_dir/'per_kg_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(metrics_rows[0]));writer.writeheader();writer.writerows(metrics_rows)
    if sensitivity_rows:
        with (result_dir/'nonoverlap_sensitivity.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(sensitivity_rows[0]));writer.writeheader();writer.writerows(sensitivity_rows)
    if changed:
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        (TABLES/f'values_{stamp}.tex').write_text(value_content,encoding='utf-8')
        (TABLES/'values.tex').write_text(value_content,encoding='utf-8')
    with (TABLES/'cells_results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(cells.values())
    scope_summary=update_scope()
    source_mtime=max(f.stat().st_mtime for f in [TABLES/'main.tex',TABLES/'preamble.tex',TABLES/'values.tex',*list((TABLES/'tables').glob('*.tex'))])
    pdf_stale=not (TABLES/'KBS_Main_Text_Tables.pdf').exists() or (TABLES/'KBS_Main_Text_Tables.pdf').stat().st_mtime < source_mtime
    if build and (changed or pdf_stale):
            compiler=Path('C:/Users/evan/.codex/plugins/cache/openai-bundled/latex/0.2.6/scripts/compile_latex.py')
            built=subprocess.run([sys.executable,str(compiler),str(TABLES/'main.tex'),'--compiler','texlive','--engine','xelatex','--output-directory',str(TABLES/'build'),'--json'],cwd=TABLES,capture_output=True,text=True,encoding='utf-8',errors='replace')
            (result_dir/'latex_build.log').write_text(built.stdout+'\n'+built.stderr,encoding='utf-8')
            if built.returncode:raise RuntimeError('LaTeX build failed; see latex_build.log')
            shutil.copy2(TABLES/'build/main.pdf',TABLES/'KBS_Main_Text_Tables.pdf')
    manifest_path=TABLES/'table_manifest.json'
    if manifest_path.exists():
        tm=json.loads(manifest_path.read_text(encoding='utf-8'))
        tm.update(all_experimental_values_unfilled=not any(not k.startswith('T1.') for k in values),
                  total_cells=len(cells),placeholder_cells=len(cells)-len(values),filled_cells=len(values),performance_source='local reruns only',
                  three_seed_groups=published,fill_audit=str(result_dir/'table_fill_audit.json'))
        tm['baseline_scope']=scope_summary
        for entry in tm.get('tables',[]):
            source=(TABLES/'tables'/entry['file']).read_text(encoding='utf-8')
            caption=re.search(r'\\caption\{([^}]+)\}',source)
            if caption:entry['title']=caption.group(1)
        dump(manifest_path,tm)
    validation_path=TABLES/'validation_report.json'
    validation=json.loads(validation_path.read_text(encoding='utf-8')) if validation_path.exists() else {}
    if build and (validation.get('pdf_sha256')!=digest(TABLES/'KBS_Main_Text_Tables.pdf') or validation.get('filled_cells')!=len(values)):
        subprocess.run([sys.executable,str(TABLES/'validate_current.py')],check=True)
    print(json.dumps({'filled_cells':len(values),'completed_groups':published,'rejected':failures,'document_changed':changed},ensure_ascii=False))
    return changed


def collect(build=True):
    lock=RUN/'results/collector.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('a+b') as handle:
        handle.seek(0,2)
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:return False
        else:
            import fcntl
            try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return False
        try:return collect_impl(build)
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


if __name__=='__main__':collect()
