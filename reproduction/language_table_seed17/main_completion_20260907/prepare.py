"""Freeze the exact runs required by the user's additional main-PDF completion request."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import shutil
import tarfile

ROOT=Path('G:/zhishitupui');SIDE=ROOT/'reproduction/language_table_seed17'
BASE=Path(__file__).resolve().parent;STAGE=BASE/'bundle'
REMOTE='/root/zhishitupui/reproduction/main_tables_completion_20260907'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def edit(src,dst,replacements):
    text=src.read_text(encoding='utf-8')
    for old,new in replacements:assert old in text,(src,old);text=text.replace(old,new)
    dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(text,encoding='utf-8')
    return {'source':src.relative_to(ROOT).as_posix(),'source_sha256':sha(src),'deployed':dst.relative_to(STAGE).as_posix(),'deployed_sha256':sha(dst),'changes':replacements}

def main():
    assert not STAGE.exists();STAGE.mkdir()
    paper=ROOT/'outputs/kbs/_main/_tables'
    backup=BASE/'before_main_pdf_update';backup.mkdir()
    for p in paper.iterdir():
        if p.is_file():copy(p,backup/p.name)
    shutil.copytree(paper/'tables',backup/'tables')
    with (paper/'cells_results.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    missing=[r['cell_id'] for r in rows if not r['value']];assert len(missing)==20
    changes=[]
    canonical=("files = {str(p.relative_to(ROOT)): sha256(p)","files = {str(p.relative_to(ROOT)).replace('/', chr(92)): sha256(p)")
    # Existing graph runtime and sources are immutable while four repeats run.
    # This additive bundle never replaces them; the queue adopts their states.
    changes.append(edit(ROOT/'reproduction/strict_baselines/common.py',STAGE/'runtime_ssaga/common.py',[
        ("ROOT = Path(__file__).resolve().parents[2]","ROOT = Path('/root/zhishitupui')\nBASE = Path(__file__).resolve().parent.parent"),
        ("SUITE = ROOT / 'reproduction' / 'strict_baselines'","SUITE = Path(__file__).resolve().parent"),canonical,
        ("objects=ROOT/'reproduction/runs/strict_baselines_20260905/code_objects'","objects=BASE/'code_objects'")]))
    changes.append(edit(ROOT/'reproduction/strict_baselines/run_ssaga.py',STAGE/'runtime_ssaga/run_ssaga.py',[
        ("source=ROOT/'reproduction/sources/SS-AGA'","source=BASE/'sources/SS-AGA'"),
        ("ROOT/'reproduction/tools/build_ssaga_mbert_embeddings.py'","BASE/'tools/build_ssaga_mbert_embeddings.py'"),
        ("    atomic_json(dest/'feature_manifest.json',provenance)","    if (dest/'feature_manifest.json').exists():\n        assert json.loads((dest/'feature_manifest.json').read_text(encoding='utf-8'))==provenance\n    else:\n        atomic_json(dest/'feature_manifest.json',provenance)"),
        ("        result=evaluate(data,tester.score,output=out/'test_queries.npz',limit=64 if opts.smoke else None)",
         "        atomic_json(out/'FINAL_EVALUATION_FREEZE.json',{'seed':opts.seed,'completed_rounds':args.round,'selected_round':state['selected_round'],'checkpoint_sha256':sha256(out/'best.pt'),'selection':'val_select only'})\n        result=evaluate(data,tester.score,output=out/'test_queries.npz',limit=64 if opts.smoke else None)")]))
    copy(ROOT/'reproduction/tools/build_ssaga_mbert_embeddings.py',STAGE/'tools/build_ssaga_mbert_embeddings.py')
    for method in ['SS-AGA']:
        source=ROOT/'reproduction/sources/SS-AGA'
        for p in source.rglob('*.py'):copy(p,STAGE/'sources'/method/p.relative_to(source))
    for runtime in ['runtime_ssaga']:
        for p in (ROOT/'reproduction/strict_baselines/data_manifests').glob('*.json'):copy(p,STAGE/runtime/'data_manifests'/p.name)
    features={}
    for ds in ['dbp5l','depkg']:
        source=ROOT/f'reproduction/strict_baselines/ssaga_data/dataset{ds}'
        for p in source.rglob('*'):
            if p.is_file():copy(p,STAGE/f'runtime_ssaga/ssaga_data/dataset{ds}'/p.relative_to(source))
        features[ds]={name:sha(source/name) for name in ['entity_embeddings.npy','entity_embeddings.provenance.json']}
    inputs={}
    for folder in [ROOT/'data/raw/dmkgc/datasetdbp5l',ROOT/'data/raw/dmkgc/datasetdepkg',ROOT/'data/raw/atransn/WK3l-15k_EN_F',ROOT/'data/raw/atransn/WK3l-15k_FR',ROOT/'data/raw/atransn/SHARED']:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ['.tsv','.txt']:inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    jobs=[]
    for method in ['LSMGA','DMKGC','IMKGC']:
        for seed in [29,43]:
            name=f'{method.lower()}_wk3l_s{seed}'
            jobs.append({'id':name,'kind':'graph','method':method,'dataset':'wk3l','seed':seed,'output':'jobs/'+name,
                'command':['python',REMOTE+'/runtime_graph/run_graph_sota.py','--method',method,'--dataset','wk3l','--seed',str(seed),'--output',REMOTE+'/jobs/'+name]})
    # Each SS-AGA target is already an independent process in the frozen driver.
    # Schedule those targets across available GPUs, preserving per-target seed and recipe.
    for ds,kgs in [('dbp5l',['el','en','es','fr','ja']),('depkg',['de','es','fr','it','jp','uk'])]:
        for seed in [17,29,43]:
            for kg in kgs:
                name=f'ssaga_{ds}_{kg}_s{seed}';output=f'jobs/ssaga_{ds}_s{seed}/{kg}'
                jobs.append({'id':name,'kind':'ssaga','method':'SS-AGA','dataset':ds,'kg':kg,'seed':seed,'output':output,
                    'command':['python',REMOTE+'/runtime_ssaga/run_ssaga.py','--dataset',ds,'--seed',str(seed),'--target',kg,'--output',REMOTE+'/'+output]})
    assert len(jobs)==39
    plan={'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'authorization':'User on 2026-09-07 explicitly requested that KBS_Main_Text_Tables.pdf also be fully filled at final delivery',
          'main_pdf':str(paper/'KBS_Main_Text_Tables.pdf'),'main_tex':str(paper/'main.tex'),'missing_cells':missing,'backup':str(backup),
          'jobs':jobs,'source_changes':changes,'input_sha256':inputs,'frozen_feature_sha256':features,
          'reuse':{'wk3l_seed17':'Existing completed LSMGA/DMKGC/IMKGC single-seed jobs','epkg_imkgc_seed17':'Ongoing full-state server continuation','epkg_imkgc_seed29_43':'Already returned completed graph_three_seed runs'},
          'repetition':'seed17/29/43 mean and sample SD; per-KG equal macro first',
          'selection':'Validation only; fixed seed/config/budget, final test once; preserve every result including unfavorable outcomes',
          'ssaga_protocol':'Frozen regenerated mBERT CLS features, supporter train+validation facts, full candidates and train-filter reporting; remains Table3 method-specific',
          'resources':{'max_parallel':6,'min_free_mib':7000,'one_job_per_gpu':True,'cpu_threads_graph':2,'cpu_threads_ssaga':4,'poll_seconds':15},
          'stop':'After 39 registered target/repeat jobs complete and both PDFs plus raw artifacts are returned and verified, stop; no new candidates'}
    for name in ['preflight.py','scheduler.py']:
        copy(BASE/name,STAGE/name)
    (STAGE/'PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (BASE/'PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files={p.relative_to(STAGE).as_posix():sha(p) for p in STAGE.rglob('*') if p.is_file()}
    (STAGE/'BUNDLE_HASHES.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8')
    archive=BASE/'bundle.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in sorted(STAGE.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(STAGE).as_posix(),recursive=False)
    (BASE/'bundle.sha256').write_text(sha(archive)+'\n',encoding='utf-8')
    print(json.dumps({'jobs':len(jobs),'new_three_seed_groups':5,'missing_cells':len(missing),'archive_bytes':archive.stat().st_size,'sha256':sha(archive)}))

if __name__=='__main__':main()
