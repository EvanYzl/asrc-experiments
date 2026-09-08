"""Freeze an isolated seed-17 completion batch for the language table."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT = Path('G:/zhishitupui')
BASE = Path(__file__).resolve().parent
RUNTIME = BASE / 'runtime'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    assert not (BASE / 'SOURCE_FREEZE.json').exists(), 'Batch already prepared'
    RUNTIME.mkdir(parents=True, exist_ok=True)
    origins = {}
    for name in ['ATransN', 'LSMGA', 'DMKGC', 'IMKGC']:
        source = ROOT / 'reproduction/sources' / name
        for path in source.rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            target = BASE / 'sources' / name / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            origins[str(path.relative_to(ROOT))] = digest(path)
    for name in ['common.py', 'run_kge.py', 'run_atransn.py']:
        path = ROOT / 'reproduction/strict_baselines' / name
        origins[str(path.relative_to(ROOT))] = digest(path)
        shutil.copy2(path, RUNTIME / name)
    for name in ['run_graph_sota.py', 'wk3l_graph_data.py']:
        path = ROOT / 'reproduction/sota' / name
        origins[str(path.relative_to(ROOT))] = digest(path)
        shutil.copy2(path, RUNTIME / name)
    shutil.copytree(ROOT / 'reproduction/strict_baselines/data_manifests', RUNTIME / 'data_manifests')

    path = RUNTIME / 'common.py'
    code = path.read_text(encoding='utf-8')
    code = replace_once(code, 'ROOT = Path(__file__).resolve().parents[2]', "ROOT = Path('G:/zhishitupui')\nBASE = Path(__file__).resolve().parent.parent")
    code = replace_once(code, "SUITE = ROOT / 'reproduction' / 'strict_baselines'", 'SUITE = Path(__file__).resolve().parent')
    code = replace_once(code, 'torch.set_num_threads(4)', "torch.set_num_threads(2)\n    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('LANG_TABLE_GPU_FRACTION', '0.20')))")
    code = replace_once(code, "objects=ROOT/'reproduction/runs/strict_baselines_20260905/code_objects'", "objects=BASE/'code_objects'")
    path.write_text(code, encoding='utf-8')

    path = RUNTIME / 'run_kge.py'
    code = path.read_text(encoding='utf-8').replace("ROOT/'reproduction/sources/ATransN/src/kge_model.py'", "BASE/'sources/ATransN/src/kge_model.py'")
    code = replace_once(code, "        final=evaluate(data,lambda b:score_all(model,b),output=out/'test_queries.npz',limit=limit)", "        atomic_json(out/'FINAL_EVALUATION_FREEZE.json', {'seed':args.seed, 'completed_steps':args.steps, 'selected_step':best_step, 'checkpoint_sha256':sha256(out/'best.pt'), 'selection':'val_select only'})\n        final=evaluate(data,lambda b:score_all(model,b),output=out/'test_queries.npz',split='val_select' if args.smoke else 'test',limit=limit)")
    path.write_text(code, encoding='utf-8')

    original_recipe = ROOT / 'reproduction/runs/strict_baselines_20260905/jobs/atransn_wk3l_s17/config.json'
    recipe = json.loads(original_recipe.read_text(encoding='utf-8'))['recipe']
    (BASE / 'atransn_recipe.json').write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')
    origins[str(original_recipe.relative_to(ROOT))] = digest(original_recipe)
    align = ROOT / 'data/raw/atransn/SHARED/wk3l-15k_en_f_fr_aligned_entity_id.txt'
    rows = [line.split() for line in align.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert all(len(row) == 2 for row in rows)
    reverse = BASE / 'fr_en_aligned_entity_id.txt'
    reverse.write_text(''.join(f'{b}\t{a}\n' for a, b in rows), encoding='utf-8')
    assert [line.split()[::-1] for line in reverse.read_text().splitlines()] == rows
    path = RUNTIME / 'run_atransn.py'
    code = path.read_text(encoding='utf-8')
    code = replace_once(code, 'from common import *', "sys.path.insert(0, 'G:/zhishitupui/reproduction/envs/pyg210_27')\nfrom common import *")
    code = code.replace("ROOT/'reproduction/sources/ATransN/src'", "BASE/'sources/ATransN/src'")
    old = "    previous=next((ROOT/'reproduction/runs/baseline_reproduction_20260904/artifacts/atransn').glob('*/config.json'))"
    code = replace_once(code, old, "    previous=BASE/'atransn_recipe.json'")
    code = replace_once(code, "teacher_data_path=str(ROOT/'data/raw/atransn/WK3l-15k_EN_F')", "teacher_data_path=str(ROOT/'data/raw/atransn/WK3l-15k_FR')")
    code = replace_once(code, "student_data_path=str(ROOT/'data/raw/atransn/WK3l-15k_FR')", "student_data_path=str(ROOT/'data/raw/atransn/WK3l-15k_EN_F')")
    code = replace_once(code, "shared_entity_path=str(ROOT/'data/raw/atransn/SHARED/wk3l-15k_en_f_fr_aligned_entity_id.txt')", "shared_entity_path=str(BASE/'fr_en_aligned_entity_id.txt')")
    code = code.replace("load_kg('wk3l','fr')", "load_kg('wk3l','en')").replace("kgout=out/'fr'", "kgout=out/'en'").replace("'per_kg':{'fr':result}", "'per_kg':{'en':result}")
    code = replace_once(code, "        result=evaluate(data,lambda b:score_all(model,b),output=kgout/'test_queries.npz',limit=64 if opts.smoke else None)", "        atomic_json(out/'FINAL_EVALUATION_FREEZE.json', {'seed':opts.seed, 'completed_steps':args.steps, 'checkpoint_sha256':sha256(out/'best.pt'), 'selection':'EN val_select only', 'direction':'FR-to-EN'})\n        result=evaluate(data,lambda b:score_all(model,b),output=kgout/'test_queries.npz',split='val_select' if opts.smoke else 'test',limit=64 if opts.smoke else None)")
    path.write_text(code, encoding='utf-8')

    path = RUNTIME / 'wk3l_graph_data.py'
    code = path.read_text(encoding='utf-8')
    code = replace_once(code, 'from frozen_data import ROOT,sha256,atomic_json', 'from common import ROOT,SUITE,sha256,atomic_json')
    code = replace_once(code, "ROOT/'reproduction/strict_baselines/graph_data'/profile/'datasetwk3l'", "SUITE/'graph_data'/profile/'datasetwk3l'")
    path.write_text(code, encoding='utf-8')
    path = RUNTIME / 'run_graph_sota.py'
    code = path.read_text(encoding='utf-8')
    code = code.replace("sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'strict_baselines'))", "# Reuse the existing local graph dependency installation.\nsys.path.insert(0, 'G:/zhishitupui/reproduction/envs/pyg210_27')")
    code = code.replace("ROOT/'reproduction/sources'/opts.method", "BASE/'sources'/opts.method").replace("ROOT/'reproduction/sources/DMKGC/src/utils.py'", "BASE/'sources/DMKGC/src/utils.py'")
    code = replace_once(code, "                if split=='test' and kgname not in DOMAINS[opts.dataset]:continue\n", '')
    code = replace_once(code, "        with torch.no_grad():\n            test=tester.evaluate_split", "        atomic_json(out/'FINAL_EVALUATION_FREEZE.json', {'seed':opts.seed, 'completed_rounds':rounds, 'selected_evaluation':state['evaluation_index'], 'checkpoint_sha256':sha256(out/'best.pt'), 'selection':'FR val_select only', 'test_targets':['fr','en']})\n        with torch.no_grad():\n            test=tester.evaluate_split")
    path.write_text(code, encoding='utf-8')
    # Extend only dataset dispatch in our copy; model, loss and optimizer stay unchanged.
    path = BASE / 'sources/DMKGC/run_model.py'
    code = path.read_text(encoding='utf-8')
    code = replace_once(code, "    else:\n        raise ValueError(f'Unsupported dataset: {args.dataset}')", "    elif args.dataset == 'wk3l':\n        all_langs = ['fr', 'en']\n    else:\n        raise ValueError(f'Unsupported dataset: {args.dataset}')")
    path.write_text(code, encoding='utf-8')
    for path in [*RUNTIME.glob('*.py'), BASE / 'sources/DMKGC/run_model.py']:
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')

    py = 'C:/Users/evan/.conda/envs/ai4/python.exe'
    jobs = []
    for method in ['TransE', 'DistMult', 'RotatE']:
        name = method.lower() + '_en_s17'
        jobs.append({'id':name, 'kind':'kge', 'method':method, 'kg':'en', 'output':str(BASE/'jobs'/name),
                     'command':[py, str(RUNTIME/'run_kge.py'), '--method',method,'--dataset','wk3l','--kg','en','--seed','17','--output',str(BASE/'jobs'/name)], 'dependencies':[]})
    name='transe_fr_teacher_s17'
    jobs.append({'id':name,'kind':'teacher','method':'TransE','kg':'fr','output':str(BASE/'jobs'/name),
                 'command':[py,str(RUNTIME/'run_kge.py'),'--method','TransE','--dataset','wk3l','--kg','fr','--teacher','--batch-size','1024','--seed','17','--output',str(BASE/'jobs'/name)],'dependencies':[]})
    name='atransn_en_s17'
    jobs.append({'id':name,'kind':'transfer','method':'ATransN','kg':'en','output':str(BASE/'jobs'/name),
                 'command':[py,str(RUNTIME/'run_atransn.py'),'--seed','17','--teacher',str(BASE/'jobs/transe_fr_teacher_s17/fr'),'--output',str(BASE/'jobs'/name)],'dependencies':['transe_fr_teacher_s17']})
    for method in ['LSMGA','DMKGC','IMKGC']:
        name=method.lower()+'_wk3l_s17'
        jobs.append({'id':name,'kind':'graph','method':method,'kg':['fr','en'],'output':str(BASE/'jobs'/name),
                     'command':[py,str(RUNTIME/'run_graph_sota.py'),'--method',method,'--dataset','wk3l','--seed','17','--output',str(BASE/'jobs'/name)],'dependencies':[]})
    manifest={'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'seed':17,'purpose':'Fill only the missing cells of the standalone language/KG table',
              'selection':'val_select only; graph checkpoints use FR selection; reverse ATransN uses EN selection; no tuning after test',
              'graph_recipe':'50 LSMGA rounds (lr .005, margin .3), 30 DMKGC/IMKGC rounds; original default/DBP recipe, no WK-specific tuning',
              'evaluation':'fixed original splits, full-candidate filtered tail ranking, ascending-ID ties; WK all filter; FR-only primary AVG',
              'resources':{'gpu':0,'max_parallel':1,'min_free_mib':5000,'gpu_memory_fraction':0.20,'cpu_threads':2,'process_priority':'below normal'},
              'existing_job_to_reuse':'reproduction/runs/strict_baselines_20260905/jobs/imkgc_depkg_s17',
              'jobs':jobs}
    (BASE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    frozen={str(p.relative_to(BASE)):digest(p) for p in BASE.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.txt']}
    receipt={'created_at':manifest['created_at'],'source_origins':origins,'files':frozen,'alignment_pairs':len(rows),'raw_alignment_sha256':digest(align),'reverse_alignment_sha256':digest(reverse),
             'changes':['isolated paths and source copies','FR-to-EN ATransN direction with reversed alignment columns','WK graph dataset dispatch and supplemental EN evaluation','2 CPU threads and 20 percent GPU memory cap; no scoring/loss changes']}
    (BASE/'SOURCE_FREEZE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (BASE/'TRAINING_LOG.md').write_text('# Language table completion, seed 17\n\n'+json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'prepared':True,'jobs':len(jobs),'alignment_pairs':len(rows),'base':str(BASE)}))


if __name__ == '__main__':
    main()
