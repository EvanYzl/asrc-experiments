"""Freeze the user-requested graph-baseline repeats without touching seed 17."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / 'reproduction/sota/graph_three_seed'
OLD = ROOT / 'reproduction/runs/strict_baselines_20260905'
METHODS = ('LSMGA', 'DMKGC', 'IMKGC')
DATASETS = ('dbp5l', 'depkg', 'dwy')
REMOTE = '/root/zhishitupui'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    assert not (PHASE / 'PLAN.json').exists(), 'Existing repeat campaign must be resumed, not overwritten'
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    archive = PHASE / 'before'
    for relative in ('reproduction/sota/TASK_SCOPE.json', 'reproduction/sota/PLAN.md',
                     'outputs/kbs/_main/_tables/cells_results.csv', 'outputs/kbs/_main/_tables/values.tex',
                     'outputs/kbs/_main/_tables/tables/t02.tex', 'outputs/kbs/_main/_tables/main.tex',
                     'outputs/kbs/_main/_tables/KBS_Main_Text_Tables.pdf',
                     'outputs/kbs/_main/_tables/table_manifest.json'):
        dest = archive / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    runtime = PHASE / 'runtime'
    source_paths = [ROOT / 'reproduction/strict_baselines/run_graph_baseline.py',
                    ROOT / 'reproduction/strict_baselines/common.py']
    for method in METHODS:
        source_paths += sorted((ROOT / f'reproduction/sources/{method}').glob('src/*.py'))
        source_paths += [ROOT / f'reproduction/sources/{method}/run_model.py']
    source_paths = sorted(set(source_paths))
    sources, adaptations = {}, []
    for original in source_paths:
        rel = original.relative_to(ROOT)
        dest = runtime / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, dest)
        if rel.as_posix() == 'reproduction/strict_baselines/common.py':
            shutil.copy2(ROOT / 'reproduction/sota/runtime/server_common.py', dest)
            adaptations.append({'path': rel.as_posix(), 'reason': 'Reuse accepted Windows manifest-key portability patch only'})
        elif original.name == 'run_graph_baseline.py':
            content = original.read_text(encoding='utf-8')
            old = "    signature={'input_facts':'train-only','k':10,'num_hops':2,'dataset':dataset,"
            new = "    cache_signature = dest/'strict_graph_manifest.json'\n    cache_keys = {}\n    if cache_signature.exists():\n        cache_keys = {k.replace('\\\\', '/'): k for k in json.loads(cache_signature.read_text())['source_files']}\n" + old
            content = content.replace(old, new)
            content = content.replace("str(p.relative_to(source)):sha256(p)", "cache_keys.get(p.relative_to(source).as_posix(), p.relative_to(source).as_posix()):sha256(p)")
            dest.write_text(content, encoding='utf-8', newline='\n')
            adaptations.append({'path': rel.as_posix(), 'reason': 'Preserve Windows frozen graph-manifest path keys on Linux; graph bytes and all numerical code unchanged'})
        elif original.name == 'run_model.py':
            content = original.read_text(encoding='utf-8')
            patched = content.replace("os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'", "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')")
            patched = patched.replace("os.environ['CUDA_VISIBLE_DEVICES'] = '0'", "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')")
            if content != patched:
                dest.write_text(patched, encoding='utf-8', newline='\n')
                adaptations.append({'path': rel.as_posix(), 'reason': 'Respect scheduler CUDA_VISIBLE_DEVICES; numerical implementation unchanged'})
        sources[rel.as_posix()] = {'seed17_sha256': sha(original), 'server_sha256': sha(dest)}
        for path in (original, dest):
            obj = PHASE / 'code_objects' / (sha(path) + '.py')
            obj.parent.mkdir(parents=True, exist_ok=True)
            if not obj.exists():
                shutil.copy2(path, obj)
    seed17 = []
    for method in METHODS:
        for ds in DATASETS:
            jid = f'{method.lower()}_{ds}_s17'
            src = OLD / 'jobs' / jid
            record = {'id': jid, 'path': src.relative_to(ROOT).as_posix(),
                      'owner': 'existing local seed17 queue', 'status': 'pending_local'}
            if (src / 'config.json').exists():
                cfg = json.loads((src / 'config.json').read_text(encoding='utf-8'))
                for name, digest in cfg['source_hashes'].items():
                    assert sha(ROOT / name.replace('\\', '/')) == digest, (jid, name)
                dest = PHASE / 'seed17_reference' / jid
                dest.mkdir(parents=True, exist_ok=True)
                for name in ('config.json', 'result.json', 'learning_curve.jsonl', 'resources.json'):
                    if (src / name).exists():
                        shutil.copy2(src / name, dest / name)
                record['config_sha256'] = sha(src / 'config.json')
                record['status'] = 'completed_local' if (src / 'result.json').exists() else 'running_local'
            seed17.append(record)
    files = {}
    for ds in DATASETS:
        for path in sorted((ROOT / f'data/raw/dmkgc/dataset{ds}').rglob('*')):
            if path.is_file() and path.suffix in ('.tsv', '.txt'):
                files[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in sorted((ROOT / 'reproduction/strict_baselines/data_manifests').glob('*.json')):
        if path.stem.split('_')[0] in DATASETS:
            files[path.relative_to(ROOT).as_posix()] = sha(path)
    cache = ROOT / 'reproduction/strict_baselines/graph_data/train_only_k10_h2'
    cache_files = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(cache.rglob('*')) if p.is_file()}
    assert len([p for p in cache_files if p.endswith('.graph')]) == 14
    write(PHASE / 'SOURCE_INPUT_FREEZE.json', {'created_at': stamp, 'sources': sources,
          'adaptations': adaptations, 'inputs': files, 'frozen_graph_cache': cache_files,
          'note': 'Read-only exact local seed17 cache reused; isolated server runtime links to fixed public data/manifests.'})
    jobs = []
    order = [('IMKGC', 'dbp5l'), ('IMKGC', 'depkg'), ('IMKGC', 'dwy'),
             ('DMKGC', 'depkg'), ('LSMGA', 'depkg'), ('LSMGA', 'dbp5l'),
             ('LSMGA', 'dwy'), ('DMKGC', 'dbp5l'), ('DMKGC', 'dwy')]
    for method, ds in order:
        for seed in (29, 43):
            jid = f'{method.lower()}_{ds}_s{seed}'
            output = 'reproduction/sota/graph_three_seed/jobs/' + jid
            command = [REMOTE + '/.envs/kgc/bin/python',
                       REMOTE + '/reproduction/sota/graph_three_seed/runtime/reproduction/strict_baselines/run_graph_baseline.py',
                       '--method', method, '--dataset', ds, '--seed', str(seed), '--output', REMOTE + '/' + output]
            jobs.append({'id': jid, 'method': method, 'dataset': ds, 'seed': seed,
                         'purpose': 'Complete the two missing fixed graph-baseline seeds for Table2',
                         'changes': 'Seed and output only; same seed17 recipe, val_select checkpoint selection, frozen full-candidate evaluation',
                         'output': output, 'command': command, 'min_free_mib': 9000,
                         'next': 'Finish only the 18 registered repeats; preserve every outcome; aggregate seeds17/29/43 after local seed17 completes'})
    write(PHASE / 'manifest.json', {'schema': 1, 'created_at': stamp, 'max_parallel': 6, 'jobs': jobs})
    write(PHASE / 'PLAN.json', {'created_at': stamp, 'authorization': 'User explicitly requested two remaining seeds for LSMGA/DMKGC/IMKGC; local IMKGC seed17 remains owned by local machine',
          'methods': METHODS, 'datasets': DATASETS, 'new_seeds': [29, 43], 'aggregate_seeds': [17, 29, 43],
          'new_training_jobs': 18, 'seed17_reuse': seed17, 'max_parallel': 6,
          'protocol': 'kbs-baselines-v1-20260905', 'selection': 'earliest maximum val_select equal-KG macro filtered tail MRR',
          'aggregation': 'exact per-KG full-candidate train+valid-filtered test ranks; equal KG macro per seed; seed mean and sample SD ddof1',
          'freeze': 'SOURCE_INPUT_FREEZE.json', 'change_policy': 'No hyperparameter search; no selecting seeds by test outcomes',
          'excluded': ['WK3l graph methods without an existing seed17 result', 'additional ASRC training or testing', 'new supporting-table experiments'],
          'stop': 'After 18 repeats, raw-result audit, 3-seed Table2 update, LaTeX compile, returned artifacts and hash verification; report remaining unknown baselines honestly'})
    write(ROOT / 'reproduction/sota/TASK_SCOPE.json', {'phase': 'graph_three_seed', 'status': 'preflight',
          'new_authorization': stamp, 'methods': METHODS, 'datasets': DATASETS, 'new_seeds': [29, 43],
          'local_seed17_queue_untouched': True, 'new_training': 18, 'asrc_locked': True,
          'further_experiments_authorized': 'Only the 18 explicitly registered baseline repeats'})
    (PHASE / 'PLAN.zh.md').write_text('# 三种图基线补齐三种子\n\n'
          'LSMGA、DMKGC、IMKGC 在 DBP-5L、E-PKG、DWY 各补 seed 29 和 43，共 18 次正式运行。'
          '本地 seed 17 由现有队列继续执行并复用，不启动重复任务。\n\n'
          '配置、完整输入、训练图缓存、数据划分和评测规则在 SOURCE_INPUT_FREEZE.json 中冻结。'
          '仅依据 val_select 选择检查点，训练预算与 seed 17 相同，最终测试结果全部保留。'
          '按每个种子各 KG 等权宏平均，再计算三个种子的均值和样本标准差。\n\n'
          '服务器每卡一个独立进程，最多六路。完成后核验原始排名，更新当前 Table 2 并编译 PDF。'
          'WK3l 图基线缺失仍如实标注；ASRC 与支持实验保持冻结。\n', encoding='utf-8')
    package = ROOT / 'deployment/graph_three_seed/runtime_cache.tar.gz'
    package.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package, 'w:gz', compresslevel=3) as tar:
        for path in sorted(runtime.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(runtime).as_posix())
        for name in cache_files:
            tar.add(ROOT / name, arcname=name)
    write(PHASE / 'DEPLOYMENT.json', {'package': package.relative_to(ROOT).as_posix(),
                                    'sha256': sha(package), 'bytes': package.stat().st_size})
    print(json.dumps({'jobs': len(jobs), 'seed17': seed17, 'runtime_sources': len(sources),
                      'input_files': len(files), 'cache_files': len(cache_files),
                      'package_mib': package.stat().st_size / 2**20}))


if __name__ == '__main__':
    main()
