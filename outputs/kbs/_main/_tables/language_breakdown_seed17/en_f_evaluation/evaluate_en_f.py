"""One supplemental EN_F evaluation of frozen ASRC seed17; reuse the saved English teacher test."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path('G:/zhishitupui')
OUT = Path(__file__).resolve().parent
COMMON_SHA = '2822266dfba89c642074719c88ea603e9a8795fabbd26db59afdb2e6ca00fad5'
CP = ROOT/'reproduction/sota/pilot02/wk3l_shared_r256_n3p01_s17/best.pt'
TEACHER = ROOT/'reproduction/runs/strict_baselines_20260905/jobs/atransn_teacher_s17/en'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(2**20), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def log(stage, text):
    with (OUT/'EVALUATION_LOG.md').open('a', encoding='utf-8') as f:
        f.write('\n## '+stage+' — '+dt.datetime.now(dt.timezone.utc).isoformat()+'\n'+text+'\n')


def arrays_for(kg):
    folder = ROOT/'data/raw/atransn'/('WK3l-15k_EN_F' if kg == 'en' else 'WK3l-15k_FR')
    return {name: np.loadtxt(folder/f'{name}_triple_id.txt', dtype=np.int64, ndmin=2)
            for name in ('train', 'valid', 'test')}


def metrics(rank):
    rank = np.asarray(rank, dtype=np.float64)
    return {'mrr': float(np.mean(1/rank)), 'h1': float(np.mean(rank == 1)),
            'h3': float(np.mean(rank <= 3)), 'h10': float(np.mean(rank <= 10)), 'n': len(rank)}


def prepare():
    assert not (OUT/'EVALUATION_FREEZE.json').exists(), 'Preparation already frozen'
    log('BEFORE preparation', 'Purpose: user requested available EN_F tests. Seed17 only; no training or checkpoint reselection. '
        'Keep all existing FR/main results fixed. Audit the original English split, freeze the existing ASRC checkpoint, '
        'and reuse the previously completed TransE teacher test. PID='+str(os.getpid())+'; GPU: none.')
    main = read(ROOT/'reproduction/sota/three_seed/RESULTS.json')['datasets']['wk3l']['seeds']['17']
    config = read(CP.parent/'config.json')
    assert digest(CP) == main['checkpoint_sha256']
    assert config['recipe']['seed'] == 17 and config['recipe']['sharing'] == 'shared'
    for name, sha in config['source_hashes'].items():
        assert digest(ROOT/'reproduction/sota'/name) == sha
    source = ROOT/f'reproduction/runs/strict_baselines_20260905/code_objects/{COMMON_SHA}.py'
    assert digest(source) == COMMON_SHA
    runtime = OUT/'runtime'
    runtime.mkdir(exist_ok=True)
    shutil.copyfile(source, runtime/'common_frozen.py')
    hashes = {}
    for p in [CP, CP.parent/'config.json', CP.parent/'result.json', Path(__file__), runtime/'common_frozen.py',
              ROOT/'reproduction/sota/train_complex.py', ROOT/'reproduction/sota/frozen_data.py',
              TEACHER/'result.json', TEACHER/'config.json', TEACHER/'best.pt',
              TEACHER/'test_queries.npz', TEACHER/'checkpoint_valid.pt']:
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    for kg in ['en', 'fr']:
        path = ROOT/f'reproduction/strict_baselines/data_manifests/wk3l_{kg}.json'
        manifest = read(path)
        assert digest(path) == config['manifests'][kg]
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        for name, sha in manifest['files'].items():
            p = ROOT/name.replace('\\', '/')
            assert digest(p) == sha
            hashes[p.relative_to(ROOT).as_posix()] = sha
    for name, item in config['alignment']['alignment_files'].items():
        assert digest(ROOT/name) == item['sha256']
        hashes[name] = item['sha256']
    a = arrays_for('en')
    train = set(map(tuple, a['train'].tolist()))
    valid = set(map(tuple, a['valid'].tolist()))
    overlap_train = np.asarray([tuple(x) in train for x in a['test']])
    overlap_valid = np.asarray([tuple(x) in valid for x in a['test']])
    keep = ~(overlap_train | overlap_valid)
    np.savez_compressed(OUT/'overlap_masks.npz', train_overlap=overlap_train,
                        valid_overlap=overlap_valid, diagnostic_keep=keep,
                        test_query_index=np.arange(len(a['test'])))
    hashes[(OUT/'overlap_masks.npz').relative_to(ROOT).as_posix()] = digest(OUT/'overlap_masks.npz')
    teacher = read(TEACHER/'result.json')
    teacher_config = read(TEACHER/'config.json')
    assert teacher['seed'] == 17 and teacher['kg'] == 'en' and teacher['status'] == 'completed'
    assert teacher_config['teacher'] and teacher_config['method'] == 'TransE'
    assert digest(TEACHER/'best.pt') == teacher['checkpoint_sha256']
    atr = read(ROOT/'reproduction/runs/strict_baselines_20260905/jobs/atransn_wk3l_s17/config.json')
    assert digest(TEACHER/'checkpoint_valid.pt') == atr['teacher_checkpoint_sha256']
    with np.load(TEACHER/'test_queries.npz', allow_pickle=False) as z:
        assert np.array_equal(z['triples'], a['test'])
        assert np.array_equal(z['query_index'], np.arange(len(a['test'])))
        for filtering in ('train', 'train_valid', 'all'):
            derived = metrics(z['rank_'+filtering])
            for metric, value in derived.items():
                assert abs(value-teacher['test']['metrics'][filtering][metric]) < 1e-12
        teacher_clean = metrics(z['rank_all'][keep])
    audit = {'test_queries': len(a['test']), 'unique_test_triples': len(set(map(tuple, a['test'].tolist()))),
             'training_overlap_queries': int(overlap_train.sum()), 'validation_overlap_queries': int(overlap_valid.sum()),
             'excluded_diagnostic_queries': int((~keep).sum()), 'diagnostic_queries': int(keep.sum()),
             'primary_split_unchanged': True, 'diagnostic_filter_sets_unchanged': True,
             'interpretation': 'Original EN_F split contains overlapping facts; primary results must not be described as overlap-free.'}
    write(OUT/'INPUT_AUDIT.json', audit)
    teacher_output = {'method': 'TransE (EN teacher)', 'seed': 17, 'kg': 'en', 'status': 'completed',
                      'action': 'reuse_saved_test', 'source_result': str((TEACHER/'result.json').relative_to(ROOT)),
                      'source_rank': str((TEACHER/'test_queries.npz').relative_to(ROOT)),
                      'checkpoint_sha256': teacher['checkpoint_sha256'], 'test': teacher['test'],
                      'diagnostic_without_train_valid_overlaps': teacher_clean,
                      'recipe': {'margin': teacher_config['margin'], 'batch_size': teacher_config['batch_size'],
                                 'embedding_dim': teacher_config['embedding_dim']},
                      'role': 'Existing ATransN source teacher, not an ATransN English-target result or the ordinary TransE recipe.'}
    write(OUT/'teacher_reused.json', teacher_output)
    freeze = {'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(), 'seed': 17,
              'purpose': 'Supplemental EN_F evaluation; all original FR results and FR-only AVG remain fixed',
              'asrc_checkpoint': CP.relative_to(ROOT).as_posix(), 'asrc_checkpoint_sha256': digest(CP),
              'asrc_recipe': config['recipe'], 'checkpoint_selection': 'Previously frozen best epoch35, selected on FR val_select only; no English reselection',
              'gpu': 0, 'cpu_threads': 2, 'batch_size': 64, 'dtype': 'float32', 'tf32': False,
              'gpu_memory_fraction_cap': 0.125, 'minimum_free_gpu_mib': 2500,
              'target': 'WK3l-15k_EN_F', 'candidate_entities': 15169, 'test_queries': 40700,
              'filtering': 'all (train+valid+test positives)', 'ties': 'ascending candidate entity ID',
              'primary': 'original fixed split, overlap counts disclosed',
              'secondary': 'same saved ranks on queries absent from both train and valid; unchanged filtering sets',
              'new_model_tests': ['asrc_en_f_s17'], 'reused_tests': ['atransn_teacher_s17/en'],
              'no_training': True, 'no_test_based_selection': True, 'input_audit': audit, 'hashes': hashes}
    write(OUT/'EVALUATION_FREEZE.json', freeze)
    log('AFTER preparation', 'Checkpoint/input/source hashes frozen; teacher raw ranks verified and reused. '
        +json.dumps(audit)+'\nNext: one ASRC EN_F evaluation using the existing frozen scoring and ranking implementation. '
        'Command: evaluate_en_f.py run; planned GPU0, batch64, two CPU threads, FP32; GPU allocation limited to one eighth of local GPU memory.')
    print(json.dumps({'prepared': True, 'audit': audit, 'teacher_reused': teacher['test']['metrics']['all']}), flush=True)


def run():
    import torch
    frozen = read(OUT/'EVALUATION_FREEZE.json')
    for name, sha in frozen['hashes'].items():
        assert digest(ROOT/name) == sha, ('Frozen artifact changed', name)
    assert not (OUT/'TEST_OPENED.json').exists(), 'English evaluation already started; inspect existing output before any recovery'
    free_mib = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip().splitlines()[0])
    assert free_mib >= frozen['minimum_free_gpu_mib'], ('Insufficient free GPU memory', free_mib)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(frozen['gpu_memory_fraction_cap'], device=0)
    torch.manual_seed(17)
    sys.path.insert(0, str(ROOT/'reproduction/sota'))
    from frozen_data import load_multikg
    from train_complex import Complex
    spec = importlib.util.spec_from_file_location('en_f_frozen_common', OUT/'runtime/common_frozen.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    # The archived module is read unchanged; only its path root is restored in this process.
    common.ROOT = ROOT
    common.SUITE = ROOT/'reproduction/strict_baselines'
    saved = torch.load(CP, map_location='cpu', weights_only=False)
    recipe = saved['config']['recipe']
    assert recipe == frozen['asrc_recipe'] and saved['best_epoch'] == 35
    data, maps, roff, n, nr, alignment = load_multikg('wk3l', recipe['sharing'])
    assert alignment == saved['config']['alignment']
    model = Complex(n, nr, recipe['rank'])
    model.load_state_dict(saved['model'])
    model = model.cuda().eval()
    del saved
    mapping = torch.as_tensor(maps['en'], device='cuda')
    d = common.load_kg('wk3l', 'en')
    assert d['entities'] == frozen['candidate_entities'] and len(d['arrays']['test']) == frozen['test_queries']
    out = OUT/'asrc_en_f_s17'
    out.mkdir(exist_ok=True)
    write(out/'config.json', frozen)
    write(OUT/'TEST_OPENED.json', {'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(), 'pid': os.getpid(),
                                 'gpu': 0, 'checkpoint_sha256': frozen['asrc_checkpoint_sha256'],
                                 'freeze_sha256': digest(OUT/'EVALUATION_FREEZE.json')})
    log('BEFORE ASRC EN_F test', 'Purpose: supplemental support-graph evaluation requested by user. Seed17, frozen epoch35. '
        'Command: evaluate_en_f.py run; PID='+str(os.getpid())+'; GPU0, free '+str(free_mib)+' MiB, batch64, CPU threads2. '
        'Output: asrc_en_f_s17/test_queries.npz and result.json. No training or checkpoint reselection.')
    t0 = time.monotonic()
    processed = 0
    def score(batch):
        nonlocal processed
        b = torch.as_tensor(batch.copy(), device='cuda')
        scores = model.scores(mapping[b[:,0]], b[:,1]+roff['en'], mapping)
        processed += len(batch)
        if processed % 8192 == 0 or processed == frozen['test_queries']:
            print(json.dumps({'evaluated_queries': processed, 'total': frozen['test_queries'], 'seconds': round(time.monotonic()-t0, 2)}), flush=True)
        return scores
    test = common.evaluate(d, score, split='test', batch_size=frozen['batch_size'], output=out/'test_queries.npz')
    with np.load(OUT/'overlap_masks.npz', allow_pickle=False) as masks:
        keep = masks['diagnostic_keep']
    with np.load(out/'test_queries.npz', allow_pickle=False) as z:
        assert np.array_equal(z['triples'], d['arrays']['test'])
        assert np.array_equal(z['query_index'], np.arange(frozen['test_queries']))
        assert np.isfinite(z['gold_score']).all()
        for filtering in ('train', 'train_valid', 'all'):
            ranks = z['rank_'+filtering]
            assert np.all(ranks >= 1) and np.all(ranks <= d['entities'])
            for metric, value in metrics(ranks).items():
                assert abs(value-test['metrics'][filtering][metric]) < 1e-12
        clean = metrics(z['rank_all'][keep])
    result = {'status': 'completed', 'purpose': 'supplemental_en_f_test', 'method': 'ASRC', 'dataset': 'wk3l',
              'kg': 'en', 'seed': 17, 'test': test, 'per_kg': {'en': test},
              'diagnostic_without_train_valid_overlaps': clean, 'input_audit': frozen['input_audit'],
              'checkpoint_sha256': frozen['asrc_checkpoint_sha256'], 'checkpoint': frozen['asrc_checkpoint'],
              'selection': frozen['checkpoint_selection'], 'protocol': 'kbs-baselines-v1-20260905',
              'scope': 'additional EN_F reporting; original FR test and AVG unchanged',
              'test_used_for_selection': False, 'wall_seconds': time.monotonic()-t0,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'raw_rank_sha256': digest(out/'test_queries.npz'), 'evaluation_freeze_sha256': digest(OUT/'EVALUATION_FREEZE.json'),
              'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(0)}}
    assert digest(CP) == frozen['asrc_checkpoint_sha256']
    write(out/'result.json', result)
    combined = {'status': 'completed', 'seed': 17, 'primary_filter': 'all', 'original_fr_unchanged': True,
                'original_avg_remains_fr_only': True, 'asrc': result, 'teacher': read(OUT/'teacher_reused.json')}
    write(OUT/'RESULTS.json', combined)
    log('AFTER ASRC EN_F test', 'Result: '+json.dumps(test['metrics']['all'])+'; overlap-excluded diagnostic: '+json.dumps(clean)+
        '. Saved all 40,700 ranks; independent numerical verification passed. Elapsed '+str(round(result['wall_seconds'],2))+
        ' seconds; peak allocated GPU bytes '+str(result['peak_cuda_allocated_bytes'])+'. Next: update only the standalone language table, '
        'label English as supplemental and the reused TransE as a teacher, compile and verify. No further model tests.')
    print(json.dumps({'completed': True, 'primary': test['metrics']['all'], 'diagnostic': clean,
                      'seconds': result['wall_seconds'], 'peak_gpu_bytes': result['peak_cuda_allocated_bytes']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        run()
