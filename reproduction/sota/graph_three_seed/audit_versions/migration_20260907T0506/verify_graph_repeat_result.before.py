"""Audit saved ranks/checkpoints for the registered graph seeds; never run inference or fill tables.
Adapted from strict_baselines/verify_single_graph_result.py; publication is a separate step.
"""
import argparse
import csv
import datetime
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / 'reproduction/strict_baselines'
OLD_RUN = ROOT / 'reproduction/runs/strict_baselines_20260905'
PHASE = ROOT / 'reproduction/sota/graph_three_seed'
torch.set_num_threads(2)
KGS = {'dbp5l': ['el', 'en', 'es', 'fr', 'ja'],
       'depkg': ['de', 'es', 'fr', 'it', 'jp', 'uk'], 'dwy': ['db', 'wk', 'yg']}
TABLE_DATASET = {'dbp5l': 'dbp', 'depkg': 'epkg', 'dwy': 'dwy'}
METRICS = ('mrr', 'h1', 'h3', 'h10')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(rank):
    rank = rank.astype(np.float64)
    return {'mrr': float(np.mean(1 / rank)),
            **{f'h{k}': float(np.mean(rank <= k)) for k in (1, 3, 10)}}


def verify(run_id):
    match = re.fullmatch(r'(lsmga|dmkgc|imkgc)_(dbp5l|depkg|dwy)_s(17|29|43)', run_id)
    assert match, run_id
    method_key, dataset, seed_text = match.groups()
    seed = int(seed_text)
    run = OLD_RUN if seed == 17 else PHASE
    manifest = read(run / 'manifest.json')
    assert run_id in {j['id'] for j in manifest['jobs']}
    job = run / 'jobs' / run_id
    result, config = read(job / 'result.json'), read(job / 'config.json')
    method = method_key.upper()
    assert result['method'] == config['method'] == method
    assert result['dataset'] == config['dataset'] == dataset
    assert result['status'] == 'completed' and result['purpose'] == 'formal' and result['full_data']
    assert result['seed'] == config['seed'] == seed and config['input_facts'] == 'train-only'
    assert result['protocol'] == 'kbs-baselines-v1-20260905'
    assert set(result['per_kg']) == set(result['selection']) == set(config['domain_order']) == set(KGS[dataset])
    state = read(run / ('queue_state.json' if seed == 17 else 'state.json'))['jobs'][run_id]
    assert state['status'] == 'completed'
    if seed == 17:
        try:
            process = psutil.Process(state['pid'])
            assert abs(process.create_time() - state['process_created_at']) > 1, 'Completed run still owns a live process'
        except psutil.NoSuchProcess:
            pass
    else:
        assert state['returncode'] == 0 and state['finished'] >= state['started']
        if str(ROOT) == '/root/zhishitupui':
            try:
                process = psutil.Process(state['pid'])
                assert abs(process.create_time() - state['started']) > 2, 'Completed server worker remains alive'
            except psutil.NoSuchProcess:
                pass
    omit = {'seed', 'data_path', 'resume_checkpoint', 'v', 'device'}
    normalized = lambda x: {k: v for k, v in x.items() if k not in omit}
    reference = next(r for r in read(PHASE / 'PREFLIGHT.json')['recipes'] if (r['method'], r['dataset']) == (method, dataset))
    assert normalized(config['parsed_arguments_before_author_main']) == normalized(reference['arguments'])
    freeze = read(PHASE / 'SOURCE_INPUT_FREEZE.json')
    last_validation_metadata = None
    if method == 'LSMGA':
        assert config['loss_contract'] == 'released PyTorch 1.10 [B,1] versus [B] broadcasting; full training batches'
        args = config['parsed_arguments_before_author_main']
        assert args['batch_size'] == args['micro_batch_size'] == 200

    history = [json.loads(line) for line in (job / 'learning_curve.jsonl').read_text().splitlines()]
    assert [item['evaluation_index'] for item in history] == list(range(1, config['rounds'] + 1))
    for item in history:
        assert all(np.isfinite(v) and 0 <= v <= 1 for v in item['val_select_macro'].values())
    chosen = max(history, key=lambda item: item['val_select_macro']['mrr'])
    assert result['best_evaluation'] == chosen['evaluation_index']
    assert sha(job / 'best.pt') == result['checkpoint_sha256']
    best = torch.load(job / 'best.pt', map_location='cpu', weights_only=False)
    last = torch.load(job / 'last.pt', map_location='cpu', weights_only=False)
    assert best['evaluation_index'] == chosen['evaluation_index']
    assert last['completed_round'] == config['rounds'] - 1
    assert best['validation_mrr'] == chosen['val_select_macro']['mrr']
    if method == 'LSMGA':
        assert last['format'] == 'lsmga_full_state_v1'
        assert set(last['all_langs']) == set(KGS[dataset])
    else:
        assert last['format'] == 'multidomain_full_state_v1'
        author_best_index = int(last['best_epoch']) + 1
        assert 1 <= author_best_index <= len(history)
        author_row = history[author_best_index - 1]
        assert set(last['best_result']) == set(KGS[dataset])
        for kg in KGS[dataset]:
            saved_metrics = last['best_result'][kg]
            expected_metrics = author_row['per_kg'][kg]['metrics']['select']
            assert len(saved_metrics) == 3
            assert all(float(saved_metrics[i]) == expected_metrics[k]
                       for i, k in enumerate(('h1', 'h10', 'mrr')))
        # Author code averages the same float64 KG metrics in training-domain
        # order. Only this redundant scalar may differ by summation roundoff;
        # the strict best checkpoint and all raw-rank metrics are checked exactly.
        roundoff_bound = len(KGS[dataset]) * np.finfo(np.float64).eps
        author_best_mrr = float(last['best_mrr'])
        assert np.isfinite(author_best_mrr)
        assert abs(author_best_mrr - author_row['val_select_macro']['mrr']) <= roundoff_bound
        assert abs(author_best_mrr - best['validation_mrr']) <= roundoff_bound
        last_validation_metadata = {
            'author_best_evaluation': author_best_index,
            'strict_best_evaluation': chosen['evaluation_index'],
            'author_best_mrr': author_best_mrr,
            'strict_best_mrr': best['validation_mrr'],
            'absolute_difference': abs(author_best_mrr - best['validation_mrr']),
            'float64_summation_roundoff_bound': float(roundoff_bound),
            'per_kg_metrics_match_recorded_validation_exactly': True}
    assert best['config'] == config
    assert all(not torch.is_tensor(v) or torch.isfinite(v).all().item() for v in best['model'].values())
    for name, checksum in config['source_hashes'].items():
        assert sha(run / 'code_objects' / (checksum + Path(name).suffix)) == checksum
        expected = freeze['sources'][name.replace(chr(92), '/')]['seed17_sha256' if seed == 17 else 'server_sha256']
        assert checksum == expected
    adaptation = config.get('dwy_adaptation')
    if method == 'IMKGC' and dataset == 'dwy':
        assert adaptation['status'] == 'passed'
        for name, checksum in adaptation['input_sha256'].items():
            assert sha(ROOT / 'data/raw/dmkgc/datasetdwy' / name) == checksum

    per_kg, selection, counts, raw, candidates = {}, {}, {}, [], []
    for kg in KGS[dataset]:
        frozen_path = SUITE / 'data_manifests' / f'{dataset}_{kg}.json'
        frozen = read(frozen_path)
        if adaptation:
            assert sha(frozen_path) == adaptation['domains'][kg]['split_manifest_sha256']
        for name, checksum in frozen['files'].items():
            assert sha(ROOT / name.replace(chr(92), '/')) == checksum
        ne = frozen['entities']
        data = ROOT / f'data/raw/dmkgc/dataset{dataset}/kg'
        expected = {'test': np.loadtxt(data / f'{kg}-test.tsv', dtype=np.int64, delimiter='\t', ndmin=2),
                    'val_select': np.loadtxt(data / f'{kg}-val.tsv', dtype=np.int64, delimiter='\t', ndmin=2)[frozen['val_select_indices']]}
        for split in ('test', 'val_select'):
            path = job / kg / f'{split}_queries.npz'
            metadata = read(path.with_suffix('.json'))
            with np.load(path) as query:
                n = len(expected[split])
                assert n == frozen['counts'][split]
                assert np.array_equal(query['triples'], expected[split])
                assert np.array_equal(query['query_index'], np.arange(n))
                assert query['gold_score'].dtype == np.float32 and np.isfinite(query['gold_score']).all()
                ids, scores = query['top10_ids'], query['top10_scores']
                assert ids.shape == scores.shape == (n, 10)
                assert np.isfinite(scores).all() and ((ids >= 0) & (ids < ne)).all()
                assert (np.diff(np.sort(ids, axis=1), axis=1) > 0).all()
                assert (scores[:, :-1] >= scores[:, 1:]).all()
                filters = ('train', 'train_valid', 'all') if split == 'test' else ('select',)
                reference = result['per_kg'][kg] if split == 'test' else result['selection'][kg]
                for filt in filters:
                    rank = query[f'rank_{filt}']
                    assert rank.shape == (n,) and np.isfinite(rank).all() and ((rank >= 1) & (rank <= ne)).all()
                    measured = metrics(rank)
                    assert all(measured[k] == metadata['metrics'][filt][k] == reference['metrics'][filt][k] for k in METRICS)
                if split == 'test':
                    assert (query['rank_all'] <= query['rank_train_valid']).all()
                    assert (query['rank_train_valid'] <= query['rank_train']).all()
                    per_kg[kg], counts[kg] = metrics(query['rank_train_valid']), n
                else:
                    selection[kg] = metrics(query['rank_select'])
            raw.append({'kg': kg, 'split': split, 'queries': n, 'sha256': sha(path)})
        path = job / kg / 'candidate_embeddings.npy'
        embeddings = np.load(path, mmap_mode='r')
        assert embeddings.shape[0] == ne and np.isfinite(embeddings).all()
        candidates.append({'kg': kg, 'entities': ne, 'shape': list(embeddings.shape), 'sha256': sha(path)})
    macro = {k: float(np.mean([values[k] for values in per_kg.values()])) for k in METRICS}
    assert all(macro[k] == result['macro'][k] for k in METRICS)
    assert abs(np.mean([values['mrr'] for values in selection.values()]) - best['validation_mrr']) < 1e-12
    receipt = {'verified_at': datetime.datetime.now().astimezone().isoformat(), 'run_id': run_id,
               'purpose': 'independent raw-result acceptance', 'passed': True, 'seed': seed, 'run_count': 1,
               'training_validations': config['rounds'], 'best_validation': chosen['evaluation_index'],
               'macro': macro, 'per_kg': per_kg, 'test_queries': sum(counts.values()), 'counts': counts,
               'process_exit_verified': True, 'selection_matches_best_validation': True,
               'all_filter_rank_sets_valid': True, 'source_objects_verified': len(config['source_hashes']),
               'checkpoint_sha256': sha(job / 'best.pt'), 'config_sha256': sha(job / 'config.json'),
               'last_checkpoint_sha256': sha(job / 'last.pt'),
               'last_checkpoint_validation_metadata': last_validation_metadata,
               'raw_artifacts': raw, 'full_entity_candidates': candidates, 'source_configuration_matches_frozen_recipe': True, 'environment': config['environment'],
               'audit_script': str(Path(__file__).resolve()), 'audit_script_sha256': sha(__file__)}
    output = PHASE / 'raw_acceptance' / f'{run_id}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    args = parser.parse_args()
    result = verify(args.run_id)
    print(json.dumps({k: result[k] for k in ('run_id', 'passed', 'test_queries', 'best_validation', 'macro')}))
