"""Independently verify a completed selected graph run; never train or fill cells."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / 'reproduction/strict_baselines'
RUN = ROOT / 'reproduction/runs/strict_baselines_20260905'
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
    manifest = read(RUN / 'manifest.json')
    batch = manifest['method_batch']
    assert batch['seeds'] == [17] and run_id in batch['job_ids']
    job = RUN / 'jobs' / run_id
    result, config = read(job / 'result.json'), read(job / 'config.json')
    method, dataset = result['method'], result['dataset']
    assert method in ('LSMGA', 'DMKGC', 'IMKGC') and dataset in KGS
    assert run_id == f'{method.lower()}_{dataset}_s17'
    assert result['status'] == 'completed' and result['purpose'] == 'formal' and result['full_data']
    assert result['seed'] == config['seed'] == 17 and config['input_facts'] == 'train-only'
    assert set(result['per_kg']) == set(result['selection']) == set(config['domain_order']) == set(KGS[dataset])
    state = read(RUN / 'queue_state.json')['jobs'][run_id]
    assert state['status'] == 'completed'
    try:
        process = psutil.Process(state['pid'])
        assert abs(process.create_time() - state['process_created_at']) > 1, 'Completed run still owns a live process'
    except psutil.NoSuchProcess:
        pass
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
    assert sha(result['checkpoint']) == result['checkpoint_sha256']
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
        assert last['best_mrr'] == best['validation_mrr']
    assert best['config'] == config
    assert all(not torch.is_tensor(v) or torch.isfinite(v).all().item() for v in best['model'].values())
    for name, checksum in config['source_hashes'].items():
        assert sha(RUN / 'code_objects' / (checksum + Path(name).suffix)) == checksum
    adaptation = config.get('dwy_adaptation')
    if method == 'IMKGC' and dataset == 'dwy':
        assert adaptation['status'] == 'passed'
        for name, checksum in adaptation['input_sha256'].items():
            assert sha(Path(adaptation['data_root']) / name) == checksum

    per_kg, selection, counts, raw, candidates = {}, {}, {}, [], []
    for kg in KGS[dataset]:
        frozen_path = SUITE / 'data_manifests' / f'{dataset}_{kg}.json'
        frozen = read(frozen_path)
        if adaptation:
            assert sha(frozen_path) == adaptation['domains'][kg]['split_manifest_sha256']
        for name, checksum in frozen['files'].items():
            assert sha(ROOT / name) == checksum
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
    prefix = f'T2.{TABLE_DATASET[dataset]}.{method.lower()}.'
    with (ROOT / 'outputs/kbs_main_tables/cells_results.csv').open(encoding='utf-8-sig') as stream:
        rows = [row for row in csv.DictReader(stream) if row['cell_id'].startswith(prefix)]
    assert len(rows) == 3
    for row in rows:
        assert float(row['value']) == macro[row['cell_id'].rsplit('.', 1)[1]]
        assert row['seed'] == '17' and row['standard_deviation'] == '' and row['run_id'] == run_id
        assert row['source_type'] == 'rerun' and row['filter_protocol'] == 'train+valid'
    audit = read(RUN / 'results/table_fill_audit.json')
    assert audit['rejected_runs'] == []
    assert {'method': method, 'dataset': dataset, 'condition': 'full', 'seeds': [17]} in audit['single_run_groups']
    receipt = {'verified_at': datetime.datetime.now().astimezone().isoformat(), 'run_id': run_id,
               'purpose': 'independent raw-result acceptance', 'passed': True, 'seed': 17, 'run_count': 1,
               'training_validations': config['rounds'], 'best_validation': chosen['evaluation_index'],
               'macro': macro, 'per_kg': per_kg, 'test_queries': sum(counts.values()), 'counts': counts,
               'process_exit_verified': True, 'selection_matches_best_validation': True,
               'all_filter_rank_sets_valid': True, 'source_objects_verified': len(config['source_hashes']),
               'checkpoint_sha256': sha(job / 'best.pt'), 'config_sha256': sha(job / 'config.json'),
               'raw_artifacts': raw, 'full_entity_candidates': candidates, 'table_standard_deviation': None,
               'audit_script': str(Path(__file__).resolve()), 'audit_script_sha256': sha(__file__)}
    output = RUN / 'results/single_run_acceptance' / f'{run_id}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    args = parser.parse_args()
    result = verify(args.run_id)
    print(json.dumps({k: result[k] for k in ('run_id', 'passed', 'test_queries', 'best_validation', 'macro')}))
