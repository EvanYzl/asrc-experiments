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


def verify_seed17_migration(job, config, freeze, history):
    """Verify only the recorded IMKGC/E-PKG full-state host migration.

    Keep the original source/input freeze intact. Check actual old/new bytes and
    the exact path/recording edits, alongside the preserved 26-round prefix.
    """
    assert job.name == 'imkgc_depkg_s17'
    migration = ROOT / 'reproduction/language_table_seed17/migration_20260907'
    bundle = migration / 'bundle'
    checkpoint = migration / 'local_epkg_checkpoint'
    original_path = bundle / 'provenance/local_epkg_config.json'
    assert sha(original_path) == 'c1f447a449eef166410b52ef367cd6f3f43ba8bc1c3f2f0c8f87a5c7e7f9e57d'
    original = read(original_path)
    deployment = read(bundle / 'DEPLOYMENT.json')
    server = read(PHASE / 'MIGRATION_SERVER_SOURCE_RECEIPT.json')
    evidence = PHASE / 'migration_provenance'
    assert server['passed'] and not server['worker_alive'] and not server['controller_alive']
    assert server['config_sha256'] == sha(job / 'config.json')
    assert server['sources'] == config['source_hashes']
    assert server['deployment_sha256'] == sha(bundle / 'DEPLOYMENT.json') == sha(evidence / 'server_deployment.json')
    assert server['state_sha256'] == sha(evidence / 'server_state.json')
    assert server['log_sha256'] == sha(evidence / 'imkgc_depkg_s17.log')
    state = read(evidence / 'server_state.json')
    assert state['exit_code'] == 0 and state['status'] == 'ready_to_return'
    transfer = read(migration / 'TRANSFER_STATE.json')['jobs'][job.name]
    assert transfer['status'] == 'returned_verified' and transfer['verification']['status'] == 'passed'
    assert sha(migration / 'returns' / (job.name + '.tar.gz')) == state['archive_sha256']
    returned = migration / 'returned_jobs' / job.name
    return_hashes = read(returned / 'RETURN_HASHES.json')
    for name, checksum in return_hashes.items():
        assert sha(returned / name) == checksum == sha(job / name), name

    prefix = 'reproduction/language_table_seed17_remote_20260907/'
    def relocated(name):
        name = name.replace(chr(92), '/')
        assert name.startswith(prefix)
        short = name[len(prefix):]
        old = ('reproduction/strict_baselines/' + short[len('runtime_epkg/'):]
               if short.startswith('runtime_epkg/') else 'reproduction/' + short)
        return short, old

    allowed = {
        'common.py': [
            ['ROOT = Path(__file__).resolve().parents[2]', "ROOT = Path('/root/zhishitupui')\nBASE = Path(__file__).resolve().parent.parent"],
            ["SUITE = ROOT / 'reproduction' / 'strict_baselines'", 'SUITE = Path(__file__).resolve().parent'],
            ["objects=ROOT/'reproduction/runs/strict_baselines_20260905/code_objects'", "objects=BASE/'code_objects'"],
            ['files = {str(p.relative_to(ROOT)): sha256(p)', "files = {str(p.relative_to(ROOT)).replace('/', chr(92)): sha256(p)"]],
        'run_graph_baseline.py': [
            ["source=ROOT/'reproduction/sources'/opts.method", "source=BASE/'sources'/opts.method"],
            ["dm_path=ROOT/'reproduction/sources/DMKGC/src/utils.py'", "dm_path=BASE/'sources/DMKGC/src/utils.py'"],
            ["        with torch.no_grad():\n            test=tester.evaluate_split('test',save=True)",
             "        atomic_json(out/'FINAL_EVALUATION_FREEZE.json', {'seed':opts.seed,'completed_rounds':rounds,'selected_evaluation':state['evaluation_index'],'checkpoint_sha256':sha256(out/'best.pt'),'selection':'val_select only','migration':'full-state continuation from local run'})\n        with torch.no_grad():\n            test=tester.evaluate_split('test',save=True)"]]
    }
    original_sources = {k.replace(chr(92), '/'): v for k, v in original['source_hashes'].items()}
    mapped_sources = {}
    for name, checksum in config['source_hashes'].items():
        short, old = relocated(name)
        old_hash = freeze['sources'][old]['seed17_sha256']
        assert old_hash == original_sources[old]
        old_object = OLD_RUN / 'code_objects' / (old_hash + Path(old).suffix)
        assert sha(old_object) == old_hash and sha(bundle / short) == checksum
        if short.startswith('runtime_epkg/'):
            edits = next(x for x in deployment['path_and_recording_changes']
                         if x['source'].replace(chr(92), '/').endswith('/' + old))
            assert edits['source_sha256'] == old_hash and edits['deployed_sha256'] == checksum
            assert edits['replacements'] == allowed[Path(short).name]
            text = old_object.read_text(encoding='utf-8')
            for before, after in edits['replacements']:
                assert before in text
                text = text.replace(before, after)
            assert text == (bundle / short).read_text(encoding='utf-8')
        else:
            assert checksum == old_hash
        mapped_sources[old] = checksum
    assert set(mapped_sources) == set(original_sources)
    allowed_fields = {'environment', 'arguments', 'parsed_arguments_before_author_main', 'message_graph', 'source_hashes'}
    assert {k for k in set(original) | set(config) if original.get(k) != config.get(k)} <= allowed_fields
    paths = {'data_path', 'resume_checkpoint'}
    before_args, after_args = original['parsed_arguments_before_author_main'], config['parsed_arguments_before_author_main']
    assert {k: v for k, v in before_args.items() if k not in paths} == {k: v for k, v in after_args.items() if k not in paths}
    assert after_args['data_path'] == '/root/zhishitupui/' + prefix + 'runtime_epkg/graph_data/train_only_k10_h2/dataset'
    assert after_args['resume_checkpoint'] == '/root/zhishitupui/' + prefix + 'jobs/imkgc_depkg_s17/last.pt'
    assert config['message_graph'] == after_args['data_path'] + 'depkg/strict_graph_manifest.json'
    argv = list(original['arguments'])
    for key in paths:
        index = argv.index('--' + key) + 1
        assert argv[index] == before_args[key]
        argv[index] = after_args[key]
    assert argv == config['arguments']
    env_before, env_after = original['environment'], config['environment']
    assert {k: v for k, v in env_before.items() if k not in {'python', 'device'}} == {k: v for k, v in env_after.items() if k not in {'python', 'device'}}
    assert env_before['python'].startswith('3.10.19 ') and env_after['python'].startswith('3.10.19 ')
    assert env_before['device'] == 'NVIDIA GeForce RTX 5080' and env_after['device'] == 'NVIDIA GeForce RTX 2080 Ti'
    for old, checksum in freeze['frozen_graph_cache'].items():
        if '/datasetdepkg/' not in old:
            continue
        target = bundle / old.replace('reproduction/strict_baselines/', 'runtime_epkg/', 1)
        assert sha(ROOT / old) == checksum
        if target.name == 'strict_graph_manifest.json':
            expected = read(ROOT / old)
            expected['source_files'] = {k.replace(chr(92), '/'): v for k, v in expected['source_files'].items()}
            assert read(target) == expected
        else:
            assert sha(target) == checksum
    captured = read(checkpoint / 'MIGRATION_CHECKPOINT.json')
    assert sha(checkpoint / 'MIGRATION_CHECKPOINT.json') == sha(job / 'MIGRATION_CHECKPOINT.json')
    assert captured['seed'] == 17 and captured['completed_rounds'] == 26 and captured['training_budget'] == 50
    assert state['preflight']['full_state_verified'] and state['preflight']['completed_rounds'] == 26
    for name, checksum in captured['sha256'].items():
        assert sha(checkpoint / name) == checksum
    prefix_history = [json.loads(line) for line in (checkpoint / 'learning_curve.jsonl').read_text().splitlines()]
    assert len(prefix_history) == 26 and prefix_history == history[:26]
    full_state = torch.load(checkpoint / 'last.pt', map_location='cpu', weights_only=False)
    assert full_state['completed_round'] == 25
    assert all(name in full_state for name in captured['resume_fields_verified'])
    final_freeze = read(job / 'FINAL_EVALUATION_FREEZE.json')
    assert final_freeze['seed'] == 17 and final_freeze['completed_rounds'] == 50
    assert final_freeze['checkpoint_sha256'] == sha(job / 'best.pt')
    assert final_freeze['selected_evaluation'] == max(history, key=lambda x: x['val_select_macro']['mrr'])['evaluation_index']
    return {'passed': True, 'original_config_sha256': sha(original_path),
            'preserved_rounds': 26, 'unchanged_model_sources': 11, 'verified_path_recording_wrappers': 2,
            'server_source_receipt_sha256': sha(PHASE / 'MIGRATION_SERVER_SOURCE_RECEIPT.json'),
            'returned_archive_sha256': state['archive_sha256'], 'return_files_verified': len(return_hashes),
            'source_input_freeze_unchanged': True, 'checkpoint_config_matches_exactly': True}


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
    migration_receipt = None
    if seed == 17 and state.get('migration'):
        migration_receipt = verify_seed17_migration(job, config, freeze, history)
    else:
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
               'migration_provenance': migration_receipt,
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
