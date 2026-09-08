"""Audit saved WK3l/SS-AGA results and aggregate complete fixed-seed groups.

This command does no training or inference. Source receipts are captured from
the existing server queue; completed raw files remain in their original paths.
"""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SIDE = ROOT / 'reproduction/language_table_seed17'
BASE = SIDE / 'main_completion_20260907'
EVIDENCE = BASE / 'publication_evidence'
METRICS = ('mrr', 'h1', 'h3', 'h10')
SEEDS = [17, 29, 43]
DOMAINS = {'dbp5l': ['el', 'en', 'es', 'fr', 'ja'], 'depkg': ['de', 'es', 'fr', 'it', 'jp', 'uk']}
torch.set_num_threads(2)


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def raw_metrics(rank):
    r = rank.astype(np.float64)
    return {'mrr': float(np.mean(1 / r)), **{f'h{k}': float(np.mean(r <= k)) for k in (1, 3, 10)}}


def verify_queries(folder, dataset, kg, result, selection_name):
    manifest = read(ROOT / f'reproduction/strict_baselines/data_manifests/{dataset}_{kg}.json')
    for name, digest in manifest['files'].items():
        assert sha(ROOT / name.replace('\\', '/')) == digest
    if dataset == 'wk3l':
        raw = ROOT / 'data/raw/atransn' / ('WK3l-15k_FR' if kg == 'fr' else 'WK3l-15k_EN_F')
        test, valid = raw / 'test_triple_id.txt', raw / 'valid_triple_id.txt'
    else:
        raw = ROOT / f'data/raw/dmkgc/dataset{dataset}/kg'
        test, valid = raw / f'{kg}-test.tsv', raw / f'{kg}-val.tsv'
    expected = {'test': np.loadtxt(test, dtype=np.int64, ndmin=2),
                'val_select': np.loadtxt(valid, dtype=np.int64, ndmin=2)[manifest['val_select_indices']]}
    artifacts, values = [], {}
    for split, triples in expected.items():
        filename = 'test_queries' if split == 'test' else selection_name
        path = folder / (filename + '.npz')
        metadata = read(path.with_suffix('.json'))
        reference = result['test'] if split == 'test' else result['selection']
        with np.load(path, allow_pickle=False) as a:
            n, ne = len(triples), manifest['entities']
            assert n == manifest['counts'][split]
            assert np.array_equal(a['triples'], triples)
            assert np.array_equal(a['query_index'], np.arange(n))
            assert a['gold_score'].dtype == np.float32 and np.isfinite(a['gold_score']).all()
            ids, scores = a['top10_ids'], a['top10_scores']
            assert ids.shape == scores.shape == (n, 10)
            assert np.isfinite(scores).all() and ((ids >= 0) & (ids < ne)).all()
            assert (np.diff(np.sort(ids, axis=1), axis=1) > 0).all()
            assert (scores[:, :-1] >= scores[:, 1:]).all()
            assert ((scores[:, :-1] != scores[:, 1:]) | (ids[:, :-1] < ids[:, 1:])).all()
            filters = ('train', 'train_valid', 'all') if split == 'test' else ('select',)
            for filt in filters:
                rank = a['rank_' + filt]
                assert rank.shape == (n,) and np.isfinite(rank).all() and ((rank >= 1) & (rank <= ne)).all()
                assert (rank == np.floor(rank)).all()
                measured = raw_metrics(rank)
                assert all(measured[k] == reference['metrics'][filt][k] == metadata['metrics'][filt][k] for k in METRICS)
                values[split + '/' + filt] = measured
            if split == 'test':
                assert (a['rank_all'] <= a['rank_train_valid']).all()
                assert (a['rank_train_valid'] <= a['rank_train']).all()
        artifacts.append({'path': path.relative_to(ROOT).as_posix(), 'queries': n, 'sha256': sha(path)})
    return values, artifacts, manifest


def verify_sources(jid, out, config, server, plan):
    local_seed = jid.endswith('_s17') and config['dataset'] == 'wk3l'
    if local_seed:
        objects = SIDE / 'code_objects'
    else:
        proof = server['jobs'][jid]
        assert proof['config_sha256'] == sha(out / 'config.json')
        assert proof['result_sha256'] == sha(out / 'result.json')
        assert proof['source_hashes'] == config['source_hashes']
        assert proof['state']['status'] == 'completed' and proof['state'].get('exit_code', 0) == 0
        objects = EVIDENCE / 'source_objects'
    frozen = read(BASE / ('early_bundle/EARLY_HASHES.json' if config['dataset'] == 'wk3l' else 'bundle/BUNDLE_HASHES.json'))
    for name, digest in config['source_hashes'].items():
        obj = objects / (digest + '.py')
        assert sha(obj) == digest
        normalized = name.replace('\\', '/')
        if local_seed:
            assert sha(ROOT / normalized) == digest
            continue
        short = normalized.split('reproduction/main_tables_completion_20260907/', 1)[1]
        if short.startswith('contexts/'):
            short = 'runtime_graph/' + short.rsplit('/', 1)[1]
        if short == 'sources/LSMGA/run_model.py':
            old = SIDE / 'code_objects' / (read(SIDE / 'jobs/lsmga_wk3l_s17/config.json')['source_hashes']['reproduction\\language_table_seed17\\sources\\LSMGA\\run_model.py'] + '.py')
            expected = old.read_text(encoding='utf-8').replace(
                "os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'",
                "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0') # Respect scheduler GPU isolation.")
            assert obj.read_text(encoding='utf-8') == expected
            repair = read(BASE / 'server_snapshot/versions/cuda_mapping_fix/REPAIR.json')
            assert repair['after_sha256'] == digest
        else:
            assert frozen[short] == digest, (jid, short, digest, frozen.get(short))
    return len(config['source_hashes'])


def verify_job(jid, server, plan, transfer):
    is_graph = '_wk3l_' in jid
    seed = int(jid.rsplit('_s', 1)[1])
    local_seed = is_graph and seed == 17
    out = SIDE / 'jobs' / jid if local_seed else BASE / 'returned_jobs' / jid
    result, config = read(out / 'result.json'), read(out / 'config.json')
    assert result['status'] == 'completed' and result['full_data'] is True
    assert result['seed'] == config['seed'] == seed
    assert result['method'] == config['method']
    assert result['dataset'] == config['dataset']
    if not local_seed:
        tr = transfer['jobs'][jid]
        assert tr['status'] == 'returned_verified' and tr['verification']['status'] == 'passed'
        assert sha(BASE / 'returns' / (jid + '.tar.gz')) == tr['archive_sha256'] == server['jobs'][jid]['return_archive_sha256']
        for name, digest in read(out / 'RETURN_HASHES.json').items():
            assert sha(out / name) == digest
    count = verify_sources(jid, out, config, server, plan)
    frozen = read(out / 'FINAL_EVALUATION_FREEZE.json')
    assert frozen['seed'] == seed
    assert sha(out / 'best.pt') == frozen['checkpoint_sha256'] == result['checkpoint_sha256']
    best = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    assert all(not torch.is_tensor(v) or torch.isfinite(v).all().item() for v in best['model'].values())
    history = [json.loads(line) for line in (out / 'learning_curve.jsonl').read_text().splitlines()]
    raw = []
    if is_graph:
        assert result['purpose'] == config['purpose'] == 'formal'
        assert result['protocol'] == 'kbs-baselines-v1-20260905'
        assert config['domain_order'] == ['fr'] and config['input_facts'] == 'train-only'
        rounds = 50 if config['method'] == 'LSMGA' else 30
        assert config['rounds'] == frozen['completed_rounds'] == rounds
        assert [x['evaluation_index'] for x in history] == list(range(1, rounds + 1))
        assert all(x['val_select_macro']['mrr'] == x['per_kg']['fr']['metrics']['select']['mrr'] for x in history)
        chosen = max(history, key=lambda x: x['val_select_macro']['mrr'])
        assert chosen['evaluation_index'] == best['evaluation_index'] == result['best_evaluation'] == frozen['selected_evaluation']
        assert best['validation_mrr'] == chosen['val_select_macro']['mrr'] and best['config'] == config
        last = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        assert last['completed_round'] == rounds - 1
        reference = read(SIDE / f"jobs/{config['method'].lower()}_wk3l_s17/config.json")
        omit = {'seed', 'data_path', 'resume_checkpoint', 'v'}
        normalize = lambda a: {k: v for k, v in a.items() if k not in omit}
        assert normalize(reference['parsed_arguments_before_author_main']) == normalize(config['parsed_arguments_before_author_main'])
        assert set(result['per_kg']) == set(result['selection']) == {'fr', 'en'}
        for kg in ('fr', 'en'):
            mm, artifacts, manifest = verify_queries(out / kg, 'wk3l', kg, {'test': result['per_kg'][kg], 'selection': result['selection'][kg]}, 'val_select_queries')
            embeddings = np.load(out / kg / 'candidate_embeddings.npy', mmap_mode='r')
            assert embeddings.shape[0] == manifest['entities'] and np.isfinite(embeddings).all()
            raw.extend(artifacts)
            if kg == 'fr':
                macro = mm['test/all']
                assert macro == result['macro']
                assert mm['val_select/select']['mrr'] == best['validation_mrr']
        selected = chosen['evaluation_index']
    else:
        assert config['method'] == 'SS-AGA' and config['recipe']['round'] == frozen['completed_rounds'] == 25
        assert [x['round'] for x in history] == list(range(25))
        chosen = max(history, key=lambda x: x['val_select']['mrr'])
        assert chosen['round'] == best['selected_round'] == result['selected_round'] == frozen['selected_round']
        assert best['seed'] == seed and best['validation_mrr'] == chosen['val_select']['mrr']
        assert config['final_filter'] == 'train, current gold retained'
        assert config['supporter_facts'] == 'train+public validation as released'
        feature = EVIDENCE / (config['dataset'] + '_feature_manifest.json')
        assert sha(feature) == config['feature_manifest_sha256']
        assert read(feature)['feature_sha256'] == plan['frozen_feature_sha256'][config['dataset']]['entity_embeddings.npy']
        mm, raw, manifest = verify_queries(out, config['dataset'], config['kg'], {'test': result, 'selection': result['selection']}, 'selection_queries')
        assert config['dataset_hash'] == manifest['dataset_hash']
        assert abs(mm['val_select/select']['mrr'] - best['validation_mrr']) < 1e-10
        path = out / 'selected_candidate_embeddings.npy'
        assert sha(path) == result['candidate_embedding_sha256']
        embeddings = np.load(path, mmap_mode='r')
        assert embeddings.shape[0] == manifest['entities'] and np.isfinite(embeddings).all()
        assert (out / 'selected_target_graph.pt').exists()
        macro = mm['test/train']
        selected = chosen['round']
    receipt = {'passed': True, 'run_id': jid, 'seed': seed, 'macro': macro, 'selected_validation': selected,
               'source_objects_verified': count, 'raw_artifacts': raw,
               'result_path': (out / 'result.json').relative_to(ROOT).as_posix(),
               'result_sha256': sha(out / 'result.json'), 'config_sha256': sha(out / 'config.json'),
               'checkpoint_sha256': sha(out / 'best.pt'), 'audit_script_sha256': sha(Path(__file__)),
               'server_source_receipt_sha256': sha(BASE / 'PUBLICATION_SOURCE_RECEIPT.json')}
    save(BASE / 'publication_acceptance' / (jid + '.json'), receipt)
    print(json.dumps({'accepted': jid, 'selected_validation': selected}), flush=True)
    return receipt


def main():
    plan, server, transfer = read(BASE / 'PLAN.json'), read(BASE / 'PUBLICATION_SOURCE_RECEIPT.json'), read(BASE / 'TRANSFER_STATE.json')
    assert server['passed'] and server['plan_sha256'] == sha(BASE / 'PLAN.json')
    for name, digest in plan['input_sha256'].items():
        assert sha(ROOT / name) == digest
    for dataset, files in plan['frozen_feature_sha256'].items():
        for name, digest in files.items():
            assert sha(BASE / 'bundle/runtime_ssaga/ssaga_data' / ('dataset' + dataset) / name) == digest
    specs = [(method, 'wk3l', [f'{method.lower()}_wk3l_s{s}' for s in SEEDS]) for method in ('LSMGA', 'DMKGC', 'IMKGC')]
    specs += [('SS-AGA', ds, [f'ssaga_{ds}_{kg}_s{s}' for s in SEEDS for kg in DOMAINS[ds]]) for ds in DOMAINS]
    groups = []
    for method, dataset, jobs in specs:
        missing = [jid for jid in jobs if not ('_wk3l_s17' in jid or jid in server['jobs'] and transfer['jobs'].get(jid, {}).get('status') == 'returned_verified')]
        group = {'method': method, 'dataset': dataset, 'seeds': SEEDS, 'status': 'pending' if missing else 'complete', 'missing_jobs': missing}
        if not missing:
            runs = [verify_job(jid, server, plan, transfer) for jid in jobs]
            if method == 'SS-AGA':
                configs = [read(ROOT / r['result_path']).get('kg') for r in runs]
                assert set(configs) == set(DOMAINS[dataset])
                recipes = [read((ROOT / r['result_path']).with_name('config.json'))['recipe'] for r in runs]
                normalize = lambda a: {k: v for k, v in a.items() if k not in {'seed', 'target_language'}}
                assert all(normalize(x) == normalize(recipes[0]) for x in recipes)
            per_seed = {str(s): {k: float(np.mean([r['macro'][k] for r in runs if r['seed'] == s])) for k in METRICS} for s in SEEDS}
            mean = {k: float(np.mean([per_seed[str(s)][k] for s in SEEDS])) for k in METRICS}
            sd = {k: float(np.std([per_seed[str(s)][k] for s in SEEDS], ddof=1)) for k in METRICS}
            group.update(runs=runs, per_seed=per_seed, mean=mean, sample_sd=sd,
                         display_percent={k: f'{100 * mean[k]:.2f} +/- {100 * sd[k]:.2f}' for k in METRICS})
        groups.append(group)
    report = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'complete_groups': sum(g['status'] == 'complete' for g in groups), 'total_groups': 5,
              'plan_sha256': sha(BASE / 'PLAN.json'), 'audit_script_sha256': sha(Path(__file__)),
              'aggregation': 'Equal target-KG macro per seed, then seeds17/29/43 mean and sample SD (ddof=1); WK3l uses FR only',
              'new_training_or_inference': False, 'groups': groups}
    save(BASE / 'PUBLICATION_RESULTS.json', report)
    print(json.dumps({'complete_groups': report['complete_groups'], 'results': [{k: g[k] for k in ('method', 'dataset', 'display_percent')} for g in groups if g['status'] == 'complete']}))


if __name__ == '__main__':
    main()
