"""Aggregate registered, accepted saved ranks; never train, infer, or publish a PDF."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / 'reproduction/sota/graph_three_seed'
OLD_RUN = ROOT / 'reproduction/runs/strict_baselines_20260905'
METRICS = ('mrr', 'h1', 'h3', 'h10')
KGS = {'dbp5l': ['el', 'en', 'es', 'fr', 'ja'],
       'depkg': ['de', 'es', 'fr', 'it', 'jp', 'uk'],
       'dwy': ['db', 'wk', 'yg']}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def accepted_seed(method, dataset, seed):
    jid = f'{method.lower()}_{dataset}_s{seed}'
    receipt_path = PHASE / 'raw_acceptance' / f'{jid}.json'
    receipt = read(receipt_path)
    assert receipt['run_id'] == jid and receipt['seed'] == seed
    for flag in ('passed', 'process_exit_verified', 'selection_matches_best_validation',
                 'all_filter_rank_sets_valid', 'source_configuration_matches_frozen_recipe'):
        assert receipt[flag] is True, (jid, flag)
    assert receipt['audit_script_sha256'] == sha(ROOT / 'reproduction/sota/verify_graph_repeat_result.py')
    run = OLD_RUN if seed == 17 else PHASE
    state = read(run / ('queue_state.json' if seed == 17 else 'state.json'))['jobs'][jid]
    assert state['status'] == 'completed'
    assert state['exit_code' if seed == 17 else 'returncode'] == 0
    job = run / 'jobs' / jid
    result = read(job / 'result.json')
    assert result['status'] == 'completed' and result['purpose'] == 'formal'
    assert result['method'] == method and result['dataset'] == dataset and result['seed'] == seed
    assert result['protocol'] == 'kbs-baselines-v1-20260905'
    assert result['best_evaluation'] == receipt['best_validation']
    assert sha(job / 'config.json') == receipt['config_sha256']
    assert sha(job / 'best.pt') == receipt['checkpoint_sha256'] == result['checkpoint_sha256']
    assert sha(job / 'last.pt') == receipt['last_checkpoint_sha256']
    assert set(receipt['per_kg']) == set(KGS[dataset])
    artifacts = {(item['kg'], item['split']): item for item in receipt['raw_artifacts']}
    assert set(artifacts) == {(kg, split) for kg in KGS[dataset] for split in ('test', 'val_select')}
    for (kg, split), item in artifacts.items():
        assert sha(job / kg / f'{split}_queries.npz') == item['sha256']
    assert {item['kg'] for item in receipt['full_entity_candidates']} == set(KGS[dataset])
    for item in receipt['full_entity_candidates']:
        assert sha(job / item['kg'] / 'candidate_embeddings.npy') == item['sha256']
    per_kg = {}
    for kg in KGS[dataset]:
        with np.load(job / kg / 'test_queries.npz') as query:
            rank = query['rank_train_valid'].astype(np.float64)
            assert len(rank) == receipt['counts'][kg] == artifacts[(kg, 'test')]['queries']
            assert np.isfinite(rank).all() and (rank >= 1).all()
            values = {'mrr': float(np.mean(1 / rank)),
                      **{f'h{k}': float(np.mean(rank <= k)) for k in (1, 3, 10)}}
            assert all(values[k] == receipt['per_kg'][kg][k] for k in METRICS)
            per_kg[kg] = values
    macro = {k: float(np.mean([per_kg[kg][k] for kg in KGS[dataset]])) for k in METRICS}
    assert all(macro[k] == receipt['macro'][k] == result['macro'][k] for k in METRICS)
    return {'seed': seed, 'run_id': jid, 'macro': macro, 'per_kg': per_kg,
            'queries': receipt['test_queries'], 'counts': receipt['counts'],
            'best_validation': receipt['best_validation'],
            'result_path': (job / 'result.json').relative_to(ROOT).as_posix(),
            'result_sha256': sha(job / 'result.json'),
            'receipt_path': receipt_path.relative_to(ROOT).as_posix(),
            'receipt_sha256': sha(receipt_path),
            'config_sha256': receipt['config_sha256'],
            'checkpoint_sha256': receipt['checkpoint_sha256'],
            'saved_artifact_hashes_reverified': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    assert str(ROOT) != '/root/zhishitupui', 'Aggregate locally after raw artifacts return'
    plan = read(PHASE / 'PLAN.json')
    assert plan['aggregate_seeds'] == [17, 29, 43]
    groups = []
    for method in plan['methods']:
        for dataset in plan['datasets']:
            missing = [seed for seed in plan['aggregate_seeds']
                       if not (PHASE / 'raw_acceptance' / f'{method.lower()}_{dataset}_s{seed}.json').exists()]
            group = {'method': method, 'dataset': dataset, 'missing_accepted_seeds': missing}
            if missing:
                group['status'] = 'pending'
            else:
                runs = [accepted_seed(method, dataset, seed) for seed in plan['aggregate_seeds']]
                assert all(run['counts'] == runs[0]['counts'] for run in runs)
                means = {k: float(np.mean([run['macro'][k] for run in runs])) for k in METRICS}
                sd = {k: float(np.std([run['macro'][k] for run in runs], ddof=1)) for k in METRICS}
                group.update(status='complete', seeds=plan['aggregate_seeds'], runs=runs,
                             mean=means, sample_sd=sd,
                             display_percent={k: f'{100 * means[k]:.2f} +/- {100 * sd[k]:.2f}' for k in METRICS})
            groups.append(group)
    completed = sum(group['status'] == 'complete' for group in groups)
    if args.require_complete:
        assert completed == 9, f'Only {completed}/9 groups complete; publication is blocked'
    report = {'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'complete' if completed == 9 else 'partial',
              'complete_groups': completed, 'total_groups': 9,
              'protocol': plan['protocol'], 'aggregation': plan['aggregation'],
              'manifest_sha256': sha(PHASE / 'manifest.json'),
              'source_input_freeze_sha256': sha(PHASE / 'SOURCE_INPUT_FREEZE.json'),
              'summary_script_sha256': sha(Path(__file__)),
              'new_training_or_inference': False, 'paper_updated': False, 'groups': groups}
    target = PHASE / ('RESULTS.json' if args.require_complete else 'PARTIAL_RESULTS.json')
    if target.exists():
        archive = PHASE / 'summary_versions'
        archive.mkdir(exist_ok=True)
        prior = archive / f'{target.stem}_{sha(target)}.json'
        if not prior.exists():
            prior.write_bytes(target.read_bytes())
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(target)
    print(json.dumps({'status': report['status'], 'complete_groups': completed,
                      'output': target.relative_to(ROOT).as_posix(),
                      'complete_results': [{k: group[k] for k in ('method', 'dataset', 'display_percent')}
                                           for group in groups if group['status'] == 'complete']}))


if __name__ == '__main__':
    main()
