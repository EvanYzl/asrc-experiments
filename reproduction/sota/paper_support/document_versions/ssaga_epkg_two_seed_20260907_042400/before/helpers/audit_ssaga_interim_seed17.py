"""Audit the already returned SS-AGA/E-PKG seed17; never infer a seed SD."""
import datetime
import numpy as np
from audit_main_completion import BASE, EVIDENCE, ROOT, DOMAINS, METRICS, read, save, sha, verify_job


def main():
    plan = read(BASE / 'PLAN.json')
    server = read(BASE / 'PUBLICATION_SOURCE_RECEIPT.json')
    transfer = read(BASE / 'TRANSFER_STATE.json')
    assert server['passed'] and server['plan_sha256'] == sha(BASE / 'PLAN.json')
    for name, digest in plan['input_sha256'].items():
        assert sha(ROOT / name) == digest
    for name, digest in plan['frozen_feature_sha256']['depkg'].items():
        assert sha(BASE / 'bundle/runtime_ssaga/ssaga_data/datasetdepkg' / name) == digest
    jobs = [f'ssaga_depkg_{kg}_s17' for kg in DOMAINS['depkg']]
    runs = [verify_job(jid, server, plan, transfer) for jid in jobs]
    configs = [read((ROOT / r['result_path']).with_name('config.json')) for r in runs]
    assert {c['kg'] for c in configs} == set(DOMAINS['depkg'])
    normalize = lambda x: {k: v for k, v in x.items() if k not in {'seed', 'target_language'}}
    assert all(c['seed'] == 17 and normalize(c['recipe']) == normalize(configs[0]['recipe']) for c in configs)
    values = {k: float(np.mean([r['macro'][k] for r in runs])) for k in METRICS}
    group = {'method': 'SS-AGA', 'dataset': 'depkg', 'status': 'interim_single_seed',
             'seeds': [17], 'planned_seeds': [17, 29, 43], 'pending_seeds': [29, 43],
             'target_kgs': DOMAINS['depkg'], 'runs': runs, 'mean': values, 'sample_sd': None,
             'aggregation': 'Equal target-KG macro for seed17 only; not a three-seed estimate',
             'display_percent': {k: f'{100*v:.2f}' for k, v in values.items()}}
    report = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'new_training_or_inference': False, 'three_seed_complete': False,
              'plan_sha256': sha(BASE / 'PLAN.json'),
              'audit_script_sha256': sha(ROOT / 'reproduction/sota/audit_main_completion.py'),
              'interim_script_sha256': sha(ROOT / 'reproduction/sota/audit_ssaga_interim_seed17.py'),
              'server_source_receipt_sha256': sha(BASE / 'PUBLICATION_SOURCE_RECEIPT.json'),
              'groups': [group]}
    save(BASE / 'INTERIM_SEED17_RESULTS.json', report)
    print(group['display_percent'])


if __name__ == '__main__':
    main()
