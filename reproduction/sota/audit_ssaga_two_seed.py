"""Audit completed SS-AGA/E-PKG seeds17/29; keep seed43 explicitly pending."""
import datetime
from pathlib import Path
import numpy as np
from audit_main_completion import BASE, ROOT, DOMAINS, METRICS, read, save, sha, verify_job


def main():
    seeds = [17, 29]
    plan, server, transfer = [read(BASE/name) for name in ['PLAN.json', 'PUBLICATION_SOURCE_RECEIPT.json', 'TRANSFER_STATE.json']]
    assert server['passed'] and server['plan_sha256'] == sha(BASE/'PLAN.json')
    for name, digest in plan['input_sha256'].items():
        assert sha(ROOT/name) == digest
    for name, digest in plan['frozen_feature_sha256']['depkg'].items():
        assert sha(BASE/'bundle/runtime_ssaga/ssaga_data/datasetdepkg'/name) == digest
    jobs = [f'ssaga_depkg_{kg}_s{seed}' for seed in seeds for kg in DOMAINS['depkg']]
    assert all(transfer['jobs'].get(jid, {}).get('status') == 'returned_verified' for jid in jobs)
    runs = [verify_job(jid, server, plan, transfer) for jid in jobs]
    configs = [read((ROOT/r['result_path']).with_name('config.json')) for r in runs]
    for seed in seeds:
        assert {c['kg'] for c in configs if c['seed']==seed} == set(DOMAINS['depkg'])
    normalize = lambda c: {k:v for k,v in c.items() if k not in {'seed', 'target_language'}}
    assert all(normalize(c['recipe']) == normalize(configs[0]['recipe']) for c in configs)
    per_seed = {str(s):{k:float(np.mean([r['macro'][k] for r in runs if r['seed']==s])) for k in METRICS} for s in seeds}
    mean = {k:float(np.mean([per_seed[str(s)][k] for s in seeds])) for k in METRICS}
    sd = {k:float(np.std([per_seed[str(s)][k] for s in seeds], ddof=1)) for k in METRICS}
    previous = read(BASE/'INTERIM_SEED17_RESULTS.json')['groups'][0]
    assert per_seed['17'] == previous['mean']
    group = {'method':'SS-AGA', 'dataset':'depkg', 'status':'interim_two_seed', 'seeds':seeds,
             'planned_seeds':[17,29,43], 'pending_seeds':[43], 'target_kgs':DOMAINS['depkg'],
             'runs':runs, 'per_seed':per_seed, 'mean':mean, 'sample_sd':sd,
             'aggregation':'Equal target-KG macro within each seed, then seeds17/29 mean and sample SD (ddof=1); seed43 pending',
             'display_percent':{k:f'{100*mean[k]:.2f} +/- {100*sd[k]:.2f}' for k in METRICS}}
    result = {'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'new_training_or_inference':False, 'three_seed_complete':False,
              'plan_sha256':sha(BASE/'PLAN.json'), 'audit_script_sha256':sha(ROOT/'reproduction/sota/audit_main_completion.py'),
              'interim_script_sha256':sha(Path(__file__)), 'server_source_receipt_sha256':sha(BASE/'PUBLICATION_SOURCE_RECEIPT.json'),
              'groups':[group]}
    save(BASE/'INTERIM_TWO_SEED_RESULTS.json', result)
    print(group['display_percent'])


if __name__=='__main__':
    main()
