"""Adapt the existing raw-result verifier to both registered seed locations."""
from pathlib import Path
root = Path(__file__).resolve().parents[2]
original = root / 'reproduction/strict_baselines/verify_single_graph_result.py'
target = root / 'reproduction/sota/verify_graph_repeat_result.py'
assert not target.exists()
s = original.read_text(encoding='utf-8')
s = s.replace('import json\n', 'import json\nimport re\n')
s = s.replace("RUN = ROOT / 'reproduction/runs/strict_baselines_20260905'", "OLD_RUN = ROOT / 'reproduction/runs/strict_baselines_20260905'\nPHASE = ROOT / 'reproduction/sota/graph_three_seed'\ntorch.set_num_threads(2)")
start = s.index('    manifest = read(RUN /')
stop = s.index("    if method == 'LSMGA':", start)
s = s[:start] + '''    match = re.fullmatch(r'(lsmga|dmkgc|imkgc)_(dbp5l|depkg|dwy)_s(17|29|43)', run_id)
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
''' + s[stop:]
s = s.replace("assert sha(result['checkpoint']) == result['checkpoint_sha256']", "assert sha(job / 'best.pt') == result['checkpoint_sha256']")
s = s.replace("assert sha(RUN / 'code_objects' / (checksum + Path(name).suffix)) == checksum", "assert sha(run / 'code_objects' / (checksum + Path(name).suffix)) == checksum\n        expected = freeze['sources'][name.replace(chr(92), '/')]['seed17_sha256' if seed == 17 else 'server_sha256']\n        assert checksum == expected")
s = s.replace("sha(Path(adaptation['data_root']) / name)", "sha(ROOT / 'data/raw/dmkgc/datasetdwy' / name)")
s = s.replace('sha(ROOT / name) == checksum', "sha(ROOT / name.replace(chr(92), '/')) == checksum")
start = s.index("    prefix = f'T2.")
stop = s.index("    receipt = {", start)
s = s[:start] + s[stop:]
s = s.replace("'passed': True, 'seed': 17, 'run_count': 1,", "'passed': True, 'seed': seed, 'run_count': 1,")
s = s.replace("output = RUN / 'results/single_run_acceptance' / f'{run_id}.json'", "output = PHASE / 'raw_acceptance' / f'{run_id}.json'")
s = s.replace("'table_standard_deviation': None,", "'source_configuration_matches_frozen_recipe': True, 'environment': config['environment'],")
s = s.replace('"""Independently verify a completed selected graph run; never train or fill cells."""', '"""Audit saved ranks/checkpoints for the registered graph seeds; never run inference or fill tables.\nAdapted from strict_baselines/verify_single_graph_result.py; publication is a separate step.\n"""')
assert 'RUN /' not in s.replace('OLD_RUN /', '')
assert 'single_run_groups' not in s and "assert result['seed'] == config['seed'] == 17" not in s
compile(s, str(target), 'exec')
target.write_text(s, encoding='utf-8')
print(target)
