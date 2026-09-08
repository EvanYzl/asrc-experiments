"""Close this campaign after locked acceptance; never kill unrelated processes."""
import datetime
import json
import subprocess
from frozen_data import ROOT,atomic_json

base=ROOT/'reproduction/sota'
report=json.loads((base/'SINGLE_SEED_ACCEPTANCE.json').read_text())
assert report['verified_datasets']==4 and report['threshold_met']
active=[]
for queue in (base/'single_seed').glob('*/queue/state.json'):
    state=json.loads(queue.read_text())
    if any(x.get('status')=='running' for x in state.get('jobs',{}).values()):active.append(str(queue))
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
running=[line for line in processes.splitlines() if any(token in line for token in ['python reproduction/sota/train_complex.py','python reproduction/sota/evaluate_single.py','python reproduction/sota/run_sota_queue.py','kgc/bin/python reproduction/sota/evaluate_single.py'])]
assert not running,running
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
scope=json.loads((base/'TASK_SCOPE.json').read_text())
scope.update(status='experiments_stopped_delivery_only',current_dataset=None,locked_datasets=['dbp5l','depkg','dwy','wk3l'],
             verdict=report['verdict'],stop_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),further_experiments_authorized=False)
atomic_json(base/'TASK_SCOPE.json',scope)
atomic_json(base/'EXPERIMENT_STOP_RECEIPT.json',{'timestamp':scope['stop_timestamp'],'campaign_processes_running':running,'gpu_snapshot':gpu,
    'verdict':report['verdict'],'strict_wins':report['strict_wins'],'next':'Only synchronize and verify documents/raw artifacts, then await user.',
    'unrelated_local_baseline_task':'Not touched; outside this campaign.'})
with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## All single-seed experiments stopped — '+scope['stop_timestamp']+'\n')
    f.write('Purpose: enforce the user stopping condition after all four locked one-test results.\n')
    f.write('Config/seed: seed17 only, four frozen ASRC recipes; no new experiments. Command: python reproduction/sota/verify_and_aggregate_single.py followed by stop_single_seed.py.\n')
    f.write('PID/GPU: no campaign training/evaluation/scheduler process remains; GPU snapshot is in EXPERIMENT_STOP_RECEIPT.json. No unrelated process was signaled.\n')
    f.write('Results: all raw test triples/order, three filter rank variants, exact macro metrics, code/checkpoint/reference hashes verified; 12/12 strict wins. '+report['verdict']+'.\n')
    f.write('Evidence: SINGLE_SEED_ACCEPTANCE.json, SINGLE_SEED_COMPARISON.csv, single_seed/*/LOCKED_RESULT.json.\n')
    f.write('Conclusion: tests locked, experiments stopped. Unknown/proxy references and repeat-count differences prevent a full comparable SOTA claim. Next: final documents and raw artifacts synchronization with hash verification, then wait for user.\n')
print(json.dumps({'experiments_running':len(running),'verdict':report['verdict'],'delivery_only':True},ensure_ascii=False))
