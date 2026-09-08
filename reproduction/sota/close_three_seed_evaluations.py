"""Record completion of the eight authorized evaluations; no additional jobs."""
import datetime
import json
import subprocess
from frozen_data import ROOT,atomic_json
base=ROOT/'reproduction/sota';phase=base/'three_seed';r=json.loads((phase/'RESULTS.json').read_text())
state=json.loads((phase/'queue/state.json').read_text());assert len(state['jobs'])==8 and all(x['status']=='completed' for x in state['jobs'].values())
running=[line for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if any(token in line for token in ['python reproduction/sota/train_complex.py','python reproduction/sota/evaluate_three_seed.py','kgc/bin/python reproduction/sota/evaluate_three_seed.py','python reproduction/sota/run_sota_queue.py'])]
assert not running,running
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
atomic_json(phase/'EXPERIMENTS_COMPLETED.json',{'timestamp':stamp,'completed_new_tests':8,'reused_seed17_tests':4,'training_runs':0,'baseline_runs':0,'campaign_processes_running':running,'gpu_snapshot':gpu})
scope=json.loads((base/'TASK_SCOPE.json').read_text());scope.update(status='three_seed_delivery_only',experiment_completed_at=stamp,further_experiments_authorized=False);atomic_json(base/'TASK_SCOPE.json',scope)
with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## Three-seed tests and raw-rank verification complete — '+stamp+'\nPurpose: finish exactly the eight missing ASRC tests and aggregate three seeds.\n')
    f.write('Config/seed/commands: frozen ASRC, seeds17/29/43; evaluate_three_seed.py through the six-GPU queue, aggregate_three_seed.py; see per-job BEFORE/AFTER/PID/GPU records.\n')
    f.write('Results: 8/8 new tests completed, four locked seed17 tests reused. All 45 raw KG rank files and 12 result files verified; seed mean/sample SD recomputed. '+r['verdict']+'; '+str(r['strict_wins'])+'/12 strict wins.\n')
    f.write('Metrics: '+json.dumps({ds:{"mean":d['mean'],"sample_sd":d['sample_sd']} for ds,d in r['datasets'].items()})+'\n')
    f.write('Artifacts: three_seed/RESULTS.json, COMPARISON.csv, freeze.json, evaluation/ and EXPERIMENTS_COMPLETED.json.\n')
    f.write('Conclusion: all authorized experiments complete and no campaign process remains. Unknown/proxy baselines remain; no complete comparable SOTA claim. Next: update only ASRC Table2 cells to mean±sample SD, compile main.tex, sync raw outputs/documents and verify hashes, then stop.\n')
print(json.dumps({'completed':8,'reused':4,'remaining_experiments':0,'verdict':r['verdict']},ensure_ascii=False))
