"""Sequential campaign stages, six independent GPU workers inside each stage."""
import datetime
import json
import subprocess
import sys
import time
from frozen_data import ROOT,atomic_json,sha256
from run_sota_queue import append_log
base=ROOT/'reproduction/sota';phase=base/'paper_support'
def record(message):append_log('## Supporting-training control — '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n'+message)
freeze=json.loads((phase/'TRAINING_SOURCE_FREEZE.json').read_text())
assert sha256(phase/'PLAN.json')==freeze['plan_sha256']
for path,digest in freeze['source_hashes'].items():assert sha256(ROOT/path)==digest
record('BEFORE input checks. Purpose: exact original clean behavior and immutable filters under registered perturbations. Command: python reproduction/sota/check_support_inputs.py. Seed: perturbation 20260906. GPU: none. Outputs: paper_support/INPUT_CHECKS.json. Next: pass checks before smoke training.')
subprocess.run([sys.executable,str(base/'check_support_inputs.py')],cwd=ROOT,check=True)
record('AFTER input checks: passed all 16 clean-equivalence/perturbation checks; no test parsed. Next: await wave1 and execute eight one-epoch validation-only smoke jobs.')
while True:
    state=json.loads((phase/'wave1/state.json').read_text())
    if 'finished' in state:break
    time.sleep(10)
assert len(state['jobs'])==10 and all(v['status']=='completed' for v in state['jobs'].values())
for stage,count in [('smoke_queue',8),('wave2',24)]:
    subprocess.run([sys.executable,str(base/'run_sota_queue.py'),'--manifest',str(phase/stage/'manifest.json')],cwd=ROOT,check=True)
    state=json.loads((phase/stage/'state.json').read_text())
    assert len(state['jobs'])==count and all(v['status']=='completed' for v in state['jobs'].values()),stage+' failed; inspect retained logs'
    if stage=='smoke_queue':
        for job in state['jobs'].values():
            output=ROOT/job['output'];assert json.loads((output/'config.json').read_text())['smoke_only'] is True
            r=json.loads((output/'result.json').read_text());assert r['completed_epochs']==1 and r['test_access'] is False
        record('Eight validation-only smoke checks passed with finite training and replayed validation ranks. They are excluded from paper statistics. Next: execute the 24 pre-registered full-budget variants, same six-GPU queue; no recipe selection.')
atomic_json(phase/'TRAINING_COMPLETED.json',{'status':'completed','new_training':34,'smoke_excluded':8,'test_access':False,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat()})
record('All 34 new validation-only trainings completed. Fourteen historical trainings remain reused. Next: freeze all checkpoints and source/input hashes before 36 new once-only tests; retain 12 locked original tests.')
