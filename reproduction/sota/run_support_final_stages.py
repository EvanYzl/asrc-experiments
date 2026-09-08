"""Wait for all training, freeze, test once, and then profile strictly serially."""
import datetime
import json
import os
import shlex
import subprocess
import sys
import time
from frozen_data import ROOT,atomic_json
from run_sota_queue import append_log
base=ROOT/'reproduction/sota';phase=base/'paper_support'
while not (phase/'TRAINING_COMPLETED.json').exists():
    for name in ['wave1','smoke_queue','wave2']:
        path=phase/name/'state.json'
        if path.exists():assert not any(v['status']=='failed' for v in json.loads(path.read_text())['jobs'].values()),name+' has failures'
    time.sleep(10)
subprocess.run([sys.executable,str(base/'prepare_support_evaluation.py')],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(base/'run_sota_queue.py'),'--manifest',str(phase/'test_queue/manifest.json')],cwd=ROOT,check=True)
state=json.loads((phase/'test_queue/state.json').read_text());assert len(state['jobs'])==36 and all(v['status']=='completed' for v in state['jobs'].values())
subprocess.run([sys.executable,str(base/'aggregate_support.py')],cwd=ROOT,check=True)
freeze=json.loads((phase/'EVALUATION_FREEZE.json').read_text());state={'jobs':{},'scheduler_pid':os.getpid()};statepath=phase/'profiling/state.json'
for ds in ['dbp5l','depkg','dwy','wk3l']:
    for mode in (['independent'] if ds=='depkg' else ['independent','shared']):
        for seed in [17,29,43]:
            ident=f'{ds}_{mode}_s{seed}';out=phase/'profiling'/ident;out.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'reproduction/sota/profile_support.py','--id',ident]
            append_log(f'## BEFORE profile_{ident} — {datetime.datetime.now(datetime.timezone.utc).isoformat()}\nPurpose: isolated fixed-validation-query ranking cost for Table8. Changes/config: EVALUATION_FREEZE.json profiling spec; batch1 latency/batch256 throughput, 3 warmups/10 measured passes, FP32.\nSeed: {seed}; GPU: 0, exclusively; command: `{shlex.join(command)}`; PID: pending.\nOutput: {out.relative_to(ROOT).as_posix()}. Conclusion: all registered training and tests completed. Next: one isolated timing job, then audit raw timings.')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            with (out/'stdout.log').open('a') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['jobs'][ident]={'status':'running','pid':proc.pid,'gpu':0,'seed':seed,'command':command,'output':out.relative_to(ROOT).as_posix()};atomic_json(statepath,state)
                append_log(f'Launch receipt profile_{ident}: PID {proc.pid}, GPU 0, seed {seed}, log {out.relative_to(ROOT).as_posix()}/stdout.log.')
                ret=proc.wait()
            ok=ret==0 and (out/'result.json').exists();result=json.loads((out/'result.json').read_text()) if ok else {}
            state['jobs'][ident].update(status='completed' if ok else 'failed',returncode=ret,result=result);atomic_json(statepath,state)
            append_log(f'## AFTER profile_{ident} — {datetime.datetime.now(datetime.timezone.utc).isoformat()}\nPurpose: Table8 isolated timing. Config/seed/command: as BEFORE. PID: {proc.pid}; GPU: 0; return code: {ret}.\nResult: {out.relative_to(ROOT).as_posix()}/result.json; metrics: {json.dumps(result.get("macro"))}.\nConclusion: {"completed fixed validation-query timing" if ok else "failed; preserve output and inspect"}. Next: continue only registered profiling jobs, then compile and verify documents.')
            assert ok and result['status']=='completed'
state['finished']=time.time();atomic_json(statepath,state)
subprocess.run([sys.executable,str(base/'aggregate_support.py'),'--require-profiles'],cwd=ROOT,check=True)
atomic_json(phase/'EXPERIMENTS_COMPLETED.json',{'status':'completed','new_training':34,'new_test_runs':36,'profiles':21,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'further_model_optimization':False})
append_log('## Supporting experiments completed\nAll 48 registered models have audited raw test results (12 locked tests reused); 21 distinct timing runs completed. No further experiment is scheduled. Next: fill Tables4–8, compile idea and tables, return all raw artifacts and verify both hosts.')
