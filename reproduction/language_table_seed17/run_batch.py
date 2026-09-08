"""Bounded, isolated completion queue; no changes to the main experiment queue."""
from pathlib import Path
import datetime as dt
import json
import os
import subprocess
import time

BASE = Path(__file__).resolve().parent


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def log(event):
    with (BASE / 'TRAINING_LOG.md').open('a', encoding='utf-8') as f:
        f.write('\n\n```json\n' + json.dumps({'time': now(), **event}, ensure_ascii=False, indent=2) + '\n```\n')


def free_memory():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return int(r.stdout.splitlines()[0].strip())


def main():
    manifest = json.loads((BASE / 'manifest.json').read_text(encoding='utf-8'))
    state_path = BASE / 'state.json'
    assert not state_path.exists(), 'Inspect an existing batch before starting another scheduler'
    state = {'status':'running','scheduler_pid':os.getpid(),'started_at':now(), 'jobs':{j['id']:{'status':'pending'} for j in manifest['jobs']}}
    (BASE / 'logs').mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', LANG_TABLE_GPU_FRACTION='0.20')
    save(state_path, state)
    for job in manifest['jobs']:
        rec = state['jobs'][job['id']]
        if any(state['jobs'][dep]['status'] != 'completed' for dep in job['dependencies']):
            rec.update(status='blocked',reason='A required teacher did not complete')
            save(state_path,state)
            continue
        if job['kind'] == 'graph':
            while not (BASE / 'GRAPH_ADAPTER_READY.json').exists():
                state.update(status='awaiting_graph_adapter_check',updated_at=now())
                save(state_path,state)
                time.sleep(15)
        phases = ['smoke','formal'] if job['kind'] == 'graph' else ['formal']
        for phase in phases:
            while free_memory() < manifest['resources']['min_free_mib']:
                state.update(status='waiting_for_memory',updated_at=now(),current_job=job['id'])
                save(state_path,state)
                time.sleep(15)
            command = job['command'].copy()
            output = Path(job['output'])
            if phase == 'smoke':
                output = BASE / 'smoke' / job['id']
                command[command.index('--output')+1] = str(output)
                command.append('--smoke')
            output.mkdir(parents=True,exist_ok=True)
            logfile = BASE / 'logs' / (job['id'] + '_' + phase + '.log')
            event = {'event':'before_training','purpose':manifest['purpose'],'changes':manifest['selection'],
                     'job':job['id'],'phase':phase,'seed':17,'command':command,'gpu':0,
                     'resources':manifest['resources'],'result_path':str(output/'result.json'),'log':str(logfile)}
            log(event)
            with logfile.open('wb') as stream:
                proc = subprocess.Popen(command, cwd=BASE, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                rec.update(status='running',phase=phase,pid=proc.pid,started_at=now(),output=str(output),log=str(logfile),gpu=0)
                state.update(status='running',current_job=job['id'],updated_at=now())
                save(state_path,state)
                log({'event':'process_started','job':job['id'],'phase':phase,'pid':proc.pid,'gpu':0})
                while proc.poll() is None:
                    time.sleep(10)
                    state['updated_at']=now()
                    save(state_path,state)
                code = proc.returncode
            result_path=output/'result.json'
            result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
            ok=code==0 and result is not None and result.get('status')=='completed'
            if phase=='formal': ok=ok and result.get('full_data') is True and result.get('seed')==17
            rec.update(status='completed' if ok else 'failed',exit_code=code,finished_at=now())
            if phase=='smoke' and ok: rec['smoke_result']=str(result_path)
            if phase=='formal' and ok: rec['result']=str(result_path)
            log({'event':'after_training','job':job['id'],'phase':phase,'pid':proc.pid,'gpu':0,'exit_code':code,
                 'result_path':str(result_path),'conclusion':'completed; preserved without test-based reselection' if ok else 'failed; inspect saved log; do not change recipe based on test',
                 'next':'next registered phase or job' if ok else 'preserve failure and continue independent jobs'})
            state['updated_at']=now()
            save(state_path,state)
            if not ok: break
            time.sleep(3)
    state.update(status='completed' if all(r['status']=='completed' for r in state['jobs'].values()) else 'needs_attention',finished_at=now(),current_job=None)
    save(state_path,state)
    log({'event':'batch_finished','status':state['status'],'next':'verify raw ranks, import the existing E-PKG run, fill the standalone table and compile'})


if __name__ == '__main__':
    main()
