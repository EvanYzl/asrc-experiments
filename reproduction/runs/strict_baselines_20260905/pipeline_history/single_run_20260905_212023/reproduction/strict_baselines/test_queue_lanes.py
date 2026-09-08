"""Exercise simultaneous CPU/GPU work, live-process adoption and scope boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from run_queue_lanes import process_matches
import psutil

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'


def wait_for(path,predicate,seconds=30):
    deadline=time.time()+seconds
    while time.time()<deadline:
        if path.exists():
            obj=json.loads(path.read_text())
            if predicate(obj):return obj
        time.sleep(.1)
    raise AssertionError('Timed out: '+str(path))


def main():
    with tempfile.TemporaryDirectory(prefix='queue-lanes-',dir=RUN/'smoke') as temp:
        root=Path(temp);worker=root/'worker.py'
        worker.write_text("import json, pathlib, sys, time\np=pathlib.Path(sys.argv[1]);p.mkdir(exist_ok=True)\nwith (p/'attempts.txt').open('a') as f:f.write('start\\n')\ntime.sleep(float(sys.argv[2]))\n(p/'result.json').write_text(json.dumps({'status':'completed'}))\n")
        jobs=[]
        for name,resource,deps,seconds in [('cpu_a','cpu',[],5),('gpu_a','gpu',[],5),('gpu_b','gpu',['gpu_a'],.3)]:
            out=root/name
            jobs.append({'id':name,'resource':resource,'cwd':str(root),'command':[sys.executable,str(worker),str(out),str(seconds)],'result':str(out/'result.json'),'depends_on':deps})
        manifest=root/'manifest.json';manifest.write_text(json.dumps({'jobs':jobs,'deferred_jobs':[{'id':'must_not_run'}]}))
        cmd=[sys.executable,str(Path(__file__).with_name('run_queue_lanes.py')),'--manifest',str(manifest),'--poll-seconds','.1','--min-free-memory-gb','0']
        flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
        with (root/'scheduler.log').open('w') as log:
            first=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
            state=wait_for(root/'queue_state.json',lambda obj:len(obj.get('current_jobs',[]))==2)
            initial={name:state['jobs'][name]['pid'] for name in ['cpu_a','gpu_a']}
            for job in jobs[:2]:
                assert process_matches(job,state['jobs'][job['id']],psutil.Process(initial[job['id']]))
                bad=dict(job,command=job['command']+['unrelated'])
                assert not process_matches(bad,state['jobs'][job['id']],psutil.Process(initial[job['id']]))
            first.terminate();first.wait(timeout=10)
            second=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
            assert second.wait(timeout=30)==0,(root/'scheduler.log').read_text()
        final=json.loads((root/'queue_state.json').read_text())
        assert final['status']=='complete' and final['current_jobs']==[]
        assert set(final['jobs'])=={'cpu_a','gpu_a','gpu_b'}
        for job in jobs:
            assert final['jobs'][job['id']]['status']=='completed'
            assert (root/job['id']/'attempts.txt').read_text()=='start\n'
        assert all(final['jobs'][name]['pid']==pid and final['jobs'][name].get('adopted_by')==second.pid for name,pid in initial.items())
        report={'passed':True,'simultaneous_cpu_gpu':'passed','same_lane_serial':'passed',
                'live_children_adopted_without_restart':'passed','unrelated_pid_rejected':'passed','deferred_jobs_not_started':'passed'}
        (RUN/'results/queue_lanes_checks.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))


if __name__=='__main__':main()
