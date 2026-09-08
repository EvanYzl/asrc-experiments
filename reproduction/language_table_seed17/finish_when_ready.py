"""Complete the registered dependency-path retry, verify results and publish once."""
from pathlib import Path
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
import psutil
from run_batch import log, save, free_memory

BASE=Path(__file__).resolve().parent
TABLE=Path('G:/zhishitupui/outputs/kbs/_main/_tables/language_breakdown_seed17')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def run():
    receipt=BASE/'finish_state.json'
    save(receipt,{'status':'waiting_for_training','pid':os.getpid(),'started_at':dt.datetime.now(dt.timezone.utc).isoformat()})
    while True:
        state=read(BASE/'state.json')
        if state['status'] in ['completed','needs_attention'] and not psutil.pid_exists(state['scheduler_pid']):break
        time.sleep(20)
    failed=[name for name,r in state['jobs'].items() if r['status']!='completed']
    if failed==['atransn_en_s17']:
        record=state['jobs'][failed[0]]
        assert "No module named 'numba'" in Path(record['log']).read_text(encoding='utf-8',errors='replace')
        assert not (BASE/'jobs/atransn_en_s17/best.pt').exists(), 'Inspect any prior training before retrying'
        shutil.copy2(BASE/'state.json',BASE/'state.before_transfer_dependency_retry.json')
        job=next(j for j in read(BASE/'manifest.json')['jobs'] if j['id']==failed[0])
        while free_memory()<5000:time.sleep(20)
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LANG_TABLE_GPU_FRACTION='0.20')
        logfile=BASE/'logs/atransn_en_s17_formal_retry1.log'
        log({'event':'before_training_retry','job':job['id'],'purpose':'Complete the already registered reverse transfer','change':'Reuse existing numba dependency path; first attempt failed before training','seed':17,'command':job['command'],'gpu':0,'result_path':job['output']+'/result.json'})
        with logfile.open('wb') as stream:
            proc=subprocess.Popen(job['command'],cwd=BASE,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            record.update(status='running',phase='formal_retry1',pid=proc.pid,log=str(logfile),started_at=dt.datetime.now(dt.timezone.utc).isoformat())
            state.update(status='running',current_job=job['id'],retry_owner_pid=os.getpid())
            save(BASE/'state.json',state);save(receipt,{'status':'retrying_transfer','pid':os.getpid(),'training_pid':proc.pid})
            code=proc.wait()
        result=Path(job['output'])/'result.json'
        okay=code==0 and result.exists() and read(result).get('status')=='completed' and read(result).get('full_data') is True
        record.update(status='completed' if okay else 'failed',exit_code=code,finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        if okay:record['result']=str(result)
        state.update(status='completed' if okay else 'needs_attention',current_job=None)
        save(BASE/'state.json',state)
        log({'event':'after_training_retry','job':job['id'],'pid':proc.pid,'gpu':0,'exit_code':code,'result_path':str(result),'conclusion':'completed' if okay else 'needs inspection','next':'import table when all registered results exist'})
    if state['status']!='completed':
        save(receipt,{'status':'needs_attention','reason':'One or more registered training jobs failed','jobs':{n:r['status'] for n,r in state['jobs'].items()}});return 1
    save(receipt,{'status':'waiting_for_existing_epkg_run','pid':os.getpid()})
    while True:
        r=subprocess.run([sys.executable,str(BASE/'publish.py')],cwd=BASE,capture_output=True,text=True)
        if r.returncode==75:time.sleep(30);continue
        (BASE/'publish.stdout.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
        if r.returncode:
            save(receipt,{'status':'needs_attention','reason':'Result import validation failed','log':str(BASE/'publish.stdout.log')});return r.returncode
        break
    plugin=Path('C:/Users/evan/.codex/plugins/cache/openai-bundled/latex/0.2.6')
    bundle='C:/Users/evan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    env=os.environ.copy();env['PATH']='G:/texlive/2026/bin/windows'+os.pathsep+env['PATH']
    command=[bundle,str(plugin/'scripts/compile_latex.py'),str(TABLE/'main.tex'),'--compiler','texlive','--engine','xelatex','--output-directory',str(TABLE/'build'),'--json']
    with (TABLE/'compile.log').open('wb') as f:r=subprocess.run(command,cwd=plugin,env=env,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:
        save(receipt,{'status':'needs_attention','reason':'LaTeX compilation failed'});return r.returncode
    r=subprocess.run(['G:/zhishitupui/work/.venv-ideaspark/Scripts/python.exe',str(BASE/'check_pdf.py')],cwd=BASE,capture_output=True,text=True)
    (BASE/'pdf_check.stdout.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
    if r.returncode:
        save(receipt,{'status':'needs_attention','reason':'PDF verification failed'});return r.returncode
    save(receipt,{'status':'completed','finished_at':dt.datetime.now(dt.timezone.utc).isoformat(),'delivery':read(BASE/'DELIVERY_VERIFICATION.json')})
    log({'event':'delivery_completed','numeric_cells':378,'missing_cells':0,'new_training_jobs':8,'next':'stop this batch and report the completed PDF; do not train additional candidates'})
    return 0


if __name__=='__main__':raise SystemExit(run())
