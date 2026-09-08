"""Deploy and start the four already-frozen graph repeats on currently idle GPUs."""
from pathlib import Path
import datetime as dt
import getpass
import hashlib
import json
import os
import shlex
import shutil
import tarfile
import paramiko

ROOT=Path('G:/zhishitupui');SIDE=ROOT/'reproduction/language_table_seed17';BASE=Path(__file__).resolve().parent
REMOTE='/root/zhishitupui/reproduction/main_tables_completion_20260907'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def main():
    stage=BASE/'early_bundle';stage.mkdir(exist_ok=False)
    for name in ['common.py','run_graph_sota.py','wk3l_graph_data.py']:
        p=SIDE/'runtime'/name;text=p.read_text(encoding='utf-8')
        if name=='common.py':
            text=text.replace("ROOT = Path('G:/zhishitupui')","ROOT = Path('/root/zhishitupui')")
            text=text.replace('BASE = Path(__file__).resolve().parent.parent',"BASE = Path('"+REMOTE+"')")
            text=text.replace('files = {str(p.relative_to(ROOT)): sha256(p)',"files = {str(p.relative_to(ROOT)).replace('/', chr(92)): sha256(p)")
        if name=='run_graph_sota.py':text=text.replace("sys.path.insert(0, 'G:/zhishitupui/reproduction/envs/pyg210_27')",'# Use validated server dependencies.')
        target=stage/'runtime_graph'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text,encoding='utf-8')
    for folder in ['data_manifests','graph_data']:
        shutil.copytree(SIDE/'runtime'/folder,stage/'runtime_graph'/folder)
    for method in ['LSMGA','DMKGC','IMKGC']:
        source=SIDE/'sources'/method
        for p in source.rglob('*.py'):copy(p,stage/'sources'/method/p.relative_to(source))
    jobs=[]
    for gpu,method,seed in [(2,'LSMGA',29),(3,'LSMGA',43),(4,'DMKGC',29),(5,'DMKGC',43)]:
        name=f'{method.lower()}_wk3l_s{seed}'
        jobs.append({'id':name,'method':method,'dataset':'wk3l','seed':seed,'gpu':gpu,'output':'jobs/'+name})
    plan={'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'authorization':'User explicitly requested all six server GPUs be occupied while completing both PDFs','jobs':jobs,
          'seeds':[17,29,43],'seed17_reused':True,'selection':'FR val_select only; one final frozen test of both KGs; fixed original recipe',
          'model_source_sha256':{p.relative_to(stage).as_posix():sha(p) for p in (stage/'sources').rglob('*.py')}}
    (stage/'EARLY_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    copy(BASE/'early_job.py',stage/'early_job.py')
    hashes={p.relative_to(stage).as_posix():sha(p) for p in stage.rglob('*') if p.is_file()}
    (stage/'EARLY_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    archive=BASE/'early_graphs.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in sorted(stage.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(stage).as_posix(),recursive=False)
    print('Prepared four fixed graph repeats; archive bytes:',archive.stat().st_size,flush=True)
    password=getpass.getpass('SSH/SFTP authentication password: ')
    c=paramiko.SSHClient();c.load_host_keys(str(Path.home()/'.ssh/known_hosts'));c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('223.109.239.36',port=10316,username='root',password=password,look_for_keys=False,allow_agent=False,timeout=20);password=None
    def command(s):
        i,o,e=c.exec_command('bash -lc '+shlex.quote(s));i.close();out=o.read().decode();error=e.read().decode();code=o.channel.recv_exit_status();assert code==0,(out,error);return out
    command('mkdir -p '+shlex.quote(REMOTE))
    s=c.open_sftp();s.put(str(archive),REMOTE+'/early_graphs.tar.gz')
    assert command('sha256sum '+shlex.quote(REMOTE+'/early_graphs.tar.gz')).split()[0]==sha(archive)
    command('tar -xzf '+shlex.quote(REMOTE+'/early_graphs.tar.gz')+' -C '+shlex.quote(REMOTE))
    launched=[]
    for job in jobs:
        code="from pathlib import Path;import subprocess,json;b=Path("+repr(REMOTE)+");(b/'logs').mkdir(exist_ok=True);f=(b/'logs'/"+repr(job['id']+'_controller.log')+").open('ab');p=subprocess.Popen(['python',str(b/'early_job.py'),"+repr(job['id'])+"],cwd=b,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'controller_pid':p.pid}))"
        result=json.loads(command('source /root/zhishitupui/activate.sh && python -c '+shlex.quote(code)));launched.append({**job,**result})
    (BASE/'EARLY_LAUNCH.json').write_text(json.dumps({'timestamp':dt.datetime.now(dt.timezone.utc).isoformat(),'server':REMOTE,'jobs':launched},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'launched':launched}),flush=True);s.close();c.close()

if __name__=='__main__':main()
