"""Authenticate in memory, deploy registered jobs, return/verify outputs, resume PDF delivery."""
from pathlib import Path, PurePosixPath
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import paramiko
import psutil

ROOT=Path('G:/zhishitupui')
SIDE=ROOT/'reproduction/language_table_seed17'
HERE=Path(__file__).resolve().parent
REMOTE='/root/zhishitupui/reproduction/language_table_seed17_remote_20260907'
JOBS=['atransn_en_s17','imkgc_depkg_s17']
STATUS=HERE/'TRANSFER_STATE.json'

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def note(message):print(now()+' '+message,flush=True)
def safe_extract(archive,dest):
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            p=PurePosixPath(m.name)
            assert not p.is_absolute() and '..' not in p.parts and m.isfile(),m.name
            target=dest.joinpath(*p.parts);target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as src,target.open('wb') as out:shutil.copyfileobj(src,out)

class Connection:
    def __init__(self,password):self.password=password;self.client=None;self.sftp=None
    def connect(self):
        if self.client is not None:
            try:self.client.close()
            except Exception:pass
        self.client=paramiko.SSHClient()
        self.client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
        self.client.set_missing_host_key_policy(paramiko.RejectPolicy())
        self.client.connect('223.109.239.36',port=10316,username='root',password=self.password,
                            look_for_keys=False,allow_agent=False,timeout=20,auth_timeout=20,banner_timeout=20)
        self.client.get_transport().set_keepalive(30);self.sftp=self.client.open_sftp()
    def command(self,command):
        prefix='export PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH='+shlex.quote(REMOTE+'/dependencies')+'${PYTHONPATH:+:$PYTHONPATH}; '
        stdin,stdout,stderr=self.client.exec_command('bash -lc '+shlex.quote(prefix+command),timeout=60)
        stdin.close();out=stdout.read().decode('utf-8',errors='replace');err=stderr.read().decode('utf-8',errors='replace')
        code=stdout.channel.recv_exit_status()
        if code:raise RuntimeError(f'Remote command exited {code}: {out[-2000:]} {err[-2000:]}')
        return out
    def read(self,path):
        with self.sftp.open(path) as f:return json.loads(f.read().decode('utf-8'))
    def upload(self,local,remote):
        self.sftp.put(str(local),remote+'.uploading')
        digest=self.command('sha256sum '+shlex.quote(remote+'.uploading')).split()[0]
        assert digest==sha(local),'Upload digest mismatch'
        self.command('mv '+shlex.quote(remote+'.uploading')+' '+shlex.quote(remote))
    def launch(self,name):
        python="import subprocess,json;from pathlib import Path;b=Path("+repr(REMOTE)+");(b/'logs').mkdir(exist_ok=True);f=(b/'logs'/"+repr(name+'_controller.log')+").open('ab');p=subprocess.Popen(['python',str(b/'server_job.py'),"+repr(name)+"],cwd=b,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'controller_pid':p.pid}))"
        return json.loads(self.command('source /root/zhishitupui/activate.sh && python -c '+shlex.quote(python)))

def verify_return(name,folder):
    files=read(folder/'RETURN_HASHES.json')
    for rel,digest in files.items():assert sha(folder/rel)==digest,(name,rel)
    sys.path.insert(0,str(SIDE));import publish
    dataset='depkg' if name=='imkgc_depkg_s17' else 'wk3l'
    method='IMKGC' if name=='imkgc_depkg_s17' else 'ATransN'
    result,values,records=publish.accepted(folder/'result.json',dataset,method)
    assert set(values)==({'de','es','fr','it','jp','uk'} if dataset=='depkg' else {'en'})
    for metric in publish.METRICS:
        assert abs(sum(v[metric] for v in values.values())/len(values)-result['macro'][metric])<1e-12
    assert sha(folder/'best.pt')==result['checkpoint_sha256']==read(folder/'FINAL_EVALUATION_FREEZE.json')['checkpoint_sha256']
    return {'status':'passed','files':len(files),'rank_queries':sum(v['n'] for v in values.values()),'checkpoint_sha256':result['checkpoint_sha256'],'raw_rank_metrics_verified':True}

def main():
    password=sys.stdin.readline().rstrip('\r\n')
    assert password,'SSH authentication input was not supplied'
    connection=Connection(password);password=None
    connection.connect()
    state=read(STATUS) if STATUS.exists() else {'started_at':now(),'server':REMOTE,'jobs':{}}
    state.update(status='deploying',pid=os.getpid(),updated_at=now());state.pop('error',None)
    save(STATUS,state)
    connection.command('mkdir -p '+shlex.quote(REMOTE))
    try:installed=connection.read(REMOTE+'/BUNDLE_HASHES.json')
    except FileNotFoundError:installed=None
    if installed!=read(HERE/'bundle/BUNDLE_HASHES.json'):
        assert not state['jobs'],'Runtime update requires reviewing already launched jobs'
        connection.upload(HERE/'runtime_bundle.tar.gz',REMOTE+'/runtime_bundle.tar.gz')
        connection.command('tar -xzf '+shlex.quote(REMOTE+'/runtime_bundle.tar.gz')+' -C '+shlex.quote(REMOTE))
    preflight=connection.command('source /root/zhishitupui/activate.sh && python '+shlex.quote(REMOTE+'/server_preflight.py'))
    state['preflight']=json.loads(preflight);save(STATUS,state);note('Remote runtime, raw data and teacher hashes verified.')
    if 'atransn_en_s17' not in state['jobs']:
        state['jobs']['atransn_en_s17']={'status':'launched',**connection.launch('atransn_en_s17')};save(STATUS,state)
    note('ATransN FR-to-EN launched on server GPU 0.')
    while not (HERE/'checkpoint_bundle.READY.json').exists():
        state.update(status='waiting_for_local_checkpoint',updated_at=now());save(STATUS,state);time.sleep(10)
    checkpoint=read(HERE/'checkpoint_bundle.READY.json')
    local=Path(checkpoint['archive']);assert sha(local)==checkpoint['sha256']
    if 'imkgc_depkg_s17' not in state['jobs']:
        connection.upload(local,REMOTE+'/checkpoint_bundle.tar.gz')
        connection.command('mkdir -p '+shlex.quote(REMOTE+'/jobs/imkgc_depkg_s17')+' && tar -xzf '+shlex.quote(REMOTE+'/checkpoint_bundle.tar.gz')+' -C '+shlex.quote(REMOTE+'/jobs/imkgc_depkg_s17'))
        preflight=connection.command('source /root/zhishitupui/activate.sh && python '+shlex.quote(REMOTE+'/server_preflight.py')+' --checkpoint')
        state['checkpoint_preflight']=json.loads(preflight);state['jobs']['imkgc_depkg_s17']={'status':'launched',**connection.launch('imkgc_depkg_s17')}
    state.update(status='remote_training',updated_at=now());save(STATUS,state)
    note('IMKGC/E-PKG full-state continuation launched on server GPU 1.')
    returned=set()
    while len(returned)<len(JOBS):
        try:
            for name in JOBS:
                if name in returned:continue
                try:remote=connection.read(REMOTE+'/states/'+name+'.json')
                except FileNotFoundError:continue
                state['jobs'][name]['server_state']=remote;state['updated_at']=now();save(STATUS,state)
                if remote['status']=='failed':raise RuntimeError('Registered server training failed: '+name)
                if remote['status']!='ready_to_return':continue
                (HERE/'returns').mkdir(exist_ok=True)
                archive=HERE/'returns'/f'{name}.tar.gz'
                connection.sftp.get(remote['archive'],str(archive)+'.downloading')
                assert sha(Path(str(archive)+'.downloading'))==remote['archive_sha256']
                os.replace(str(archive)+'.downloading',archive)
                dest=HERE/'returned_jobs'/name;safe_extract(archive,dest)
                accepted=verify_return(name,dest);save(HERE/'returns'/f'{name}.verification.json',accepted)
                state['jobs'][name].update(status='returned_verified',verification=accepted,local_output=str(dest));save(STATUS,state)
                returned.add(name);note(name+' returned; all archive files, test ranks, metrics and checkpoint hashes verified.')
            if len(returned)<len(JOBS):time.sleep(30)
        except (OSError,paramiko.SSHException,EOFError) as e:
            note('SSH/SFTP connection interrupted; retrying authentication without writing credentials.')
            time.sleep(15);connection.connect()
    try:connection.sftp.get('/root/zhishitupui/SOTA_LOG.md',str(HERE/'SERVER_SOTA_LOG.md'))
    except OSError:pass
    # The handoff owns only the two named outputs; both local schedulers must have stopped.
    state['status']='waiting_for_local_owners';save(STATUS,state)
    while psutil.pid_exists(89940) or psutil.pid_exists(75848) or psutil.pid_exists(106428):time.sleep(10)
    parentdir=ROOT/'reproduction/runs/strict_baselines_20260905'
    for name in JOBS:
        dest=(parentdir/'jobs'/name) if name=='imkgc_depkg_s17' else SIDE/'jobs'/name
        if dest.exists():shutil.copytree(dest,HERE/'before_result_import'/name)
        shutil.copytree(HERE/'returned_jobs'/name,dest,dirs_exist_ok=True)
    parent=read(parentdir/'queue_state.json');row=parent['jobs']['imkgc_depkg_s17']
    row.update(status='completed',exit_code=0,finished=time.time(),result=str(parentdir/'jobs/imkgc_depkg_s17/result.json'),
               completion_evidence='Full-state server continuation; returned hashes and raw ranks verified',migration=str(HERE/'TRANSFER_STATE.json'),server_pid=state['jobs']['imkgc_depkg_s17']['server_state']['pid'])
    active=read(parentdir/'manifest.json')['jobs']
    assert all(parent['jobs'][j['id']]['status']=='completed' for j in active)
    parent.update(status='complete',current_job=None,current_jobs=[],updated_at=time.time());save(parentdir/'queue_state.json',parent)
    local=read(SIDE/'state.json');row=local['jobs']['atransn_en_s17']
    row.update(status='completed',phase='formal_server',exit_code=0,result=str(SIDE/'jobs/atransn_en_s17/result.json'),finished_at=now(),
               server_pid=state['jobs']['atransn_en_s17']['server_state']['pid'],migration=str(HERE/'TRANSFER_STATE.json'))
    assert all(v['status']=='completed' for v in local['jobs'].values())
    local.update(status='completed',current_job=None,updated_at=now());save(SIDE/'state.json',local)
    finisher=psutil.Process(104496)
    assert any(x.replace('\\','/').endswith('/language_table_seed17/finish_when_ready.py') for x in finisher.cmdline())
    finisher.resume();state.update(status='compiling_table',updated_at=now());save(STATUS,state)
    note('Accepted results installed; original finisher resumed for 378-cell import, PDF compilation and verification.')
    while True:
        final=read(SIDE/'finish_state.json')
        if final['status']=='completed':
            state.update(status='completed',finished_at=now(),delivery=final['delivery']);save(STATUS,state);note('Full language table delivered and verified. This migration has no further jobs.');return 0
        if final['status']=='needs_attention':raise RuntimeError('PDF completion needs attention: '+json.dumps(final))
        time.sleep(15)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        previous=read(STATUS) if STATUS.exists() else {}
        save(STATUS,{**previous,'status':'needs_attention','error':str(exc),'updated_at':now()})
        note('Transfer worker stopped for inspection: '+str(exc));raise
