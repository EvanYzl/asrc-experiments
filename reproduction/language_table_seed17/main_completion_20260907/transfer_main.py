"""Deploy a finite six-GPU queue and automatically return its verified raw artifacts."""
from pathlib import Path, PurePosixPath
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import paramiko
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=Path('G:/zhishitupui')
REMOTE='/root/zhishitupui/reproduction/main_tables_completion_20260907'
STATE=BASE/'TRANSFER_STATE.json'
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def note(s):print(now()+' '+s,flush=True)
class Connection:
    def __init__(self,password):self.password=password;self.client=None
    def connect(self):
        for attempt in range(1,7):
            if self.client:
                try:self.client.close()
                except Exception:pass
            self.client=paramiko.SSHClient();self.client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
            self.client.set_missing_host_key_policy(paramiko.RejectPolicy())
            try:
                self.client.connect('223.109.239.36',port=10316,username='root',password=self.password,look_for_keys=False,allow_agent=False,timeout=30,auth_timeout=30,banner_timeout=45)
                self.client.get_transport().set_keepalive(15)
                self.sftp=paramiko.SFTPClient.from_transport(self.client.get_transport(),window_size=8*1024*1024)
                return
            except (paramiko.AuthenticationException,paramiko.BadHostKeyException):raise
            except (OSError,paramiko.SSHException,EOFError) as exc:
                note(f'SSH connection attempt {attempt}/6 failed ({type(exc).__name__}); retrying transient connection failure.')
                if attempt==6:raise
                time.sleep(min(30,attempt*5))
    def command(self,code):
        i,o,e=self.client.exec_command('bash -lc '+shlex.quote('export PYTHONDONTWRITEBYTECODE=1; '+code),timeout=60)
        i.close();out=o.read().decode(errors='replace');error=e.read().decode(errors='replace');status=o.channel.recv_exit_status()
        if status:raise RuntimeError(f'Remote command failed ({status}): {out[-1500:]} {error[-1500:]}')
        return out
    def read(self,path):
        with self.sftp.open(path) as f:return json.loads(f.read().decode())
    def launch(self,script):
        code="from pathlib import Path;import subprocess,json;b=Path("+repr(REMOTE)+");f=(b/'logs'/"+repr(script+'.log')+").open('ab');p=subprocess.Popen(['python','-B',str(b/"+repr(script)+")],cwd=b,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'pid':p.pid}))"
        return json.loads(self.command('source /root/zhishitupui/activate.sh && python -c '+shlex.quote(code)))
    def download_atomic(self,remote,local):
        local.parent.mkdir(parents=True,exist_ok=True);tmp=local.with_suffix(local.suffix+'.downloading')
        if remote.endswith('.tar.gz'):
            total=self.sftp.stat(remote).st_size
            offset=tmp.stat().st_size if tmp.exists() else 0
            if offset>total:offset=0
            note(f'Returning {local.name}: resume at {offset}/{total} bytes.')
            with self.sftp.open(remote,'rb') as source,tmp.open('ab' if offset else 'wb') as target:
                while offset<total:
                    end=min(total,offset+2*1024*1024)
                    chunks=[(p,min(32768,end-p)) for p in range(offset,end,32768)]
                    block=b''.join(source.readv(chunks))
                    assert len(block)==end-offset
                    target.write(block);target.flush();offset=end
            assert tmp.stat().st_size==total
        else:self.sftp.get(remote,str(tmp))
        os.replace(tmp,local)
def extract(archive,dest):
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            p=PurePosixPath(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts
            target=dest.joinpath(*p.parts);target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
def verify(job,out):
    files=read(out/'RETURN_HASHES.json')
    for rel,digest in files.items():assert sha(out/rel)==digest,(job['id'],rel)
    result=read(out/'result.json');config=read(out/'config.json');freeze=read(out/'FINAL_EVALUATION_FREEZE.json')
    assert result['status']=='completed' and result['full_data'] is True and result['seed']==job['seed']
    assert config['seed']==job['seed'] and config['method']==job['method']
    assert sha(out/'best.pt')==freeze['checkpoint_sha256']==result['checkpoint_sha256']
    groups=[(job['kg'],out,result,'train')] if job['kind']=='ssaga' else [(kg,out/kg,value,'all') for kg,value in result['per_kg'].items()]
    counts={}
    for kg,folder,group,primary in groups:
        a=np.load(folder/'test_queries.npz',allow_pickle=False);r=a['rank_'+primary].astype(np.float64)
        manifest=read(ROOT/f'reproduction/strict_baselines/data_manifests/{job["dataset"]}_{kg}.json')
        assert len(r)==manifest['counts']['test'] and len(r)>64
        assert np.array_equal(a['query_index'],np.arange(len(r)))
        raw=(ROOT/f'data/raw/dmkgc/dataset{job["dataset"]}/kg/{kg}-test.tsv') if job['dataset']!='wk3l' else ROOT/'data/raw/atransn'/('WK3l-15k_FR' if kg=='fr' else 'WK3l-15k_EN_F')/'test_triple_id.txt'
        assert np.array_equal(a['triples'],np.loadtxt(raw,dtype=np.int64,ndmin=2))
        assert np.isfinite(r).all() and (r>=1).all()
        mm={'mrr':float((1/r).mean()),'h1':float((r<=1).mean()),'h10':float((r<=10).mean())}
        for key,v in mm.items():assert abs(v-group['metrics'][primary][key])<1e-12
        counts[kg]=len(r)
    return {'status':'passed','verified_at':now(),'files':len(files),'query_counts':counts,'checkpoint_sha256':sha(out/'best.pt')}
def sync_audit(connection):
    for name in ['QUEUE_STATE.json','TRAINING_LOG.md','PREFLIGHT.json','PREFLIGHT_FAILED.json','SCHEDULER_ERROR.json','EARLY_PLAN.json','EARLY_HASHES.json']:
        try:connection.download_atomic(REMOTE+'/'+name,BASE/'server_snapshot'/name)
        except FileNotFoundError:pass
    connection.download_atomic('/root/zhishitupui/SOTA_LOG.md',BASE/'server_snapshot/SOTA_LOG.md')
    # Preserve the sole graph-source repair without changing the original source.
    for name in ['versions/cuda_mapping_fix/REPAIR.json','versions/cuda_mapping_fix/LSMGA_run_model.py','sources/LSMGA/run_model.py']:
        try:connection.download_atomic(REMOTE+'/'+name,BASE/'server_snapshot'/name)
        except FileNotFoundError:pass
def main():
    password=sys.stdin.readline().rstrip('\r\n');assert password
    c=Connection(password);password=None;c.connect()
    plan=read(BASE/'PLAN.json');state=read(STATE) if STATE.exists() else {'started_at':now(),'jobs':{}}
    state.update(status='deploying',pid=os.getpid(),updated_at=now());save(STATE,state)
    if not state.get('deployed'):
        archive=BASE/'bundle.tar.gz';expected=sha(archive)
        note('Uploading additive SS-AGA features, frozen inputs and automatic queue; existing graph runs continue.')
        c.sftp.put(str(archive),REMOTE+'/main_bundle.tar.gz.uploading')
        assert c.command('sha256sum '+shlex.quote(REMOTE+'/main_bundle.tar.gz.uploading')).split()[0]==expected
        c.command('mv '+shlex.quote(REMOTE+'/main_bundle.tar.gz.uploading')+' '+shlex.quote(REMOTE+'/main_bundle.tar.gz')+' && tar -xzf '+shlex.quote(REMOTE+'/main_bundle.tar.gz')+' -C '+shlex.quote(REMOTE))
        state['deployed']={'time':now(),'archive_sha256':expected,'bytes':archive.stat().st_size};save(STATE,state)
    if not state.get('preflight'):
        state['preflight']=c.launch('preflight.py');save(STATE,state)
    if not state.get('scheduler'):
        state['scheduler']=c.launch('scheduler.py');save(STATE,state)
    note('Automatic six-GPU scheduler started; adopted existing repeats and enabled refill on idle GPUs.')
    last_sync=0
    while True:
        try:
            snapshot=c.read(REMOTE+'/QUEUE_STATE.json');state['remote_counts']=snapshot['counts'];state['gpu']=snapshot['gpu']
            state.update(status='remote_training',updated_at=now())
            if snapshot.get('status')=='needs_attention':state['status']='needs_attention'
            for job in plan['jobs']:
                if state['jobs'].get(job['id'],{}).get('status')=='returned_verified':continue
                remote=snapshot['jobs'][job['id']]
                if not remote.get('return_archive'):continue
                archive=BASE/'returns'/(job['id']+'.tar.gz')
                c.download_atomic(remote['return_archive'],archive);assert sha(archive)==remote['return_sha256']
                out=BASE/'returned_jobs'/job['id'];extract(archive,out);receipt=verify(job,out)
                save(BASE/'returns'/(job['id']+'.verification.json'),receipt)
                c.download_atomic(REMOTE+'/logs/'+job['id']+'.log',BASE/'returned_logs'/(job['id']+'.log'))
                state['jobs'][job['id']]={'status':'returned_verified','output':str(out),'verification':receipt,'archive_sha256':remote['return_sha256']}
                save(STATE,state);note(job['id']+' returned; frozen test triples, ranks, metrics and file hashes verified.')
            if time.time()-last_sync>=300:sync_audit(c);last_sync=time.time()
            save(STATE,state)
            if len(state['jobs'])==len(plan['jobs']) and all(j['status']=='returned_verified' for j in state['jobs'].values()):break
            time.sleep(30)
        except (OSError,paramiko.SSHException,EOFError):
            note('Connection interrupted or remote state not ready; reconnecting with in-memory authentication.');time.sleep(15);c.connect()
    state.update(status='ready_for_main_pdf',updated_at=now());save(STATE,state);sync_audit(c)
    publisher=BASE/'publish_main.py'
    if publisher.exists():
        subprocess.run([sys.executable,'-B',str(publisher),'--wait-for-reused-results'],check=True)
        state.update(status='completed',finished_at=now(),delivery=read(BASE/'MAIN_DELIVERY_VERIFICATION.json'));save(STATE,state)
    note('All 39 registered jobs returned and verified; no further experiments are scheduled.')
if __name__=='__main__':
    try:main()
    except Exception:
        state=read(STATE) if STATE.exists() else {};state.update(status='needs_attention',updated_at=now(),error=traceback.format_exc());save(STATE,state);raise
