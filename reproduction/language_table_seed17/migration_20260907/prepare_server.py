"""Build an isolated, credential-free migration bundle from frozen local inputs."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import tarfile

ROOT=Path('G:/zhishitupui')
SIDE=ROOT/'reproduction/language_table_seed17'
HERE=Path(__file__).resolve().parent
STAGE=HERE/'bundle'
REMOTE='/root/zhishitupui/reproduction/language_table_seed17_remote_20260907'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def edited(src,dst,changes):
    text=src.read_text(encoding='utf-8')
    for old,new in changes:
        assert old in text,(src,old)
        text=text.replace(old,new)
    dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(text,encoding='utf-8')
    return {'source':str(src),'source_sha256':sha(src),'deployed_sha256':sha(dst),'replacements':changes}


def main():
    assert not STAGE.exists()
    STAGE.mkdir()
    changes=[]
    for name in ['run_atransn.py','run_kge.py']:
        src=SIDE/'runtime'/name
        changes.append(edited(src,STAGE/'runtime_transfer'/name,
            [("sys.path.insert(0, 'G:/zhishitupui/reproduction/envs/pyg210_27')","# Use the previously validated server environment.")] if name=='run_atransn.py' else []))
    changes.append(edited(SIDE/'runtime/common.py',STAGE/'runtime_transfer/common.py',
        [("ROOT = Path('G:/zhishitupui')","ROOT = Path('/root/zhishitupui')"),
         ("files = {str(p.relative_to(ROOT)): sha256(p)","files = {str(p.relative_to(ROOT)).replace('/', chr(92)): sha256(p)")]))
    changes.append(edited(ROOT/'reproduction/strict_baselines/common.py',STAGE/'runtime_epkg/common.py',[
        ("ROOT = Path(__file__).resolve().parents[2]","ROOT = Path('/root/zhishitupui')\nBASE = Path(__file__).resolve().parent.parent"),
        ("SUITE = ROOT / 'reproduction' / 'strict_baselines'","SUITE = Path(__file__).resolve().parent"),
        ("objects=ROOT/'reproduction/runs/strict_baselines_20260905/code_objects'","objects=BASE/'code_objects'"),
        ("files = {str(p.relative_to(ROOT)): sha256(p)","files = {str(p.relative_to(ROOT)).replace('/', chr(92)): sha256(p)")]))
    changes.append(edited(ROOT/'reproduction/strict_baselines/run_graph_baseline.py',STAGE/'runtime_epkg/run_graph_baseline.py',[
        ("source=ROOT/'reproduction/sources'/opts.method","source=BASE/'sources'/opts.method"),
        ("dm_path=ROOT/'reproduction/sources/DMKGC/src/utils.py'","dm_path=BASE/'sources/DMKGC/src/utils.py'"),
        ("        with torch.no_grad():\n            test=tester.evaluate_split('test',save=True)",
         "        atomic_json(out/'FINAL_EVALUATION_FREEZE.json', {'seed':opts.seed,'completed_rounds':rounds,'selected_evaluation':state['evaluation_index'],'checkpoint_sha256':sha256(out/'best.pt'),'selection':'val_select only','migration':'full-state continuation from local run'})\n        with torch.no_grad():\n            test=tester.evaluate_split('test',save=True)")]))
    for method in ['IMKGC','DMKGC','ATransN']:
        src=(SIDE/'sources'/method) if method=='ATransN' else ROOT/'reproduction/sources'/method
        for p in src.rglob('*.py'):copy(p,STAGE/'sources'/method/p.relative_to(src))
    for runtime in ['runtime_transfer','runtime_epkg']:
        for p in (ROOT/'reproduction/strict_baselines/data_manifests').glob('*.json'):
            copy(p,STAGE/runtime/'data_manifests'/p.name)
    for name in ['atransn_recipe.json','fr_en_aligned_entity_id.txt']:
        copy(SIDE/name,STAGE/name)
    teacher=SIDE/'jobs/transe_fr_teacher_s17/fr'
    for name in ['checkpoint_valid.pt','config.json','result.json','FINAL_EVALUATION_FREEZE.json']:
        copy(teacher/name,STAGE/'teacher_fr'/name)
    cache=ROOT/'reproduction/strict_baselines/graph_data/train_only_k10_h2/datasetdepkg'
    for p in cache.rglob('*'):
        if p.is_file():copy(p,STAGE/'runtime_epkg/graph_data/train_only_k10_h2/datasetdepkg'/p.relative_to(cache))
    # The Windows-generated graph manifest uses platform-specific separators.
    manifest=STAGE/'runtime_epkg/graph_data/train_only_k10_h2/datasetdepkg/strict_graph_manifest.json'
    original=json.loads(manifest.read_text(encoding='utf-8'))
    original['source_files']={k.replace('\\','/'):v for k,v in original['source_files'].items()}
    manifest.write_text(json.dumps(original,indent=2)+'\n',encoding='utf-8')
    inputs={}
    for folder in [ROOT/'data/raw/dmkgc/datasetdepkg',ROOT/'data/raw/atransn/WK3l-15k_EN_F',ROOT/'data/raw/atransn/WK3l-15k_FR']:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ['.txt','.tsv']:
                inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    config=json.loads((ROOT/'reproduction/runs/strict_baselines_20260905/jobs/imkgc_depkg_s17/config.json').read_text(encoding='utf-8'))
    for rel,digest in config['source_hashes'].items():assert sha(ROOT/rel)==digest
    copy(ROOT/'reproduction/runs/strict_baselines_20260905/jobs/imkgc_depkg_s17/config.json',STAGE/'provenance/local_epkg_config.json')
    for name in ['server_job.py','server_preflight.py']:
        copy(HERE/name,STAGE/name)
    tasks={
        'atransn_en_s17':{'gpu':0,'seed':17,'method':'ATransN','dataset':'wk3l','output':'jobs/atransn_en_s17',
            'command':['python',REMOTE+'/runtime_transfer/run_atransn.py','--seed','17','--teacher',REMOTE+'/teacher_fr','--output',REMOTE+'/jobs/atransn_en_s17'],
            'purpose':'Complete the registered fixed-seed FR-to-EN transfer; original local import failure happened before training'},
        'imkgc_depkg_s17':{'gpu':1,'seed':17,'method':'IMKGC','dataset':'depkg','output':'jobs/imkgc_depkg_s17',
            'command':['python',REMOTE+'/runtime_epkg/run_graph_baseline.py','--method','IMKGC','--dataset','depkg','--seed','17','--output',REMOTE+'/jobs/imkgc_depkg_s17'],
            'purpose':'Continue the existing seed17 run from its full-state checkpoint to the original 50-round budget'}
    }
    deployment={'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'root':REMOTE,'jobs':tasks,'input_sha256':inputs,
        'path_and_recording_changes':changes,'teacher_sha256':sha(teacher/'checkpoint_valid.pt'),
        'selection':'validation only; no recipe change or test-based selection','resource_changes':'Independent server GPUs, existing validated Python/PyTorch/PyG environment; transfer process memory fraction 0.85',
        'completion':'Return raw artifacts with hashes; fill the standalone seed17 language table and compile; stop these two runs'}
    (STAGE/'DEPLOYMENT.json').write_text(json.dumps(deployment,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files={p.relative_to(STAGE).as_posix():sha(p) for p in STAGE.rglob('*') if p.is_file()}
    (STAGE/'BUNDLE_HASHES.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8')
    archive=HERE/'runtime_bundle.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in sorted(STAGE.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(STAGE).as_posix(),recursive=False)
    (HERE/'runtime_bundle.sha256').write_text(sha(archive)+'\n',encoding='utf-8')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files),'raw_input_files':len(inputs)}))


if __name__=='__main__':main()
