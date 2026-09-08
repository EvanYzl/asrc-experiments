"""Preserve exact adapter and released-source bytes under content hashes."""
import hashlib
import json
import subprocess
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'


def main():
    objects=RUN/'code_objects';objects.mkdir(exist_ok=True)
    paths=list((ROOT/'reproduction/strict_baselines').glob('*.py'))
    paths.extend((ROOT/'reproduction/tools').glob('*.py'))
    for method in ['ATransN','KEnS','SS-AGA','LSMGA','DMKGC','IMKGC','AlignKGC']:
        paths.extend((ROOT/'reproduction/sources'/method).rglob('*.py'))
    entries=[]
    for path in sorted(paths):
        if '__pycache__' in path.parts:continue
        content=path.read_bytes();digest=hashlib.sha256(content).hexdigest();artifact=objects/(digest+'.py')
        if not artifact.exists():artifact.write_bytes(content)
        entries.append({'source':str(path.relative_to(ROOT)),'sha256':digest,'object':artifact.name,'bytes':len(content)})
    payload={'algorithm':'SHA-256','files':entries,'purpose':'content-addressed source snapshots; code objects are never overwritten'}
    (RUN/'code_snapshot_manifest.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    for name,python in [('pytorch','C:/Users/evan/.conda/envs/ai4/python.exe'),('pytorch_with_pyg_overlay','C:/Users/evan/.conda/envs/ai4/python.exe'),('tensorflow',str(ROOT/'reproduction/envs/kens-tf210/Scripts/python.exe'))]:
        path=RUN/f'{name}_requirements_snapshot.txt'
        if not path.exists():
            env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'reproduction/envs/pyg210_27') if name=='pytorch_with_pyg_overlay' else ''
            result=subprocess.run([python,'-c',"import importlib.metadata as m; names=sorted({d.metadata['Name'] for d in m.distributions() if d.metadata['Name']}); print('\\n'.join(name+'=='+m.version(name) for name in names))"],capture_output=True,text=True,check=True,env=env)
            path.write_text(result.stdout,encoding='utf-8')
    print('Archived source files:',len(entries))


if __name__=='__main__':main()
