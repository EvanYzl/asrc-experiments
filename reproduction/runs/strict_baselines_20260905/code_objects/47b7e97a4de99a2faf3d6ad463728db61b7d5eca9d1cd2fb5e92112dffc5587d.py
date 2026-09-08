"""Released SS-AGA training with local features and validation-only selection."""
import argparse
import importlib
import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import *


def prepare(dataset):
    raw=ROOT/f'data/raw/dmkgc/dataset{dataset}'
    dest=SUITE/'ssaga_data'/f'dataset{dataset}'
    for sub in ['entity','kg','seed_alignlinks']:
        (dest/sub).mkdir(parents=True,exist_ok=True)
        for f in sorted((raw/sub).glob('*')):
            if f.is_file() and not (dest/sub/f.name).exists():shutil.copy2(f,dest/sub/f.name)
    if not (dest/'relations.txt').exists():shutil.copy2(raw/'relations.txt',dest/'relations.txt')
    features=dest/'entity_embeddings.npy'
    if not features.exists():
        if (raw/features.name).exists():
            shutil.copy2(raw/features.name,features)
            shutil.copy2(raw/'entity_embeddings.provenance.json',features.with_suffix('.provenance.json'))
        else:
            subprocess.run([sys.executable,str(ROOT/'reproduction/tools/build_ssaga_mbert_embeddings.py'),
                '--dataset',str(raw),'--output',str(features),'--batch-size','64'],check=True)
    provenance=json.loads(features.with_suffix('.provenance.json').read_text(encoding='utf-8'))
    provenance['source_text']=f'URL-decoded {dataset} entity URI labels; regenerated mBERT CLS features'
    assert np.load(features,mmap_mode='r').shape==(sum(sum(1 for _ in f.open(encoding='utf-8')) for f in (raw/'entity').glob('*.tsv')),768)
    provenance['feature_sha256']=sha256(features)
    atomic_json(dest/'feature_manifest.json',provenance)
    return dest


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['dbp5l','depkg'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--target');p.add_argument('--smoke',action='store_true');opts=p.parse_args()
    out=Path(opts.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if opts.target:run_target(opts,out);return
    per_kg={}
    for kg in DOMAINS[opts.dataset][:1] if opts.smoke else DOMAINS[opts.dataset]:
        result=out/kg/'result.json'
        if not result.exists():
            cmd=[sys.executable,__file__,'--dataset',opts.dataset,'--seed',str(opts.seed),'--output',str(out/kg),'--target',kg]
            if opts.smoke:cmd.append('--smoke')
            subprocess.run(cmd,check=True)
        per_kg[kg]=json.loads(result.read_text(encoding='utf-8'))
    atomic_json(out/'result.json',{'status':'completed','purpose':'smoke' if opts.smoke else 'formal','full_data':not opts.smoke,
        'method':'SS-AGA','dataset':opts.dataset,'seed':opts.seed,'domains':per_kg,
        'macro':macro_metrics({k:v['metrics']['train'] for k,v in per_kg.items()}),
        'protocol':'ssaga-regenerated-mbert-train-filter-v1','primary_filter':'train'})


def run_target(opts,out):
    seed_all(opts.seed);os.chdir(out)
    data_path=prepare(opts.dataset)
    source=ROOT/'reproduction/sources/SS-AGA';sys.path.insert(0,str(source))
    module=importlib.import_module('run_model')
    original_load=torch.load
    def trusted_load(*a,**kw):kw.setdefault('weights_only',False);return original_load(*a,**kw)
    torch.load=trusted_load
    args=module.parse_args(['--target_language','el','--dataset',opts.dataset,'--data_path',str(data_path.parent/'dataset'),
        '--seed',str(opts.seed),'--round','1' if opts.smoke else '25','--dim','256','--batch_size','200',
        '--micro_batch_size','200','--test_batch_size','32','--precompute_batch_size','64','--knn_batch_size','512',
        '--self_learning_device','cpu','--lr','0.005','--align_lr','0.001','--epoch10','1' if opts.smoke else '3',
        '--epoch11','1' if opts.smoke else '2','--epoch2','1' if opts.smoke else '2','--burin_in_epoch','15','--generation_freq','5'])
    args.target_language=opts.target
    if opts.smoke:args.MAX_SAM=64
    data=load_kg(opts.dataset,opts.target);instances=[]
    atomic_json(out/'config.json',{'method':'SS-AGA','dataset':opts.dataset,'kg':opts.target,'seed':opts.seed,'recipe':vars(args),
        'environment':version_info(),'source_hashes':source_hashes([Path(__file__),SUITE/'common.py',source/'run_model.py',*sorted(source.glob('src/*.py'))]),
        'dataset_hash':data['manifest']['dataset_hash'],'feature_manifest':str(data_path/'feature_manifest.json'),
        'features':'locally regenerated mBERT CLS on entity URI labels; not released descriptions',
        'selection':'val_select filtered tail MRR, earliest maximum; no test access',
        'final_filter':'train, current gold retained','supporter_facts':'train+public validation as released',
        'alignment':'released preserve_ratio=.1 mask for alignment training; all seed links for KG graph'})

    class StrictTester(module.Tester):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.best=-1.;self.index=0;self.last=None;instances.append(self)

        def score(self,batch):
            ids=torch.as_tensor(batch.copy(),device=self.device)
            table=self.target_kg.computed_entity_embedidng_KG
            prediction=self.model.predict(table[ids[:,0]],ids[:,1]).squeeze(1)
            return -torch.cdist(prediction,table,p=2)

        def test(self,args,is_val=True,is_lifted=False):
            if not is_val:return self.last
            self.index+=1
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(opts.seed+900001)
                self.pre_compute_all_embeddings(args.precompute_batch_size)
                result=evaluate(data,self.score,split='val_select',limit=64 if opts.smoke else None)
            mm=result['metrics']['select'];self.last=[mm['h1'],mm['h10'],mm['mrr']]
            with (out/'learning_curve.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'round':self.index-1,'val_select':mm})+'\n')
            if mm['mrr']>self.best:
                self.best=mm['mrr']
                caller=inspect.currentframe().f_back.f_locals
                atomic_checkpoint(out/'best.pt',{'model':self.model.state_dict(),'optimizer':caller['optimizer'].state_dict(),
                    'selected_round':self.index-1,'validation_mrr':self.best,'seed':opts.seed})
                np.save(out/'selected_candidate_embeddings.npy',self.target_kg.computed_entity_embedidng_KG.detach().cpu().numpy())
                # Preserve the evolving target graph at the selected round.
                atomic_checkpoint(out/'selected_target_graph.pt',{'kg':self.target_kg.subgraph_list_kg,'align':self.target_kg.subgraph_list_align})
            print(f'SS-AGA {opts.dataset}/{opts.target} seed={opts.seed} round={self.index-1} val={mm["mrr"]:.6f} best={self.best:.6f}',flush=True)
            return self.last

    module.Tester=StrictTester
    module.save_model=lambda *a,**kw:None
    # Original calls to the test split now return cached validation metrics;
    # only StrictTester writes a checkpoint, selected exclusively on validation.
    with ResourceTrace(out/'resources.csv') as trace:
        module.main(args)
        tester=instances[0];state=torch.load(out/'best.pt',map_location='cuda')
        tester.model.load_state_dict(state['model']);tester.model.eval()
        tester.target_kg.computed_entity_embedidng_KG=torch.from_numpy(np.load(out/'selected_candidate_embeddings.npy')).to('cuda')
        result=evaluate(data,tester.score,output=out/'test_queries.npz',limit=64 if opts.smoke else None)
        selection=evaluate(data,tester.score,split='val_select',output=out/'selection_queries.npz',limit=64 if opts.smoke else None)
        assert abs(selection['metrics']['select']['mrr']-state['validation_mrr'])<1e-10
    atomic_json(out/'result.json',dict(result,status='completed',full_data=not opts.smoke,method='SS-AGA',dataset=opts.dataset,
        kg=opts.target,seed=opts.seed,checkpoint=str(out/'best.pt'),checkpoint_sha256=sha256(out/'best.pt'),
        candidate_embedding_sha256=sha256(out/'selected_candidate_embeddings.npy'),resources=trace.summary,
        selected_round=state['selected_round'],selection=selection))


if __name__=='__main__':main()
