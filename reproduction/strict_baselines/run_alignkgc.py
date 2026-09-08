"""AlignKGC local reruns using frozen 20% EA/20% RA preprocessing.

The released trainer.step is unchanged. All checkpoint selection and final
ranking use the audited tail evaluator; the original Hits@10 <=11 bug is
therefore not propagated into a table labelled Hits@10.
"""
import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from common import *


def prepare(dataset,source):
    dest=SUITE/'alignkgc_data'/dataset
    if (dest/'preparation_manifest.json').exists():return dest
    if dataset=='dbp5l':
        original=ROOT/'reproduction/prepared/alignkgc/DBP-5L'
        for sub in ['kgs','entity_lists','seed_alignment','seed_alignment_20','combined/Combined_20_20']:
            shutil.copytree(original/sub,dest/sub,dirs_exist_ok=True,ignore=shutil.ignore_patterns('model_*','*.pt','*.ckpt'))
        for name in ['relations.txt','relat_20.txt']:shutil.copy2(original/name,dest/name)
    else:
        raw=ROOT/f'data/raw/dmkgc/dataset{dataset}'
        for srcsub,dstsub in [('kg','kgs'),('entity','entity_lists'),('seed_alignlinks','seed_alignment')]:
            shutil.copytree(raw/srcsub,dest/dstsub,dirs_exist_ok=True)
        shutil.copy2(raw/'relations.txt',dest/'relations.txt')
        np.random.seed(41)
        sampler=importlib.import_module('importers.dbp5l_sampler')
        (dest/'seed_alignment_20').mkdir(exist_ok=True)
        for f in sorted((dest/'seed_alignment').glob('*.tsv')):
            sampler.sample_ent_equiv(str(f),20,str(dest/'seed_alignment_20'/f.name))
        sampler.sample_multiling_rels(SimpleNamespace(inpath=str(dest),rel_percent=20))
        combiner=importlib.import_module('importers.dbp5l_combiner')
        # Same insertion order and connected components, O(1) membership.
        class FastGraph(combiner.Graph):
            def addNode(self,v):
                if v not in self.adj:self.V.append(v);self.adj[v]=[]
        def collect_train(meta,lang,graph,ents):
            seen=set()
            for h,r,t in np.loadtxt(Path(meta.dir)/'kgs'/f'{lang}-train.tsv',dtype=np.int64,delimiter='\t',ndmin=2):
                for e in [int(h),int(t)]:
                    if e not in seen:ents.append(e);seen.add(e)
                    graph.addNode((lang,e))
        combiner.Graph=FastGraph;combiner.collect_train_kg=collect_train
        (dest/'combined').mkdir(exist_ok=True)
        combiner.main(SimpleNamespace(dbp5l=str(dest),combined=str(dest/'combined'),ea_percent=20,ra_percent=20))
    hashes={str(f.relative_to(dest)):sha256(f) for sub in ['kgs','seed_alignment_20','combined/Combined_20_20'] for f in sorted((dest/sub).glob('*')) if f.is_file()}
    hashes['relat_20.txt']=sha256(dest/'relat_20.txt')
    for kgfile in (dest/'kgs').glob('*.tsv'):
        if not any(kgfile.name.endswith('-'+split+'.tsv') for split in ['train','val','test']):continue
        rawfile=ROOT/f'data/raw/dmkgc/dataset{dataset}/kg'/kgfile.name
        assert np.array_equal(np.loadtxt(kgfile,dtype=np.int64,ndmin=2),np.loadtxt(rawfile,dtype=np.int64,ndmin=2)),f'Prepared/raw mismatch: {kgfile}'
    atomic_json(dest/'preparation_manifest.json',{'dataset':dataset,'alignment_percent':20,'relation_alignment_percent':20,
        'preprocessing_seed':41,'recipe':'released sampler + connected-component combiner; frozen across training seeds','files':hashes})
    return dest


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['dbp5l','depkg'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    opts=p.parse_args();out=Path(opts.output).resolve();out.mkdir(parents=True,exist_ok=True);os.chdir(out)
    source=ROOT/'reproduction/sources/AlignKGC/alignkgc-master';sys.path.insert(0,str(source))
    base=importlib.import_module('AlignKGC.alignkgc_base')
    module=importlib.import_module('AlignKGC.AlignKGC_trainer')
    data_path=prepare(opts.dataset,source);seed_all(opts.seed)
    args=base.get_base_argparser().parse_args(['--dbp5l',str(data_path),'--seed',str(opts.seed),'--output_root',str(out/'training'),
        '--ea_percent','20','--ra_percent','20','--eval_batch_size','256','--tensorboard_log_every','100'])
    if opts.smoke:args.MAX_SAM=64
    (out/'training').mkdir(exist_ok=True)
    args.resume_checkpoint=None
    data={kg:load_kg(opts.dataset,kg) for kg in DOMAINS[opts.dataset]}
    atomic_json(out/'config.json',{'method':'AlignKGC','dataset':opts.dataset,'seed':opts.seed,'recipe':vars(args),
        'environment':version_info(),'source_hashes':source_hashes([Path(__file__),SUITE/'common.py',*sorted(source.glob('AlignKGC/*.py')),*sorted(source.glob('importers/*.py'))]),
        'preprocessing_manifest':str(data_path/'preparation_manifest.json'),
        'selection':'val_select tail MRR, equal-KG macro, earliest maximum',
        'evaluation_adaptation':'all original target entity IDs; released alignment coalescing and OOV mapping; train+valid filter; deterministic ID ties; Hits@10 means rank<=10',
        'resume_policy':'completed jobs retained; interrupted dataset training restarts from seed; selected checkpoint retained for audit/inference'})
    with ResourceTrace(out/'resources.csv') as trace:
        trainer=module.AlignKGC(meta=base.EaRaKgcData(str(data_path)),**vars(args))
        mapping={}
        for line in (data_path/'combined/Combined_20_20/mapping.txt').read_text().splitlines():
            gid,local,kg=line.split();mapping[kg,int(local)]=gid
        aligned={int(x) for x in (data_path/'relat_20.txt').read_text().split()}
        entity_lookup={};relation_lookup={};mapper=trainer.dltrain.kb
        model=trainer.scoring_function
        for kg,d in data.items():
            entity_lookup[kg]=torch.tensor([mapper.entity_map.get(mapping.get((kg,e),'__unobserved__'),len(mapper.entity_map)-1) for e in range(d['entities'])],device='cuda')
            lid=trainer.meta.lang_to_lid(kg);uid=trainer.meta.maxlid_plus_one()
            relation_lookup[kg]=torch.tensor([mapper.relation_map.get(str(trainer.meta.lang_rel_do_prefix(uid if r in aligned else lid,int(r))),len(mapper.relation_map)-1) for r in range(d['relations'])],device='cuda')
            np.savez_compressed(out/f'{kg}_model_id_map.npz',entity=entity_lookup[kg].cpu().numpy(),relation=relation_lookup[kg].cpu().numpy())

        def score_fn(kg):
            def score(batch):
                b=torch.as_tensor(batch.copy(),device='cuda')
                scores=model(entity_lookup[kg][b[:,0]][:,None],relation_lookup[kg][b[:,1]][:,None],None)
                return scores[:,entity_lookup[kg]]
            return score

        def validation(step):
            model.eval()
            per={kg:evaluate(d,score_fn(kg),split='val_select',limit=64 if opts.smoke else None)['metrics']['select'] for kg,d in data.items()}
            mm=macro_metrics(per)
            with (out/'learning_curve.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'step':step,'val_select':per,'macro':mm})+'\n')
            model.train();return mm

        best=-1.;steps=32 if opts.smoke else int(args.max_epochs*len(trainer.dltrain.kb.facts)/args.batch_size)
        for step in range(steps):
            loss=trainer.step(step)
            if (step+1)%100==0:
                with (out/'training_curve.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'step':step+1,**loss})+'\n')
            if (step+1)%1000==0 or step+1==steps:
                with torch.random.fork_rng(devices=[0]):mm=validation(step+1)
                trainer.scheduler.step(mm['mrr'])
                payload={'model':model.state_dict(),'optimizer':trainer.optim.state_dict(),'scheduler':trainer.scheduler.state_dict(),
                    'step':step+1,'validation_mrr':mm['mrr'],'entity_map':mapper.entity_map,'relation_map':mapper.relation_map,
                    'b':trainer.b.detach().cpu(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
                atomic_checkpoint(out/'last.pt',payload)
                if mm['mrr']>best:best=mm['mrr'];atomic_checkpoint(out/'best.pt',payload)
                print(f'AlignKGC {opts.dataset} s{opts.seed} step={step+1}/{steps} val={mm["mrr"]:.6f} best={best:.6f}',flush=True)
        trainer.tbwriter.flush();trainer.tbwriter.close();sys.stdout=trainer.saved_stdout;sys.stderr=trainer.saved_stderr
        selected=torch.load(out/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(selected['model']);model.eval()
        results={}
        for kg,d in data.items():
            results[kg]=evaluate(d,score_fn(kg),output=out/kg/'test_queries.npz',limit=64 if opts.smoke else None)
            evaluate(d,score_fn(kg),split='val_select',output=out/kg/'selection_queries.npz',limit=64 if opts.smoke else None)
    atomic_json(out/'result.json',{'status':'completed','purpose':'smoke' if opts.smoke else 'formal','full_data':not opts.smoke,
        'method':'AlignKGC','dataset':opts.dataset,'seed':opts.seed,'per_kg':results,
        'macro':macro_metrics({kg:r['metrics']['train_valid'] for kg,r in results.items()}),
        'checkpoint':str(out/'best.pt'),'checkpoint_sha256':sha256(out/'best.pt'),'resources':trace.summary,
        'protocol':'alignkgc-20-20-local-tail-v1','selected_step':selected['step']})


if __name__=='__main__':main()
