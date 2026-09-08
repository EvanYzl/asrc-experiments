"""Protocol adapter around the released LSMGA/DMKGC/IMKGC training loops.

Published modules remain untouched. All graph edges come from train files;
the shared tester substitutes val_select for the released in-loop test calls.
"""
import argparse
import importlib
import importlib.util
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from common import *


def prepare_data(dataset,profile):
    source=ROOT/f'data/raw/dmkgc/dataset{dataset}'
    dest=SUITE/'graph_data'/profile/f'dataset{dataset}'
    for sub in ['entity','kg','seed_alignlinks']:
        (dest/sub).mkdir(parents=True,exist_ok=True)
        for p in sorted((source/sub).iterdir()):
            if p.is_file() and not (dest/sub/p.name).exists():shutil.copy2(p,dest/sub/p.name)
    if not (dest/'relations.txt').exists():shutil.copy2(source/'relations.txt',dest/'relations.txt')
    cache_signature = dest/'strict_graph_manifest.json'
    cache_keys = {}
    if cache_signature.exists():
        cache_keys = {k.replace('\\', '/'): k for k in json.loads(cache_signature.read_text())['source_files']}
    signature={'input_facts':'train-only','k':10,'num_hops':2,'dataset':dataset,
               'source_files':{cache_keys.get(p.relative_to(source).as_posix(), p.relative_to(source).as_posix()):sha256(p) for p in sorted((source/'kg').glob('*-train.tsv'))},
               'adaptation':'DMKGC released train+validation message graph replaced by train-only; cache isolated from legacy runs'}
    if (dest/'strict_graph_manifest.json').exists():
        assert json.loads((dest/'strict_graph_manifest.json').read_text())==signature
    else:atomic_json(dest/'strict_graph_manifest.json',signature)
    return dest


def train_edges(kg_dir,language):
    triples=np.loadtxt(Path(kg_dir)/f'{language}-train.tsv',dtype=np.int64,delimiter='\t',ndmin=2)
    edges=np.vstack([np.r_[triples[:,0],triples[:,2]],np.r_[triples[:,2],triples[:,0]]])
    return torch.from_numpy(edges),torch.from_numpy(np.r_[triples[:,1],triples[:,1]])


def load_validation_progress(out):
    """Restore the score threshold separately from completed training rounds."""
    out=Path(out)
    best_path,last_path=out/'best.pt',out/'last.pt'
    best=-1.;best_index=0;completed=0
    if best_path.exists():
        saved=torch.load(best_path,map_location='cpu',weights_only=False)
        best=float(saved['validation_mrr']);best_index=int(saved['evaluation_index'])
        assert np.isfinite(best) and 0<=best<=1 and best_index>0
    if last_path.exists():
        saved=torch.load(last_path,map_location='cpu',weights_only=False)
        completed=int(saved['completed_round'])+1
        assert completed>0 and best_path.exists(),'Completed training checkpoint has no validation-best checkpoint'
    # A crash can happen after saving best.pt but before committing last.pt.
    # That last round is replayed using its original evaluation number.
    assert best_index<=completed+1,'Best checkpoint is ahead of the resumable training state'
    return best,completed


def audit_imkgc_dwy(module,args,data_path,utility):
    """Compare the released loader with the exact shared evaluation inputs."""
    assert args.dataset=='dwy'
    args.langs=list(DOMAINS['dwy'])
    loader=module.MKGDataset(args,MAX_SAM=10**10)
    source=ROOT/'data/raw/dmkgc/datasetdwy'
    files={}
    for sub in ['entity','kg','seed_alignlinks']:
        for original in sorted((source/sub).iterdir()):
            if not original.is_file():continue
            relative=original.relative_to(source)
            digest=sha256(original)
            assert sha256(data_path/relative)==digest,f'DWY graph input differs: {relative}'
            files[relative.as_posix()]=digest
    files['relations.txt']=sha256(source/'relations.txt')
    assert sha256(data_path/'relations.txt')==files['relations.txt']
    domains={}
    for kg in args.langs:
        frozen=load_kg('dwy',kg)
        train,valid,test,entities,relations=loader.load_kg_data(kg)
        for split,actual in [('train',train),('valid',valid),('test',test)]:
            np.testing.assert_array_equal(actual.numpy(),frozen['arrays'][split])
        assert (entities,relations)==(frozen['entities'],frozen['relations'])
        edge_index,edge_type=utility.get_kg_edges_for_each(str(data_path/'kg'),kg)
        expected=frozen['arrays']['train']
        np.testing.assert_array_equal(edge_index.numpy(),np.vstack([
            np.r_[expected[:,0],expected[:,2]],np.r_[expected[:,2],expected[:,0]]]))
        np.testing.assert_array_equal(edge_type.numpy(),np.r_[expected[:,1],expected[:,1]])
        domains[kg]={'entities':entities,'counts':frozen['manifest']['counts'],
                     'dataset_hash':frozen['manifest']['dataset_hash'],
                     'split_manifest_sha256':sha256(SUITE/'data_manifests'/f'dwy_{kg}.json')}
    alignments={}
    for (left,right),links in loader.load_align_links().items():
        if left not in domains or right not in domains:continue
        assert links.ndim==2 and links.shape[1]==2 and len(links)>0
        assert int(links.min())>=0
        assert int(links[:,0].max())<domains[left]['entities']
        assert int(links[:,1].max())<domains[right]['entities']
        alignments[f'{left}-{right}']=len(links)
    assert {'-'.join(sorted(pair.split('-'))) for pair in alignments}=={'db-wk','db-yg','wk-yg'}
    return {'status':'passed','data_root':str(source),'domains':domains,
            'alignment_counts':alignments,'input_sha256':files,
            'message_graph':'train-only; exact edges checked against the frozen train triples',
            'selection':'shared val_select; train+val_select filter',
            'test':'all entities; train+valid filter; equal db/wk/yg macro average',
            'recipe':'released IMKGC DWY: 30 rounds, reason_step=3, epoch_each=3; shared local batch_size=128 (release=300)'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['LSMGA','DMKGC','IMKGC'],required=True)
    parser.add_argument('--dataset',choices=['dbp5l','depkg','dwy'],required=True)
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--rounds',type=int)
    parser.add_argument('--preflight',action='store_true',help='Audit IMKGC-DWY input compatibility without training')
    parser.add_argument('--config-only',action='store_true',help='Write and inspect the parsed recipe without training')
    opts=parser.parse_args();seed_all(opts.seed)
    if opts.preflight:assert opts.method=='IMKGC' and opts.dataset=='dwy'
    out=Path(opts.output).resolve();out.mkdir(parents=True,exist_ok=True);os.chdir(out)
    if (out/'result.json').exists():return
    source=ROOT/'reproduction/sources'/opts.method
    sys.path.insert(0,str(source))
    # These are trusted locally generated graph/checkpoint files.
    original_load=torch.load
    def trusted_load(*a,**kw):kw.setdefault('weights_only',False);return original_load(*a,**kw)
    torch.load=trusted_load
    utility=importlib.import_module('src.utils')
    # Reuse the already audited exact CSR traversal and packed representation.
    dm_path=ROOT/'reproduction/sources/DMKGC/src/utils.py'
    spec=importlib.util.spec_from_file_location('strict_packed_graph_utils',dm_path)
    packed=importlib.util.module_from_spec(spec);sys.modules[spec.name]=packed;spec.loader.exec_module(packed)
    utility.get_kg_edges_for_each=train_edges
    utility.create_subgraph_list=packed.create_subgraph_list
    utility.get_k_subgraph_list=packed.get_k_subgraph_list
    module=importlib.import_module('run_model')
    OriginalTester=module.Tester
    instances=[]
    data_path=prepare_data(opts.dataset, 'train_only_k10_h2')
    # DWY also ships legacy one-letter entity files. Enumerating the whole
    # directory lets LSMGA mistake these files for additional target KGs.
    # Use the same frozen public domains as the other adapters and evaluator.
    for kg in DOMAINS[opts.dataset]:
        assert (data_path/'entity'/f'{kg}.tsv').exists()
        for split in ['train','val','test']:
            assert (data_path/'kg'/f'{kg}-{split}.tsv').exists()
    module.get_language_list=lambda directory:list(DOMAINS[opts.dataset])
    rounds=opts.rounds or ({'dbp5l':80,'depkg':50,'dwy':50}[opts.dataset] if opts.method=='LSMGA' else (50 if opts.dataset=='depkg' else 30))
    if opts.smoke:rounds=1
    command=['--dataset',opts.dataset if opts.method!='LSMGA' or opts.dataset!='dwy' else 'dbp5l',
             '--data_path',str(data_path.parent/'dataset'),'--seed',str(opts.seed),'--round',str(rounds),
             '--batch_size','200' if opts.method=='LSMGA' else '128','--test_batch_size','100',
             '--precompute_batch_size','256','--resume_checkpoint',str(out/'last.pt')]
    if opts.method=='LSMGA':command+=['--micro_batch_size','200']
    if opts.method=='DMKGC':
        command+=['--fuse_entity_groups','--v',f'strict_s{opts.seed}']
        if opts.dataset=='depkg':command+=['--s_strength','1']
    if opts.method=='IMKGC':
        command+=['--model','imkgc','--alpha','0.1','--beta','0.0001','--gamma','0.005',
                  '--omega','0.05','--vq_loss_w','1.0','--lr','0.001','--margin','0.5',
                  '--reason_step','4','--codebook_ratio','0.8','--epoch_each','2','--commit_loss','0.5','--v',f'strict_s{opts.seed}']
        if opts.dataset=='dwy':
            # The released DWY recipe uses three reasoning steps and three
            # epochs per round. Keep the shared local memory-sized batch.
            command[command.index('--reason_step')+1]='3'
            command[command.index('--epoch_each')+1]='3'
        elif opts.dataset=='depkg':
            # Dataset-specific settings from the frozen release README.
            for flag,value in {'--alpha':'0.05','--omega':'0.001',
                               '--reason_step':'2','--codebook_ratio':'0.6'}.items():
                command[command.index(flag)+1]=value
    if opts.smoke:command+=['--MAX_SAM','64','--epoch_each','1']
    args=module.parse_args(command)
    args.dataset=opts.dataset
    dwy_audit=audit_imkgc_dwy(module,args,data_path,utility) if opts.method=='IMKGC' and opts.dataset=='dwy' else None
    if dwy_audit:
        atomic_json(out/'data_adaptation.json',dwy_audit)
    if opts.preflight:
        atomic_json(out/'preflight.json',{'status':'passed','purpose':'preflight','training_performed':False,
                    'arguments':command,'data_adaptation':dwy_audit})
        print(json.dumps({'preflight':'passed','domains':list(dwy_audit['domains']),
                          'test_queries':sum(d['counts']['test'] for d in dwy_audit['domains'].values())}),flush=True)
        return
    config={'method':opts.method,'dataset':opts.dataset,'seed':opts.seed,'arguments':command,
            'purpose':'configuration_check' if opts.config_only else ('smoke' if opts.smoke else 'formal'),
            'parsed_arguments_before_author_main':vars(args).copy(),
            'domain_order':DOMAINS[opts.dataset],
            'rounds':rounds,'selection':'val_select KG macro filtered tail MRR; earliest tie',
            'input_facts':'train-only','message_graph':str(data_path/'strict_graph_manifest.json'),
            'dwy_extension':'For LSMGA-DWY, same released model; default lr=.005 margin=.3; 50 rounds; no published DWY-specific recipe' if opts.method=='LSMGA' and opts.dataset=='dwy' else None,
            'dwy_adaptation':dwy_audit,
            'loss_contract':'released PyTorch 1.10 [B,1] versus [B] broadcasting; full training batches' if opts.method=='LSMGA' else 'released model loss',
            'environment':version_info(),'source_hashes':source_hashes([Path(__file__),SUITE/'common.py',dm_path,*sorted(source.glob('src/*.py')),source/'run_model.py'])}
    atomic_json(out/'config.json',config)
    if opts.config_only:
        print(json.dumps({'configuration_check':'passed','method':opts.method,
                          'dataset':opts.dataset,'arguments':vars(args)}),flush=True)
        return

    class StrictTester(OriginalTester):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.data={kg:load_kg(opts.dataset,kg) for kg in self.kg_objects_dict}
            self.best,self.iteration=load_validation_progress(out)
            self.last_metrics=None
            instances.append(self)

        def score_fn(self,kgname):
            kg=self.kg_objects_dict[kgname]
            table=kg.computed_entity_embedding_kg
            def score(batch):
                ids=torch.as_tensor(batch.copy(),device=self.device)
                heads=table[ids[:,0]]
                if opts.method=='LSMGA':pred=self.model.predict(heads,ids[:,1])
                else:pred=self.model.predict(heads,self.model.predict_r_embedding(ids[:,1]))
                if pred.ndim==3:pred=pred[:,0,:]
                candidate=table if opts.method=='LSMGA' else self.model.predict_candidate(table)
                if candidate.ndim==3:candidate=candidate[0]
                return -torch.cdist(pred,candidate,p=2)
            return score

        def evaluate_split(self,split,save=False):
            results={}
            for kgindex,kgname in enumerate(self.kg_objects_dict):
                kg=self.kg_objects_dict[kgname];kg.computed_entity_embedding_kg=None
                # Diffusion sampling is frozen for evaluation and does not advance training RNG.
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(opts.seed+900001+kgindex);torch.cuda.manual_seed_all(opts.seed+900001+kgindex)
                    self.pre_compute_all_embeddings(kgname)
                kgout=out/kgname;kgout.mkdir(exist_ok=True)
                results[kgname]=evaluate(self.data[kgname],self.score_fn(kgname),split=split,
                    batch_size=100,output=kgout/f'{split}_queries.npz' if save else None,limit=64 if opts.smoke else None)
                if save and split=='test':
                    np.save(kgout/'candidate_embeddings.npy',kg.computed_entity_embedding_kg.detach().cpu().numpy())
                kg.computed_entity_embedding_kg=None
            return results

        def test(self,is_val=True,is_filtered=False,**kw):
            if opts.method=='LSMGA' and not is_val:return self.last_metrics
            self.iteration+=1
            with torch.no_grad():results=self.evaluate_split('val_select')
            mm=macro_metrics({k:v['metrics']['select'] for k,v in results.items()})
            with (out/'learning_curve.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'evaluation_index':self.iteration,'val_select_macro':mm,'per_kg':results})+'\n')
            if mm['mrr']>self.best:
                self.best=mm['mrr']
                atomic_checkpoint(out/'best.pt',{'model':self.model.state_dict(),'validation_mrr':self.best,
                    'evaluation_index':self.iteration,'config':config})
            self.last_metrics={k:[torch.tensor(v['metrics']['select'][metric],dtype=torch.float64) for metric in ['h1','h10','mrr']] for k,v in results.items()}
            print(f'STRICT_SELECTION {opts.method} {opts.dataset} s{opts.seed} evaluation={self.iteration} val_mrr={mm["mrr"]:.6f}',flush=True)
            return self.last_metrics

    module.Tester=StrictTester
    module.save_model=lambda *a,**kw:None # best saved above; original full-state round checkpoint retained.
    with ResourceTrace(out/'resources.csv') as trace:
        module.main(args)
        tester=instances[-1]
        state=torch.load(out/'best.pt',map_location='cuda')
        tester.model.load_state_dict(state['model']);tester.model.eval()
        with torch.no_grad():
            test=tester.evaluate_split('test',save=True)
            selection=tester.evaluate_split('val_select',save=True)
    per_kg={kg:result['metrics'][result['primary_filter']] for kg,result in test.items()}
    atomic_json(out/'result.json',{'status':'completed','purpose':'smoke' if opts.smoke else 'formal','full_data':not opts.smoke,
        'method':opts.method,'dataset':opts.dataset,'seed':opts.seed,'macro':macro_metrics(per_kg),
        'per_kg':test,'selection':selection,'best_evaluation':state['evaluation_index'],
        'checkpoint':str(out/'best.pt'),'checkpoint_sha256':sha256(out/'best.pt'),'config':str(out/'config.json'),
        'protocol':PROTOCOL_VERSION,'resources':trace.summary,'parameter_count':sum(p.numel() for p in tester.model.parameters())})


if __name__=='__main__':main()
