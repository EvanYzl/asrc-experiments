"""ATransN WK3l reproduction: original adversarial updates, independent selection."""
import argparse
import importlib
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from common import *
from run_kge import score_all


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--teacher',required=True);p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true');opts=p.parse_args()
    out=Path(opts.output).resolve();out.mkdir(parents=True,exist_ok=True);os.chdir(out)
    seed_all(opts.seed)
    source=ROOT/'reproduction/sources/ATransN/src';sys.path.insert(0,str(source))
    module=importlib.import_module('run_wgan_kge')
    original_load=torch.load
    def trusted_load(*a,**kw):kw.setdefault('weights_only',False);return original_load(*a,**kw)
    torch.load=trusted_load
    previous=next((ROOT/'reproduction/runs/baseline_reproduction_20260904/artifacts/atransn').glob('*/config.json'))
    config=json.loads(previous.read_text(encoding='utf-8'))
    config.update(seed=opts.seed,teacher_data_path=str(ROOT/'data/raw/atransn/WK3l-15k_EN_F'),
                  student_data_path=str(ROOT/'data/raw/atransn/WK3l-15k_FR'),
                  shared_entity_path=str(ROOT/'data/raw/atransn/SHARED/wk3l-15k_en_f_fr_aligned_entity_id.txt'),
                  teacher_model_path=str(Path(opts.teacher).resolve()),save_path=str(out/'checkpoints'),
                  steps=32 if opts.smoke else 10000,eval_steps=32 if opts.smoke else 2000,save_steps=2000,
                  print_steps=100,num_workers=0,test_batch=32,MAX_SAM=10000000000)
    Path(config['save_path']).mkdir(exist_ok=True)
    args=SimpleNamespace(**config);data=load_kg('wk3l','fr')
    provenance={'method':'ATransN','seed':opts.seed,'dataset':'wk3l','recipe':config,
                'selection':'val_select filtered tail MRR; no test or certification access during selection',
                'teacher_checkpoint_sha256':sha256(Path(opts.teacher)/'checkpoint_valid.pt'),
                'dataset_hash':data['manifest']['dataset_hash'],'environment':version_info(),
                'source_hashes':source_hashes([Path(__file__),SUITE/'common.py',*sorted(source.glob('*.py'))])}
    atomic_json(out/'config.json',provenance)
    # Preserve the released training body, replacing only selection/reporting.
    code=inspect.getsource(module.train)
    expr='valid_best_metrics["MRR"] + valid_best_metrics["HITS@3"] < valid_metrics["MRR"] + valid_metrics["HITS@3"]'
    assert code.count(expr)==1
    code=code.replace(expr,'valid_best_metrics["MRR"] < valid_metrics["MRR"]')
    marker="logging.info('evaluating on test dataset...')"
    pos=code.index(marker)
    start=code.rfind('            logging.info("--------------------------------------")',0,pos)
    end=code.index('            learner.train()',pos)
    assert start>=0 and end>start
    code=code[:start]+code[end:]
    code=code.replace('    log_metrics(\'test-best\', test_best_metrics["step"], test_best_metrics)','')
    (out/'adapted_training_function.py').write_text(code,encoding='utf-8')
    exec(compile(code,str(out/'adapted_training_function.py'),'exec'),module.__dict__)
    model_box={};eval_index=0

    def strict_link(args,model,unused_dataloaders):
        nonlocal eval_index
        eval_index+=1;model_box['model']=model
        result=evaluate(data,lambda b:score_all(model,b),split='val_select',limit=64 if opts.smoke else None)
        with (out/'learning_curve.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'step':min(eval_index*args.eval_steps,args.steps),'val_select':result})+'\n')
        m=result['metrics']['select']
        return {'MRR':m['mrr'],'HITS@1':m['h1'],'HITS@3':m['h3'],'HITS@10':m['h10']}

    def save_state(payload,filename):
        converted={k:v.state_dict() if hasattr(v,'state_dict') else v for k,v in payload.items()}
        converted['cpu_rng']=torch.get_rng_state();converted['cuda_rng']=torch.cuda.get_rng_state()
        target=out/('best.pt' if Path(filename).name=='checkpoint_valid.pt' else 'last.pt')
        atomic_checkpoint(target,converted)

    module.link_prediction=strict_link;module.save_model=save_state
    module.set_logger(str(out/'train.log'))
    with ResourceTrace(out/'resources.csv') as trace:
        module.train(args)
        state=torch.load(out/'best.pt',map_location='cuda')
        model=model_box['model'];model.load_state_dict(state['model']);model.eval()
        kgout=out/'fr';kgout.mkdir(exist_ok=True)
        result=evaluate(data,lambda b:score_all(model,b),output=kgout/'test_queries.npz',limit=64 if opts.smoke else None)
        selection=evaluate(data,lambda b:score_all(model,b),split='val_select',output=kgout/'selection_queries.npz',limit=64 if opts.smoke else None)
    atomic_json(out/'result.json',{'status':'completed','purpose':'smoke' if opts.smoke else 'formal','full_data':not opts.smoke,
        'method':'ATransN','dataset':'wk3l','seed':opts.seed,'macro':{k:result['metrics']['all'][k] for k in ['mrr','h1','h3','h10']},
        'per_kg':{'fr':result},'selection':selection,'checkpoint':str(out/'best.pt'),'checkpoint_sha256':sha256(out/'best.pt'),
        'resources':trace.summary,'protocol':PROTOCOL_VERSION,'parameter_count':sum(p.numel() for p in model.parameters())})


if __name__=='__main__':main()
