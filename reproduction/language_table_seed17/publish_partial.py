"""Import the four completed foundation/teacher runs without marking the batch done."""
import argparse
import copy
import csv
import datetime as dt
import json
import runpy
from pathlib import Path
import numpy as np
import publish as p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backup',required=True,type=Path)
    args=parser.parse_args()
    assert (args.backup/'language_kg_seed17.json').exists()
    assert not (p.BASE/'IMPORT_RECEIPT.json').exists()
    data=p.read(p.TABLE/'language_kg_seed17.json')
    before=copy.deepcopy(data)
    assert not data.get('completion_batch')
    state=p.read(p.BASE/'state.json')
    with (p.TABLE/'language_kg_seed17.csv').open(encoding='utf-8-sig',newline='') as f:
        records=list(csv.DictReader(f))
    index={(r['dataset'],r['method'],r['kg']):r for r in records}
    new_jobs=[]
    freezes={}
    for name,method,kg in [('transe_en_s17','TransE','en'),('distmult_en_s17','DistMult','en'),
                           ('rotate_en_s17','RotatE','en'),('transe_fr_teacher_s17','TransE','fr')]:
        assert state['jobs'][name]['status']=='completed' and state['jobs'][name]['exit_code']==0
        source=p.BASE/'jobs'/name/'result.json'
        result,values,new=p.accepted(source,'wk3l',method)
        assert set(values)=={kg}
        freeze=p.read(source.parent/kg/'FINAL_EVALUATION_FREEZE.json')
        checkpoint_hash=p.sha(source.parent/kg/'best.pt')
        assert freeze['checkpoint_sha256']==checkpoint_hash==result['domains'][kg]['checkpoint_sha256']
        assert freeze['seed']==17 and freeze['selection']=='val_select only'
        freezes[name]={**freeze,'source_result':p.relative(source)}
        if kg=='en':
            row=next(r for r in data['rows']['wk3l'] if r['method']==method)
            row['supplemental_kg_results']={'en_f':values['en']}
            row['supplemental_sources']={'en_f':{'result':p.relative(source),'rank':new[0]['source_rank'],'checkpoint_sha256':checkpoint_hash}}
            index[('wk3l',method,'en_f')]=new[0]
        else:
            assert result['teacher'] is True
            teacher=next(r for r in data['rows']['wk3l'] if r['method_key']=='en_teacher')
            old_label=teacher['method'];label='TransE (teacher)'
            english=dict(index[('wk3l',old_label,'en_f')]);english['method']=label
            old_hash=teacher['checkpoint_sha256']
            english_hash=old_hash['en'] if isinstance(old_hash,dict) else old_hash
            teacher.update(method=label,role='source_teachers',source=p.relative(source),per_kg=values,
                           macro={m:values['fr'][m] for m in p.METRICS},checkpoint_sha256={'en':english_hash,'fr':checkpoint_hash})
            index={k:v for k,v in index.items() if not (k[0]=='wk3l' and k[1]==old_label)}
            index[('wk3l',label,'en_f')]=english
            index[('wk3l',label,'fr')]={**new[0],'method':label}
            index[('wk3l',label,'AVG')]={**new[0],'method':label,'kg':'AVG','source_rank':''}
        new_jobs.append(name)

    # Every previously displayed unrounded value must survive the import.
    for ds,old_rows in before['rows'].items():
        for old in old_rows:
            row=next(r for r in data['rows'][ds] if r['method_key']==old['method_key'])
            for group in ['per_kg','supplemental_kg_results']:
                for kg,value in old.get(group,{}).items():assert row[group][kg]==value
            if old.get('macro') is not None:assert row['macro']==old['macro']
    count=lambda d:sum(3*len(r.get('per_kg',{}))+3*len(r.get('supplemental_kg_results',{}))+int(r.get('macro') is not None) for rows in d['rows'].values() for r in rows)
    assert count(data)==335
    source_hashes={};diagnostics=[]
    keep=np.load(p.TABLE/'en_f_evaluation/overlap_masks.npz',allow_pickle=False)['diagnostic_keep']
    assert len(keep)==40700 and int(keep.sum())==40654
    for record in index.values():
        if record['status']!='completed':continue
        row=next(r for r in data['rows'][record['dataset']] if r['method']==record['method'])
        kg=record['kg']
        value=row['macro'] if kg=='AVG' else row.get('supplemental_kg_results',{}).get(kg,row['per_kg'].get(kg))
        for m in p.METRICS:assert abs(value[m]-float(record[m]))<1e-12
        if kg=='AVG':continue
        rank_path=p.ROOT/record['source_rank']
        actual=p.rank_values(rank_path,record['dataset'],kg,record['primary_filter'])
        for m in p.METRICS:assert abs(actual[m]-value[m])<1e-12
        source_hashes[record['source_rank']]=p.sha(rank_path)
        source_hashes[record['source_result']]=p.sha(p.ROOT/record['source_result'])
        if record['dataset']=='wk3l' and kg=='en_f':
            with np.load(rank_path,allow_pickle=False) as z:rank=z['rank_all'].astype(np.float64)[keep]
            diagnostics.append({'method':record['method'],'seed':17,'n':len(rank),'h1':float(np.mean(rank<=1)),
                                'h10':float(np.mean(rank<=10)),'mrr':float(np.mean(1/rank)),'source_rank':record['source_rank']})
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    data.update(timestamp=now,partial_completion_batch=p.relative(p.BASE/'manifest.json'),publication_status='partial')
    (p.TABLE/'language_kg_seed17.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (p.TABLE/'language_kg_seed17.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(index.values())
    with (p.BASE/'EN_F_overlap_excluded.partial.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(diagnostics[0]));writer.writeheader();writer.writerows(diagnostics)
    runpy.run_path(str(p.BASE/'render_completed.py'),run_name='__main__')
    receipt={'timestamp':now,'status':'partial_ready_to_compile','numeric_cells':count(data),'new_numeric_cells':count(data)-count(before),
             'missing_numeric_cells':378-count(data),'all_original_values_preserved':True,'completed_new_jobs':new_jobs,
             'checkpoint_freezes':freezes,'source_hashes':source_hashes,'backup':str(args.backup),'wk3l_avg':'FR only'}
    for path in [p.BASE/'PARTIAL_IMPORT_RECEIPT.json',p.TABLE/'DATA_AUDIT.json']:
        path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (p.BASE/'TRAINING_LOG.md').open('a',encoding='utf-8') as f:
        f.write('\n\n'+json.dumps({'time':now,'event':'partial_table_import','purpose':'Publish already completed seed17 results on user request',
                                 'jobs':new_jobs,'new_numeric_cells':count(data)-count(before),'numeric_cells':count(data),
                                 'training_changed':False,'next':'Compile and verify this partial PDF; existing training continues'},ensure_ascii=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','numeric_cells','new_numeric_cells','missing_numeric_cells']}))


if __name__=='__main__':main()
