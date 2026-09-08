"""Verify every saved ranking, fill all missing cells, and render the frozen table."""
from pathlib import Path
import ast
import csv
import copy
import datetime as dt
import hashlib
import json
import shutil
import numpy as np

ROOT = Path('G:/zhishitupui')
BASE = Path(__file__).resolve().parent
TABLE = ROOT / 'outputs/kbs/_main/_tables/language_breakdown_seed17'
METRICS = ('h1', 'h10', 'mrr')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def rank_values(path, ds, kg, filt):
    with np.load(path, allow_pickle=False) as z:
        rank = z['rank_' + filt].astype(np.float64)
        triples = z['triples']
        assert np.array_equal(z['query_index'], np.arange(len(rank)))
    if ds == 'wk3l':
        raw = ROOT / 'data/raw/atransn' / ('WK3l-15k_EN_F' if kg in ['en','en_f'] else 'WK3l-15k_FR') / 'test_triple_id.txt'
    else:
        raw = ROOT / f'data/raw/dmkgc/dataset{ds}/kg/{kg}-test.tsv'
    assert np.array_equal(triples, np.loadtxt(raw, dtype=np.int64, ndmin=2)), (ds, kg, path)
    assert np.isfinite(rank).all() and np.all(rank >= 1)
    return {'n':len(rank),'h1':float(np.mean(rank<=1)),'h10':float(np.mean(rank<=10)),'mrr':float(np.mean(1/rank))}


def accepted(source, ds, method):
    result = read(source)
    assert result['status']=='completed' and result['seed']==17 and result['dataset']==ds
    assert result.get('full_data') is True and result['method'].lower()==method.lower()
    groups = result.get('per_kg') or {kg:item['test'] for kg,item in result['domains'].items()}
    values, records = {}, []
    for kg, group in groups.items():
        filt = 'all' if ds=='wk3l' else 'train_valid'
        assert group['primary_filter']==filt
        ranks = source.parent / kg / 'test_queries.npz'
        v = rank_values(ranks,ds,kg,filt)
        for metric in METRICS:
            assert abs(v[metric]-group['metrics'][filt][metric])<1e-12
        assert group['count']==v['n']
        values[kg]=v
        records.append({'dataset':ds,'method':method,'seed':17,'kg':'en_f' if ds=='wk3l' and kg=='en' else kg,
                        'status':'completed',**v,'primary_filter':filt,'source_result':relative(source),'source_rank':relative(ranks)})
    return result, values, records


def main():
    if (BASE/'IMPORT_RECEIPT.json').exists():
        print(json.dumps({'status':'already_imported','next':'compile and verify the current table'}))
        return 0
    state = read(BASE/'state.json')
    if state['status']!='completed':
        print(json.dumps({'status':'waiting','reason':'supplemental training is not complete'}))
        return 75
    parent_state = read(ROOT/'reproduction/runs/strict_baselines_20260905/queue_state.json')
    existing = parent_state['jobs']['imkgc_depkg_s17']
    if existing['status']!='completed':
        print(json.dumps({'status':'waiting','reason':'existing IMKGC/E-PKG seed17 has not completed'}))
        return 75
    data = read(TABLE/'language_kg_seed17.json')
    before = copy.deepcopy(data)
    rows = data['rows']
    with (TABLE/'language_kg_seed17.csv').open(encoding='utf-8-sig',newline='') as f:
        records = list(csv.DictReader(f))
    index = {(r['dataset'],r['method'],r['kg']):r for r in records}

    for method in ['TransE','DistMult','RotatE','ATransN']:
        source=BASE/'jobs'/f'{method.lower()}_en_s17/result.json'
        result, values, new = accepted(source,'wk3l',method)
        assert set(values)=={'en'}
        row=next(r for r in rows['wk3l'] if r['method']==method)
        row['supplemental_kg_results']={'en_f':values['en']}
        row['supplemental_sources']={'en_f':{'result':relative(source),'rank':new[0]['source_rank']}}
        index[('wk3l',method,'en_f')]=new[0]
    for method in ['LSMGA','DMKGC','IMKGC']:
        source=BASE/'jobs'/f'{method.lower()}_wk3l_s17/result.json'
        result, values, new=accepted(source,'wk3l',method)
        assert set(values)=={'fr','en'}
        assert read(source.parent/'FINAL_EVALUATION_FREEZE.json')['checkpoint_sha256']==sha(source.parent/'best.pt')
        row=next(r for r in rows['wk3l'] if r['method']==method)
        row.update(status='completed',source=relative(source),per_kg={'fr':values['fr']},
                   supplemental_kg_results={'en_f':values['en']},checkpoint_sha256=result['checkpoint_sha256'],
                   macro={m:values['fr'][m] for m in METRICS},selection_scope='FR val_select only')
        for record in new:index[('wk3l',method,record['kg'])]=record
        index[('wk3l',method,'AVG')]={**next(r for r in new if r['kg']=='fr'),'kg':'AVG','source_rank':''}

    source=ROOT/'reproduction/runs/strict_baselines_20260905/jobs/imkgc_depkg_s17/result.json'
    result,values,new=accepted(source,'depkg','IMKGC')
    row=next(r for r in rows['depkg'] if r['method']=='IMKGC')
    row.update(status='completed',source=relative(source),per_kg=values,checkpoint_sha256=result['checkpoint_sha256'],
               macro={m:float(np.mean([v[m] for v in values.values()])) for m in METRICS})
    for record in new:index[('depkg','IMKGC',record['kg'])]=record
    index[('depkg','IMKGC','AVG')]={**new[0],'kg':'AVG','n':sum(v['n'] for v in values.values()),**row['macro'],'source_rank':''}

    # Both teacher checkpoints now exist under the same seed and teacher recipe.
    source=BASE/'jobs/transe_fr_teacher_s17/result.json'
    result,values,new=accepted(source,'wk3l','TransE')
    assert set(values)=={'fr'} and result['teacher'] is True
    teacher=next(r for r in rows['wk3l'] if r['method_key']=='en_teacher')
    old_label=teacher['method']; new_label='TransE (teacher)'
    previous_teacher_hash=teacher['checkpoint_sha256']
    english_teacher_hash=previous_teacher_hash['en'] if isinstance(previous_teacher_hash,dict) else previous_teacher_hash
    english_record=dict(index[('wk3l',old_label,'en_f')]);english_record['method']=new_label
    teacher.update(method=new_label,role='source_teachers',source=relative(source),per_kg=values,
                   macro={m:values['fr'][m] for m in METRICS},
                   checkpoint_sha256={'en':english_teacher_hash,'fr':result['domains']['fr']['checkpoint_sha256']})
    index={k:v for k,v in index.items() if not (k[0]=='wk3l' and k[1]==old_label)}
    index[('wk3l',new_label,'en_f')]=english_record
    index[('wk3l',new_label,'fr')]={**new[0],'method':new_label}
    index[('wk3l',new_label,'AVG')]={**new[0],'method':new_label,'kg':'AVG','source_rank':''}

    ordered=[];diagnostics=[];source_hashes={}
    keep=np.load(TABLE/'en_f_evaluation/overlap_masks.npz',allow_pickle=False)['diagnostic_keep']
    assert len(keep)==40700 and int(keep.sum())==40654
    for ds,dataset_rows in rows.items():
        for row in dataset_rows:
            assert row['status']=='completed'
            for kg in [*row['per_kg'],*row.get('supplemental_kg_results',{}),'AVG']:
                record=index[(ds,row['method'],kg)]
                assert record['status']=='completed'
                if kg!='AVG':
                    rank_path=ROOT/record['source_rank'];v=rank_values(rank_path,ds,kg,record['primary_filter'])
                    for m in METRICS:assert abs(float(record[m])-v[m])<1e-12
                    source_hashes[record['source_rank']]=sha(rank_path)
                    source_hashes[record['source_result']]=sha(ROOT/record['source_result'])
                    if ds=='wk3l' and kg=='en_f':
                        r=np.load(rank_path,allow_pickle=False)['rank_all'].astype(np.float64)[keep]
                        diagnostics.append({'method':row['method'],'seed':17,'n':len(r),'h1':float(np.mean(r<=1)),
                                            'h10':float(np.mean(r<=10)),'mrr':float(np.mean(1/r)),'source_rank':record['source_rank']})
                ordered.append(record)
    assert sum(len(r['per_kg'])*3+len(r.get('supplemental_kg_results',{}))*3+1 for rs in rows.values() for r in rs)==378
    for ds, old_rows in before['rows'].items():
        for old_row in old_rows:
            new_row=next(r for r in rows[ds] if r['method_key']==old_row['method_key'])
            for group in ['per_kg','supplemental_kg_results']:
                for kg,values in old_row.get(group,{}).items():
                    assert new_row[group][kg]==values, (ds,old_row['method'],kg)
            if old_row.get('macro') is not None:
                assert new_row['macro']==old_row['macro']
    stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup=TABLE/'versions'/('before_completed_training_import_'+stamp);backup.mkdir(parents=True,exist_ok=False)
    for p in TABLE.iterdir():
        if p.is_file():shutil.copy2(p,backup/p.name)
    previous_numeric_cells=sum(len(r.get('per_kg',{}))*3+len(r.get('supplemental_kg_results',{}))*3+int(r.get('macro') is not None) for rs in before['rows'].values() for r in rs)
    data.pop('partial_completion_batch',None)
    data.update(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),completion_batch=relative(BASE/'manifest.json'),publication_status='completed')
    (TABLE/'language_kg_seed17.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (TABLE/'language_kg_seed17.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(ordered)
    with (BASE/'EN_F_overlap_excluded.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(diagnostics[0]));writer.writeheader();writer.writerows(diagnostics)

    script=(TABLE/'build_table.py').read_text(encoding='utf-8')
    constants={}
    for node in ast.parse(script).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['SPECS','METRICS']:
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    constants['SPECS']['wk3l']=('WK3l-15k',[('en_f',r'EN\_F (supplemental)'),('fr','FR (primary)')])
    ns=dict(constants,OUT=TABLE,method_rows=rows,supplement=True)
    render=script[script.index('def column_value('):script.index('for rel, item in raw_sources.items():')]
    exec(compile(render,str(TABLE/'build_table.py')+' [render verified completed results]','exec'),ns)
    tex_path=TABLE/'table_language_kg_seed17.tex';tex=tex_path.read_text(encoding='utf-8')
    old='TransE (EN teacher) is the reused ATransN source teacher (margin 4, batch 1024), with no FR/AVG entry; it is not an English-target ATransN result.'
    new='TransE (teacher) uses the source-teacher recipe (margin 4, batch 1024) in both languages. English ATransN uses FR-to-EN transfer. Graph methods use a WK3l data-format adaptation with fixed recipes; one FR-validation-selected checkpoint serves both languages.'
    assert old in tex;tex=tex.replace(old,new)
    old='P: seed-17 result pending; NR: no accepted result; \\textit{n/a}: not applicable. All filled values were verified against saved ranks and source JSON. English ASRC inference was added using frozen weights; no model was retrained.'
    new='All displayed results are complete seed-17 runs, verified against saved ranks. Missing baselines were trained with fixed recipes and validation-only checkpoint selection; existing ASRC and baseline results were reused.'
    assert old in tex;tex=tex.replace(old,new)
    tex_path.write_text(tex,encoding='utf-8')
    receipt={'timestamp':data['timestamp'],'status':'ready_to_compile','numeric_cells':378,'new_numeric_cells':378-previous_numeric_cells,'batch_total_new_numeric_cells':56,
             'empty_numeric_cells':0,'all_seed17':True,'teacher_row':'Both language-specific teachers, same teacher recipe',
             'all_original_primary_values_preserved':True,'wk3l_avg':'FR only','backup':str(backup),
             'source_hashes':source_hashes,'diagnostic_en_count':40654,'completed_new_training_jobs':8}
    (BASE/'IMPORT_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (TABLE/'DATA_AUDIT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ['status','numeric_cells','new_numeric_cells','empty_numeric_cells']}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
