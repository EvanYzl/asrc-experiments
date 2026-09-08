"""Update the explicitly provisional SS-AGA/E-PKG row from one to two seeds."""
import argparse
import datetime
import shutil
from pathlib import Path
import numpy as np
from publish_available_main_tables import ROOT, BASE, PAPER, TABLES, read, save, sha, readcsv
from refresh_table2_highlights import refresh_highlights
import csv


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    rev=p.parse_args().archive.resolve()
    assert rev.is_relative_to((PAPER/'document_versions').resolve())
    assert sha(rev/'before/tables/cells_results.csv')==sha(TABLES/'cells_results.csv')
    summary=read(BASE/'INTERIM_TWO_SEED_RESULTS.json');g=summary['groups'][0]
    assert summary['interim_script_sha256']==sha(ROOT/'reproduction/sota/audit_ssaga_two_seed.py')
    assert summary['audit_script_sha256']==sha(ROOT/'reproduction/sota/audit_main_completion.py')
    assert summary['plan_sha256']==sha(BASE/'PLAN.json')
    assert g['status']=='interim_two_seed' and g['seeds']==[17,29] and g['planned_seeds']==[17,29,43]
    for r in g['runs']:
        assert r==read(BASE/'publication_acceptance'/(r['run_id']+'.json')) and r['passed']
        result=ROOT/r['result_path']
        for path,key in [(result,'result_sha256'),(result.with_name('config.json'),'config_sha256'),(result.with_name('best.pt'),'checkpoint_sha256')]:
            assert sha(path)==r[key]
        for raw in r['raw_artifacts']:assert sha(ROOT/raw['path'])==raw['sha256']
    for m in ['mrr','h1','h3','h10']:
        x=[g['per_seed'][str(s)][m] for s in g['seeds']]
        assert float(np.mean(x))==g['mean'][m] and float(np.std(x,ddof=1))==g['sample_sd'][m]
    src=rev/'source_results';src.mkdir()
    for n in ['INTERIM_TWO_SEED_RESULTS.json','PUBLICATION_SOURCE_RECEIPT.json']:shutil.copy2(BASE/n,src/n)
    (src/'acceptance').mkdir()
    for r in g['runs']:shutil.copy2(BASE/'publication_acceptance'/(r['run_id']+'.json'),src/'acceptance'/(r['run_id']+'.json'))
    fields,rows=readcsv(TABLES/'cells_results.csv');before={r['cell_id']:r.copy() for r in rows};updates={}
    for r in rows:
        cid=r['cell_id']
        if not cid.startswith('T3.epkg.ssaga.'):continue
        assert r['seed']=='17' and r['notes'].startswith('INTERIM_SINGLE_SEED:')
        m=cid.rsplit('.',1)[1];label=m=='paper_table'
        if not label:assert float(r['value'])==g['per_seed']['17'][m]
        r.update(value='17,29 (2/3)' if label else str(g['mean'][m]),
                 standard_deviation='' if label else str(g['sample_sd'][m]),
                 placeholder_kind='TXT' if label else 'PM',seed='17;29',
                 run_id=';'.join(x['run_id'] for x in g['runs']),
                 checkpoint=';'.join(Path(x['result_path']).with_name('best.pt').as_posix() for x in g['runs']),
                 aggregation=g['aggregation'],
                 notes='INTERIM_TWO_SEED: Complete seeds17/29, each containing all six target KGs. Mean and sample SD use these two seeds only; seed43 remains pending. This is not the registered three-seed final result; preserve this version when replacing it.')
        updates[cid]=(r,m)
    assert len(updates)==4 and all(r==before[r['cell_id']] for r in rows if r['cell_id'] not in updates)
    with (TABLES/'cells_results.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    fields,template=readcsv(TABLES/'cells_template.csv')
    for r in template:
        if r['cell_id'] in updates:r['placeholder_kind']=updates[r['cell_id']][0]['placeholder_kind']
    with (TABLES/'cells_template.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(template)
    approved=read(PAPER/'APPROVED_BASELINE_UPDATES.json');values=(TABLES/'values.tex').read_text(encoding='utf-8').splitlines()
    for cid,(r,m) in updates.items():
        values=[v for v in values if not v.startswith('\\SetResult{'+cid+'}')]
        shown=r['value'] if m=='paper_table' else r'\('+f"{100*float(r['value']):.2f}"+r'\mathbin{\pm}'+f"{100*float(r['standard_deviation']):.2f}"+r'\)'
        values.append('\\SetResult{'+cid+'}{'+shown+'}')
        approved['cells'][cid]=r
        approved['statistics_sources'][cid]={'path':(src/'INTERIM_TWO_SEED_RESULTS.json').relative_to(ROOT).as_posix(),'sha256':sha(src/'INTERIM_TWO_SEED_RESULTS.json'),'method':'SS-AGA','dataset':'depkg','metric':m,'status':'interim_two_seed'}
    (TABLES/'values.tex').write_text('\n'.join(values)+'\n',encoding='utf-8')
    path=TABLES/'tables/t03.tex';text=path.read_text(encoding='utf-8')
    for cid,(r,m) in updates.items():
        if m!='paper_table':text=text.replace('\\V{'+cid+'}','\\PM{'+cid+'}')
    old='SS-AGA on E-PKG reports seed 17 only (1/3); seeds 29/43 and the sample SD are pending.'
    new='SS-AGA on E-PKG reports two-seed mean $\\pm$ sample SD (17, 29; 2/3); seed 43 is pending.'
    assert text.count(old)==1;path.write_text(text.replace(old,new),encoding='utf-8')
    (rev/'published').mkdir();shutil.copy2(path,rev/'published/t03.tex')
    approved['table_sources']['tables/t03.tex']={'sha256':sha(path),'archived_source':(rev/'published/t03.tex').relative_to(ROOT).as_posix(),'previous_sha256':sha(rev/'before/tables/tables/t03.tex'),'authorization':'User requested all available results filled and transferred; pending third seed is explicitly identified','changed_cells':list(updates),'revision':rev.relative_to(ROOT).as_posix()}
    approved.setdefault('revisions',[]).append(rev.relative_to(ROOT).as_posix());save(PAPER/'APPROVED_BASELINE_UPDATES.json',approved)
    ranks=read(TABLES/'table2_highlights.json')['columns'];assert refresh_highlights()['columns']==ranks
    manifest=read(TABLES/'table_manifest.json')
    manifest.update(interim_cells=list(updates),pending_three_seed_cells=list(updates),all_registered_three_seed_groups_complete=False,latest_baseline_revision=rev.relative_to(ROOT).as_posix(),task_scope='Table2 and completed Table3 rows frozen; SS-AGA/E-PKG seeds17/29 interim, awaiting seed43')
    manifest['single_run_groups']=[]
    manifest['interim_groups']=[{'method':'SS-AGA','dataset':'depkg','seeds':[17,29],'planned_seeds':[17,29,43],'status':'interim_two_seed'}]
    comparison=read(BASE/'CURRENT_COMPARISON.json');comparison['csv_sha256']=sha(TABLES/'cells_results.csv')
    comparison['remaining_table3_cells_reason']='Two-seed interim values displayed; registered three-seed statistics await seed43'
    comparison['latest_interim_results_sha256']=sha(src/'INTERIM_TWO_SEED_RESULTS.json');save(BASE/'CURRENT_COMPARISON.json',comparison)
    manifest['current_comparison_sha256']=sha(BASE/'CURRENT_COMPARISON.json');save(TABLES/'table_manifest.json',manifest)
    numbers=' / '.join(f"{g['mean'][m]*100:.2f}±{g['sample_sd'][m]*100:.2f}" for m in ['mrr','h1','h10'])
    old='Table 3 的 SS-AGA/E-PKG 已完成 seed17 的六个 KG，单种子宏平均为 28.84 / 17.07 / 51.82（MRR / H@1 / H@10，%），当前明确标注为 1/3 的暂值，不提供跨种子标准差；正式三种子结果仍待原独立队列完成 seed29/43。'
    new=f'Table 3 的 SS-AGA/E-PKG 已完成 seed17/29，每个种子覆盖六个 KG；两种子均值±样本标准差为 {numbers}（MRR / H@1 / H@10，%），当前明确标注为 2/3 的暂统计。正式三种子结果仍待原独立队列完成 seed43。'
    idea=ROOT/'work/ideaspark/_run/multidomain-kgc-local/_2/phase4'
    for doc in [idea/'idea.std.zh.md',idea/'idea.std.zh.tex',TABLES/'EXPERIMENT_DESIGN.zh.md']:
        text=doc.read_text(encoding='utf-8');find=old.replace('%',r'\%') if doc.suffix=='.tex' else old
        assert text.count(find)==1
        text=text.replace(find,new.replace('%',r'\%') if doc.suffix=='.tex' else new)
        if doc.name=='EXPERIMENT_DESIGN.zh.md':
            text+='\n## 2026-09-07 SS-AGA/E-PKG 两种子更新\n\n'+new+' 服务器最新为 33/39 项完成、6 项 seed43 运行，无排队任务。恢复并修复本地回传程序的 SSH 重连与归档断点续传，不改服务器训练。语言汇总表 378 项及 152 个来源文件已复核，主表其余已完成结果保留。核验数据见 reproduction/language_table_seed17/main_completion_20260907/INTERIM_TWO_SEED_RESULTS.json。\n'
        doc.write_text(text,encoding='utf-8')
    legacy=ROOT/'work/ideaspark_run/multidomain-kgc-local_2/phase4'
    for ext in ['md','tex']:shutil.copy2(idea/f'idea.std.zh.{ext}',legacy/f'idea.std.zh.{ext}')
    audit={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'updated_cells':list(updates),'status':'interim_two_seed','three_seed_complete':False,'seeds':[17,29],'pending_seeds':[43],'all_prior_final_rows_unchanged':True,'previous_interim_preserved_in_archive':True,'table2_cells_and_ranks_unchanged':True,'new_training_or_inference':False,'summary_path':(src/'INTERIM_TWO_SEED_RESULTS.json').relative_to(ROOT).as_posix(),'summary_sha256':sha(src/'INTERIM_TWO_SEED_RESULTS.json'),'archive':rev.relative_to(ROOT).as_posix()}
    save(rev/'IMPORT_AUDIT.json',audit);save(PAPER/'LATEST_BASELINE_IMPORT.json',audit)
    print(g['display_percent'])


if __name__=='__main__':main()
