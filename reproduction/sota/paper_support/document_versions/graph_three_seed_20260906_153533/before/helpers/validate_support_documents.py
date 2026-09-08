"""CSV/source/full-precision/PDF consistency and preservation checks."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import pymupdf
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'reproduction/sota';phase=base/'paper_support';tables=ROOT/'outputs/kbs/_main/_tables'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
p=argparse.ArgumentParser();p.add_argument('--allow-pending',action='store_true');args=p.parse_args()
plan=json.loads((phase/'PLAN.json').read_text());archive=ROOT/plan['old_version_archive'];schema=json.loads((phase/'TABLE_SCHEMA.json').read_text())
rows=readcsv(tables/'cells_results.csv');old=readcsv(archive/'tables/cells_results.csv');lookup={r['cell_id']:r for r in rows}
approved_path=phase/'APPROVED_BASELINE_UPDATES.json'
approved_record=json.loads(approved_path.read_text()) if approved_path.exists() else {}
approved=approved_record.get('cells',{})
assert len(lookup)==len(rows)==schema['total_cells']
for r in old:
    if r['table_id'] in ['T1','T2','T3']:assert lookup[r['cell_id']]==approved.get(r['cell_id'],r),r['cell_id']
approved_sources=approved_record.get('table_sources',{})
assert set(approved_sources)<=set(f'tables/t{i:02}.tex' for i in [1,2,3])
for number in [1,2,3]:
    name=f'tables/t{number:02}.tex'
    if name in approved_sources:
        spec=approved_sources[name];source=ROOT/spec['archived_source']
        assert source.resolve().is_relative_to((phase/'document_versions').resolve())
        assert sha(tables/name)==sha(source)==spec['sha256']
    else:assert sha(tables/name)==sha(archive/'tables'/name)
ids={}
for path in sorted((tables/'tables').glob('t*.tex')):
    for kind,ident in re.findall(r'\\(PM|V|N|CI|TXT)\{([^}]+)\}',path.read_text(encoding='utf-8')):
        assert ident not in ids,ident;ids[ident]=kind
assert set(ids)==set(lookup)
for ident,kind in ids.items():assert lookup[ident]['placeholder_kind']==kind,(ident,kind)
values=dict(re.findall(r'^\\SetResult\{([^}]+)\}\{(.*)\}$',(tables/'values.tex').read_text(encoding='utf-8'),flags=re.M))
assert set(values)=={r['cell_id'] for r in rows if r['value']}
for cid,row in approved.items():
    shown=f"{100*float(row['value']):.2f}"
    if row['placeholder_kind']=='PM':
        assert row['seed']=='17;29;43' and float(row['standard_deviation'])>=0
        shown+=r'\mathbin{\pm}'+f"{100*float(row['standard_deviation']):.2f}"
        spec=approved_record['statistics_sources'][cid];source=ROOT/spec['path']
        assert source.resolve().is_relative_to((phase/'document_versions').resolve()) and sha(source)==spec['sha256']
        summary=json.loads(source.read_text());group=next(g for g in summary['groups'] if (g['method'],g['dataset'])==(spec['method'],spec['dataset']))
        assert group['status']=='complete' and group['seeds']==[17,29,43]
        assert float(row['value'])==group['mean'][spec['metric']]
        assert float(row['standard_deviation'])==group['sample_sd'][spec['metric']]
    else:assert row['placeholder_kind']=='V'
    assert values[cid]=='\\('+shown+'\\)',cid
support=[lookup[c['cell_id']] for c in schema['support_cells']]
if not args.allow_pending:
    assert all(r['value'] for r in support)
    result=json.loads((phase/'RESULTS.json').read_text());assert result['status']=='completed'
    rr={(t,r['key']):r for t,items in result['tables'].items() for r in items}
    for c in schema['support_cells']:
        row=lookup[c['cell_id']];raw=rr[(c['table_id'],c['row_key'])]
        if c['metric']=='queries':assert int(row['value'])==raw['queries']
        elif row['value']=='N/A':assert raw['queries']==0
        else:
            st=raw['stats'][c['metric']];assert float(row['value'])==st['mean']
            if row['placeholder_kind']=='PM':assert float(row['standard_deviation'])==st['sd']
    assert sha(base/'three_seed/RESULTS.json')==plan['table2_results_sha256']
pdfs=[tables/'KBS_Main_Text_Tables.pdf',ROOT/'work/ideaspark/_run/multidomain-kgc-local/_2/phase4/idea.std.zh.pdf'];pages={}
for pdf in pdfs:
    doc=pymupdf.open(pdf);text='\n'.join(p.get_text() for p in doc).replace('\u2212','-');assert '??' not in text
    for page in doc:
        for word in page.get_text('words'):
            assert word[0]>=-1 and word[1]>=-1 and word[2]<=page.rect.width+1 and word[3]<=page.rect.height+1,(pdf.name,word)
    if pdf==pdfs[0]:
        assert len(doc)==4
        for i in range(1,9):assert f'Table {i}' in text
        for number in ['74.88','65.78','89.29','56.41','43.99','76.17','63.70','54.43','78.58','51.52','41.92','70.63']:assert number in text
        for cid,row in approved.items():
            assert f"{100*float(row['value']):.2f}" in text,cid
            if row['placeholder_kind']=='PM':assert f"{100*float(row['standard_deviation']):.2f}" in text,cid
        if not args.allow_pending:
            for c in schema['support_cells']:
                r=lookup[c['cell_id']]
                if r['value']=='N/A':continue
                digits=0 if c['metric']=='qps' else (1 if c['metric'] in ['rss_mib','vram_mib'] else 2)
                token=str(int(r['value'])) if r['placeholder_kind']=='N' else f"{float(r['value']):.{digits}f}"
                assert token in text,(c['cell_id'],token)
    pages[pdf.relative_to(ROOT).as_posix()]={'pages':len(doc),'sha256':sha(pdf)}
for log in [tables/'build/main.log',pdfs[1].with_suffix('.log')]:
    lines=log.read_text(encoding='utf-8',errors='replace').splitlines()
    bad=[x for x in lines if any(s in x for s in ['Overfull','Missing character','Undefined control sequence','Float too large'])];assert not bad,bad
canonical=pdfs[1].parent;legacy=ROOT/'work/ideaspark_run/multidomain-kgc-local_2/phase4'
for ext in ['md','tex','pdf']:assert sha(canonical/f'idea.std.zh.{ext}')==sha(legacy/f'idea.std.zh.{ext}')
report={'passed':True,'support_final':not args.allow_pending,'total_cells':len(rows),'support_cells':len(support),
    'filled_support_cells':sum(bool(r['value']) for r in support),'all_prior_T1_T2_T3_cells_and_sources_unchanged':not bool(approved),
    'all_unapproved_T1_T2_T3_cells_and_table_sources_unchanged':True,'approved_baseline_cell_updates':list(approved),
    'approved_table_source_updates':list(approved_sources),'approved_three_seed_statistics_verified':len(approved_record.get('statistics_sources',{})),
    'pdfs':pages,'full_precision_csv_source_consistency':True,'canonical_legacy_idea_identical':True,
    'unfilled_external_cells':[r['cell_id'] for r in rows if not r['value'] and r['table_id'] in ['T1','T2','T3']]}
path=phase/('DESIGN_VALIDATION.json' if args.allow_pending else 'DOCUMENT_VALIDATION.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if not args.allow_pending:(tables/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
