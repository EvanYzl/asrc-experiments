"""Verify final document numbers, comparator retention, and saved PDF layout."""
import csv
import datetime
import hashlib
import json
import re
from pathlib import Path
import pymupdf
ROOT=Path(__file__).resolve().parents[2]
def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic_json(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)
base=ROOT/'reproduction/sota';tables=ROOT/'outputs/kbs/_main/_tables'
report=json.loads((base/'SINGLE_SEED_ACCEPTANCE.json').read_text())
refs=json.loads((base/'SINGLE_SEED_REFERENCES.json').read_text())
with (tables/'cells_results.csv').open(encoding='utf-8-sig',newline='') as f:rows={x['cell_id']:x for x in csv.DictReader(f)}
retained=[]
for cid,old in refs['original_comparison_cells'].items():
    assert cid in rows;row=rows[cid]
    if old['status']=='local_reproduction':
        assert float(row['value'])==old['value']
        for key in ['standard_deviation','seed','filter_protocol','aggregation','run_id']:assert row[key]==old[key]
    else:assert row['value']=='' and row['source_type']=='pending'
    retained.append(cid)
assert len(retained)==75
for ds,d in report['datasets'].items():
    prefix={'dbp5l':'dbp','depkg':'epkg'}.get(ds,ds)
    for metric in ['mrr','h1','h10']:
        row=rows[f'T2.{prefix}.asrc.{metric}']
        assert float(row['value'])==d['macro'][metric] and row['seed']=='17' and row['standard_deviation']==''
canonical=ROOT/'work/ideaspark/_run/multidomain-kgc-local/_2/phase4'
legacy=ROOT/'work/ideaspark_run/multidomain-kgc-local_2/phase4'
for ext in ['md','tex','pdf']:assert sha256(canonical/f'idea.std.zh.{ext}')==sha256(legacy/f'idea.std.zh.{ext}')
assert (base/'history/20260906T042743Z/phase4/idea.std.zh.pdf').exists()
records=[]
for pdf in [tables/'KBS_Main_Text_Tables.pdf',tables/'Table_2_SOTA_single_seed.pdf',canonical/'idea.std.zh.pdf']:
    doc=pymupdf.open(pdf);text='\n'.join(p.get_text() for p in doc)
    for ds,d in report['datasets'].items():
        for key in ['mrr','h1','h10']:assert f"{d['macro'][key]*100:.2f}" in text
    assert '??' not in text
    for page in doc:
        for word in page.get_text('words'):
            x0,y0,x1,y1=word[:4];assert x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1
    records.append({'path':pdf.relative_to(ROOT).as_posix(),'pages':len(doc),'sha256':sha256(pdf)})
for log in [canonical/'idea.std.zh.log',tables/'Table_2_SOTA_single_seed.log']:
    assert log.exists()
    bad=[line for line in log.read_text(encoding='utf-8',errors='replace').splitlines() if any(x in line for x in ['Overfull','Float too large','Missing character','Undefined control sequence'])]
    assert not bad,bad
layout=json.loads((tables/'validation_report.json').read_text());assert layout['passed'] and layout['total_cells']==len(rows)
assert layout['pdf_sha256']==sha256(tables/'KBS_Main_Text_Tables.pdf')
out={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'external_comparison_cells_retained':75,
     'candidate_cells_match_raw_acceptance':12,'single_seed':17,'pdfs':records,'original_versions_preserved':True,
     'canonical_legacy_idea_hashes_match':True,'pdf_numeric_text_and_page_bounds_valid':True,'main_tables_schema_layout_passed':True}
atomic_json(base/'DOCUMENT_DELIVERY_VALIDATION.json',out)
print(json.dumps(out,ensure_ascii=False))
