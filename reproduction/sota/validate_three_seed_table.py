"""Verify three-seed Table2 values and prove every other cell stayed unchanged."""
import csv
import datetime
import hashlib
import json
import re
from pathlib import Path
import pymupdf
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'reproduction/sota';phase=base/'three_seed';tables=ROOT/'outputs/kbs/_main/_tables'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return {r['cell_id']:r for r in csv.DictReader(f)}
rows=read_csv(tables/'cells_results.csv');old=read_csv(base/'history/before_three_seed/tables/cells_results.csv');assert set(rows)==set(old)
r=json.loads((phase/'RESULTS.json').read_text());filled={};untouched=0
for cid,row in rows.items():
    if '.asrc.' not in cid:
        assert row==old[cid],cid;untouched+=1;continue
    assert cid.startswith('T2.');_,ds,_,metric=cid.split('.');ds={'dbp':'dbp5l','epkg':'depkg'}.get(ds,ds);d=r['datasets'][ds]
    assert float(row['value'])==d['mean'][metric] and float(row['standard_deviation'])==d['sample_sd'][metric]
    assert row['placeholder_kind']=='PM' and row['seed']=='17;29;43'
    filled[cid]=f"{d['mean'][metric]*100:.2f}±{d['sample_sd'][metric]*100:.2f}"
assert len(filled)==12
old_tables=base/'history/before_three_seed/tables/tables'
for path in (tables/'tables').glob('*.tex'):
    if path.name!='t02.tex':assert sha(path)==sha(old_tables/path.name)
values=(tables/'values.tex').read_text(encoding='utf-8')
for cid in filled:assert re.search(r'\\SetResult\{'+re.escape(cid)+r'\}\{',values)
pdf=tables/'KBS_Main_Text_Tables.pdf';doc=pymupdf.open(pdf);text='\n'.join(p.get_text() for p in doc);compact=re.sub(r'\s+','',text)
for formatted in filled.values():assert formatted in compact,formatted
assert 'ASRCreportsmean±sampleSDoverseeds17,29and43.' in compact
assert 'ASRCreportsonerun' not in compact and '??' not in text
layout=json.loads((tables/'validation_report.json').read_text());assert layout['passed'] and layout['pdf_sha256']==sha(pdf)
result={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'seed_order':[17,29,43],
    'asrc_mean_sd_cells_verified':12,'all_other_cells_unchanged':untouched,'other_table_sources_unchanged':7,
    'pdf_pages':len(doc),'pdf_sha256':sha(pdf),'pdf':'outputs/kbs/_main/_tables/KBS_Main_Text_Tables.pdf',
    'main_tex':'outputs/kbs/_main/_tables/main.tex','table_tex':'outputs/kbs/_main/_tables/tables/t02.tex',
    'results_sha256':sha(phase/'RESULTS.json'),'sample_sd_ddof':1,'layout_passed':True}
(phase/'TABLE_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
doc[0].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(phase/'table2_preview.png')
print(json.dumps(result,ensure_ascii=False))
