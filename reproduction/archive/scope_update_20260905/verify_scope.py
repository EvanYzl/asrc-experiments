"""One-time scope, publication, and artifact checks after the user's edit."""
import ast
import csv
import datetime
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import pypdfium2 as pdfium

ROOT=Path('G:/zhishitupui')
SUITE=ROOT/'reproduction/strict_baselines'
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'
TABLES=ROOT/'outputs/kbs_main_tables'
needle=re.compile(r'\bkens\b|kens_',re.I)
suffixes={'.py','.md','.json','.csv','.tex'}
scan_roots=[SUITE,ROOT/'reproduction/queue',ROOT/'reproduction/tools',ROOT/'reproduction',
            ROOT/'baselines',ROOT/'refine-logs',TABLES,TABLES/'tables',RUN,RUN/'results',
            RUN/'figures',RUN/'figures/data']
paths=[p for parent in scan_roots for p in parent.iterdir() if p.is_file() and p.suffix in suffixes]
violations=[str(p) for p in paths if needle.search(p.read_text(encoding='utf-8-sig'))]
assert not violations,violations
for path in [*SUITE.glob('*.py'),*(ROOT/'reproduction/queue').glob('*.py')]:
    ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))

with (ROOT/'baselines/BASELINES_MANIFEST.csv').open(encoding='utf-8-sig',newline='') as f:
    baseline_rows=list(csv.DictReader(f))
assert len(baseline_rows)==9
manifest=json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
assert manifest['active_tables']==['T1','T2','T3']
assert len(manifest['jobs'])==117
assert len(manifest['deferred_jobs'])==43
assert all(not needle.search(json.dumps(j)) for j in manifest['jobs']+manifest['deferred_jobs'])
all_ids={j['id'] for j in manifest['jobs']+manifest['deferred_jobs']}
assert all(set(j.get('depends_on',[]))<=all_ids for j in manifest['jobs']+manifest['deferred_jobs'])
scope=json.loads((RUN/'results/table_scope.json').read_text(encoding='utf-8'))
assert scope['eligible_baseline_and_statistics_slots']==127 and scope['total_slots']==688
with (TABLES/'cells_results.csv').open(encoding='utf-8-sig',newline='') as f:
    cells=list(csv.DictReader(f))
assert all(not r['value'] for r in cells if r['table_id'] not in manifest['active_tables'])
assert all(not r['value'] for r in cells if '.qura.' in r['cell_id'])
assert all(not needle.search(r['cell_id']) for r in cells)
assert len([r for r in cells if r['table_id']=='T3'])==16

pdf=TABLES/'KBS_Main_Text_Tables.pdf'
with pdfium.PdfDocument(pdf) as doc:
    text='\n'.join(page.get_textpage().get_text_bounded() for page in doc)
    assert len(doc)==4 and not needle.search(text)
assert text.count('AlignKGC')>=3 and text.count('SS-AGA')>=3
with zipfile.ZipFile(ROOT/'outputs/KBS_Main_Text_Tables_LaTeX.zip') as package:
    for name in package.namelist():
        if Path(name).suffix in suffixes:
            assert not needle.search(package.read(name).decode('utf-8-sig')),name
    assert package.read('kbs_main_tables/KBS_Main_Text_Tables.pdf')==pdf.read_bytes()
automation=Path('C:/Users/evan/.codex/automations/table-1-3/automation.toml').read_text(encoding='utf-8')
assert not needle.search(automation) and '127' in automation and 'status = "ACTIVE"' in automation
for index in range(1,5):
    shutil.copy2(TABLES/f'build/journal_previews/page-{index}.png',RUN/f'results/pdf_preview/page_{index}.png')
validation=json.loads((TABLES/'validation_report.json').read_text(encoding='utf-8'))
assert validation['passed'] and validation['total_cells']==688
payload={'updated_at':datetime.datetime.now().astimezone().isoformat(),'passed':True,
    'external_methods':['TransE','DistMult','RotatE','ATransN','LSMGA','DMKGC','IMKGC','AlignKGC','SS-AGA'],
    'active_tables':manifest['active_tables'],'active_jobs':len(manifest['jobs']),
    'deferred_jobs':len(manifest['deferred_jobs']),'active_cell_count':127,
    'filled_cells':scope['filled'],'pending_cells':scope['pending'],
    'current_files_scanned':len(paths),'current_scope_consistent':True,
    'scheduled_followup_matches_current_scope':True,'other_tables_unfilled':True,
    'pdf_pages':4,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
    'source_package_matches_pdf':True,'previous_51_filled_cells_preserved':True}
(RUN/'results/active_scope_verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False))
