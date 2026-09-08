"""Export only the current table sources and the matching validated PDF."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]
TABLES=ROOT/'outputs/kbs_main_tables'
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'


def publish():
    package=ROOT/'outputs/KBS_Main_Text_Tables_LaTeX.zip'
    receipt=RUN/'results/table_package.json'
    names=['main.tex','preamble.tex','values.tex','cells_template.csv','cells_results.csv',
           'README.zh.md','build.ps1','validate_document.py','validate_current.py',
           'KBS_Main_Text_Tables.pdf']
    paths=[TABLES/name for name in names]+sorted((TABLES/'tables').glob('*.tex'))
    fingerprints={p.relative_to(TABLES).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old=json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {}
    if package.exists() and old.get('input_sha256')==fingerprints:
        return False
    validation=json.loads((TABLES/'validation_report.json').read_text(encoding='utf-8'))
    assert validation['passed'] and validation['pdf_sha256']==fingerprints['KBS_Main_Text_Tables.pdf']
    paths += [TABLES/'table_manifest.json',TABLES/'validation_report.json']
    temporary=package.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path,(Path('kbs_main_tables')/path.relative_to(TABLES)).as_posix())
    os.replace(temporary,package)
    report={'input_sha256':fingerprints,'files':len(paths),'pdf_sha256':validation['pdf_sha256'],
            'filled_cells':validation['filled_cells'],'package':str(package),
            'package_sha256':hashlib.sha256(package.read_bytes()).hexdigest()}
    temporary=receipt.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary,receipt)
    return True


if __name__=='__main__':
    print(json.dumps({'package_updated':publish()}))
