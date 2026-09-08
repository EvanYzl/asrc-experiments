"""Validate the current partially populated tables; experiment audit is separate."""
import argparse
import csv
import datetime
import hashlib
import json
import re
from pathlib import Path

import pypdfium2 as pdfium


def validate(render=False):
    root = Path(__file__).resolve().parent
    pdf = root / 'KBS_Main_Text_Tables.pdf'
    log = (root / 'build/main.log').read_text(encoding='utf-8', errors='replace')
    cells = {}
    sources = []
    for path in sorted((root / 'tables').glob('*.tex')):
        source = path.read_text(encoding='utf-8')
        sources.append(source)
        for kind, ident in re.findall(r'\\(PM|V|CI|N|TXT)\{([^}]+)\}', source):
            assert ident not in cells, 'Duplicate table cell: ' + ident
            cells[ident] = kind
    with (root / 'cells_template.csv').open(encoding='utf-8-sig', newline='') as f:
        template = list(csv.DictReader(f))
    with (root / 'cells_results.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert cells == {r['cell_id']: r['placeholder_kind'] for r in template}
    assert cells == {r['cell_id']: r['placeholder_kind'] for r in rows}
    filled = {r['cell_id']: r for r in rows if r['value'] != ''}
    values = re.findall(r'\\SetResult\{([^}]+)\}', (root / 'values.tex').read_text(encoding='utf-8'))
    assert len(values) == len(set(values)) and set(values) == set(filled)
    assert all(r['source_type'] == ('derived' if key.startswith('T1.') else 'rerun')
               for key, r in filled.items())
    assert all(r['source_type'] == 'pending' for r in rows if r['value'] == '')
    all_tex = '\n'.join(sources) + (root / 'preamble.tex').read_text(encoding='utf-8')
    monochrome = not re.search(r'\\(?:rowcolor|cellcolor|definecolor|textcolor)\b', all_tex)
    doc = pdfium.PdfDocument(pdf)
    captions, records, bounds = [], [], []
    previews = root / 'build/journal_previews'
    if render:
        previews.mkdir(exist_ok=True)
    for i, page in enumerate(doc):
        width, height = page.get_size()
        textpage = page.get_textpage()
        text = textpage.get_text_bounded()
        found = re.findall(r'(?m)^Table\s+(\d+)\s*$', text)
        captions.extend(found)
        for index in range(textpage.count_chars()):
            if not textpage.get_text_range(index, 1).strip():
                continue
            x0, y0, x1, y1 = textpage.get_charbox(index)
            if x0 < -1 or y0 < -1 or x1 > width + 1 or y1 > height + 1:
                bounds.append({'page': i + 1, 'character_index': index, 'bbox': [x0, y0, x1, y1]})
        records.append({'page': i + 1, 'captions': found, 'portrait': height > width,
                        'unresolved_ref': '??' in text, 'chars': len(text)})
        if render:
            page.render(scale=1.8).to_pil().save(previews / f'page-{i+1}.png')
        textpage.close()
        page.close()
    doc.close()
    warnings = [s for s in log.splitlines() if any(k in s for k in
                ('Overfull', 'Float too large', 'undefined references', 'Undefined control sequence',
                 'Missing character', 'Font Warning'))]
    passed = (captions == [str(i) for i in range(1, 9)] and not warnings and not bounds and monochrome
              and all(r['portrait'] and not r['unresolved_ref'] for r in records))
    report = {'purpose': 'Current PDF layout and table-cell consistency; raw experiment validation is in table_fill_audit.json',
              'updated_at': datetime.datetime.now().astimezone().isoformat(), 'passed': passed,
              'pages': len(records), 'main_tables': 8, 'total_cells': len(cells),
              'filled_cells': len(filled), 'placeholder_cells': len(cells) - len(filled),
              'all_results_unfilled': not filled, 'performance_source': 'local reruns only',
              'csv_matches_source': True, 'values_match_csv': True, 'monochrome_source': bool(monochrome),
              'caption_ids': captions, 'layout_warnings': warnings, 'out_of_page_characters': bounds,
              'page_records': records, 'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest()}
    destination = root / 'validation_report.json'
    temp = destination.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(destination)
    assert passed, 'PDF validation failed; see validation_report.json'
    print(json.dumps({k: report[k] for k in ['passed', 'pages', 'filled_cells', 'placeholder_cells']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    validate(parser.parse_args().render)
