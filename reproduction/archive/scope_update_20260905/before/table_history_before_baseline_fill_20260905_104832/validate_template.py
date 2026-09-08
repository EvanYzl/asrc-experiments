"""Check the monochrome table-only preview; this does not validate experiments."""
import csv, hashlib, json, re
from pathlib import Path
import pymupdf
root = Path(__file__).resolve().parent
pdf = root / "build" / "main.pdf"
log = (root / "build" / "main.log").read_text(encoding="utf-8", errors="replace")
cell_map = {}
sources = []
for path in sorted((root / "tables").glob("*.tex")):
    source = path.read_text(encoding="utf-8")
    sources.append(source)
    for kind, ident in re.findall(r"\\(PM|V|CI|N|TXT)\{([^}]+)\}", source):
        assert ident not in cell_map, "Duplicate cell ID " + ident
        cell_map[ident] = kind
with (root / "cells_template.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
assert len(cell_map) == len(rows) == 694
assert cell_map == {r["cell_id"]:r["placeholder_kind"] for r in rows}
assert all(not any(r[k] for k in ("value","standard_deviation","ci_lower","ci_upper")) for r in rows)
active_values = "\n".join(s for s in (root / "values.tex").read_text(encoding="utf-8").splitlines() if not s.lstrip().startswith("%"))
assert "\\SetResult{" not in active_values
all_tex = "\n".join(sources) + (root/"preamble.tex").read_text(encoding="utf-8")
assert not re.search(r"\\(?:rowcolor|cellcolor|definecolor|textcolor)\b|\\ref\{tab:A|\\appendix", all_tex)
doc = pymupdf.open(pdf)
previews = root / "build" / "journal_previews"
previews.mkdir(exist_ok=True)
captions, records, nonblack, bounds = [], [], [], []
for i, page in enumerate(doc):
    text = page.get_text()
    found = re.findall(r"(?m)^Table\s+(\d+)\s*$", text)
    captions.extend(found)
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["color"] != 0:
                    nonblack.append({"page":i+1,"color":span["color"]})
                x0,y0,x1,y1 = span["bbox"]
                if x0 < -1 or y0 < -1 or x1 > page.rect.width+1 or y1 > page.rect.height+1:
                    bounds.append({"page":i+1,"bbox":span["bbox"]})
    records.append({"page":i+1,"captions":found,"portrait":page.rect.height>page.rect.width,
                    "unresolved_ref":"??" in text, "chars":len(text)})
    page.get_pixmap(matrix=pymupdf.Matrix(1.8,1.8),alpha=False).save(previews/f"page-{i+1}.png")
warnings = [s for s in log.splitlines() if any(k in s for k in
            ("Overfull", "Float too large", "undefined references", "Undefined control sequence",
             "Missing character", "Font Warning"))]
passed = (captions==[str(i) for i in range(1,9)] and not warnings and not nonblack and not bounds
          and all(r["portrait"] and not r["unresolved_ref"] for r in records))
report = {"purpose":"Document layout/template validation only", "passed":passed,
          "pages":len(doc),"main_tables":8,"appendix_tables":0,"cover_pages":0,
          "style":"Monochrome booktabs; elsarticle 5p table-only typeset preview",
          "placeholder_cells":len(cell_map),"all_results_unfilled":True,"csv_matches_source":True,
          "caption_ids":captions,"layout_warnings":warnings,"nonblack_text_spans":nonblack,
          "out_of_page_spans":bounds,"page_records":records,
          "pdf_sha256":hashlib.sha256(pdf.read_bytes()).hexdigest()}
(root/"validation_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2,ensure_ascii=False))
assert passed
