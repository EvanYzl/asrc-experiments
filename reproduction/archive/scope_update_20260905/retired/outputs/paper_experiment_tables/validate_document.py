"""Validate template structure and render the compiled PDF for layout review."""
from __future__ import annotations
import csv
import hashlib
import json
import re
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
pdf_path = root / "build" / "main.pdf"
manifest = json.loads((root / "table_manifest.json").read_text(encoding="utf-8"))
log = (root / "build" / "main.log").read_text(encoding="utf-8", errors="replace")
source_cells = {}
for path in sorted((root / "tables").glob("*.tex")):
    for kind, cell_id in re.findall(r"\\(PM|V|CI|N|TXT)\{([^}]+)\}", path.read_text(encoding="utf-8")):
        assert cell_id not in source_cells, f"Duplicate cell ID: {cell_id}"
        source_cells[cell_id] = kind
with (root / "cells_template.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
csv_cells = {r["cell_id"]: r["placeholder_kind"] for r in rows}
assert len(csv_cells) == len(rows)
assert source_cells == csv_cells, "CSV/source placeholder mismatch"
assert all(not r["value"] and not r["standard_deviation"] and not r["ci_lower"] and not r["ci_upper"] for r in rows)
active_values = "\n".join(line for line in (root / "values.tex").read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("%"))
assert r"\SetResult{" not in active_values, "An experimental value has been filled"
render_dir = root / "build" / "previews"
render_dir.mkdir(parents=True, exist_ok=True)
doc = pymupdf.open(pdf_path)
page_records = []
captions = []
thumbnails = []
for i, page in enumerate(doc):
    text = page.get_text()
    captions += re.findall(r"Table (A?\d+):", text)
    words = page.get_text("words")
    body = [w for w in words if w[1] > 42 and w[3] < page.rect.height - 18]
    low_words = [w[4] for w in body if w[3] > page.rect.height - 43]
    page_records.append({
        "page": i + 1, "width_pt": round(page.rect.width, 2),
        "height_pt": round(page.rect.height, 2), "text_chars": len(text),
        "captions": re.findall(r"Table (A?\d+):", text),
        "reading_rule_present": "Reading rule" in text,
        "low_body_words": low_words[:30],
        "unresolved_ref_token": "??" in text,
    })
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
    png = render_dir / f"page-{i+1:02d}.png"
    pix.save(png)
    thumb = Image.open(png).convert("RGB")
    thumb.thumbnail((640, 454))
    tile = Image.new("RGB", (660, 484), "#e5e9ed")
    tile.paste(thumb, ((660 - thumb.width)//2, 24))
    ImageDraw.Draw(tile).text((12, 7), f"Page {i+1}", fill="#18344b")
    thumbnails.append(tile)
for first in range(0, len(thumbnails), 8):
    batch = thumbnails[first:first+8]
    sheet = Image.new("RGB", (1320, 484*((len(batch)+1)//2)), "white")
    for j, im in enumerate(batch):
        sheet.paste(im, ((j%2)*660, (j//2)*484))
    sheet.save(render_dir / f"contact-{first//8+1:02d}.png")
expected = [str(i) for i in range(1, 7)] + [f"A{i}" for i in range(1, 20)]
warnings = [line for line in log.splitlines() if any(s in line for s in ["Overfull", "Float too large", "Undefined control sequence", "undefined references", "Missing character", "Font Warning", "Package xeCJK Warning"])]
report = {
    "purpose": "Template/document validation only; not an experiment result audit",
    "pdf_path": str(pdf_path),
    "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
    "pages": len(doc), "expected_tables": len(expected),
    "observed_caption_ids": captions,
    "missing_captions": sorted(set(expected)-set(captions)),
    "placeholder_cells": len(source_cells),
    "csv_matches_source": True, "all_experiment_results_unfilled": True,
    "tex_layout_warnings": warnings,
    "pages_with_unresolved_refs": [p["page"] for p in page_records if p["unresolved_ref_token"]],
    "page_records": page_records,
}
report["passed"] = (not warnings and not report["missing_captions"] and not report["pages_with_unresolved_refs"])
(root / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="page_records"}, indent=2, ensure_ascii=False))
print("Low body text:", [(p["page"], p["low_body_words"]) for p in page_records if p["low_body_words"]])

