"""Create a separate portrait layout from a frozen copy of the original table."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
SNAPSHOT = HERE / "original_table_snapshot.tex"
NUMBER = re.compile(r"\b\d+\.\d{2}\b")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if not SNAPSHOT.exists():
    originals = ["main.tex", "table_language_kg_seed17.tex", "All_Datasets_Language_KG_Seed17.pdf"]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Layout-only portrait edition; original files are read-only inputs.",
        "original_files": {name: digest(SOURCE / name) for name in originals},
    }
    SNAPSHOT.write_bytes((SOURCE / "table_language_kg_seed17.tex").read_bytes())
    (HERE / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

source = SNAPSHOT.read_text(encoding="utf-8")
blocks = re.findall(r"\\begin\{tabularx\}.*?\\end\{tabularx\}", source, re.S)
specs = [
    ("DBP-5L", ["EL", "EN", "ES", "FR", "JA"], 7),
    ("E-PKG", ["DE", "ES", "FR", "IT", "JP", "UK"], 7),
    ("DWY", ["DBpedia", "Wikidata", "YAGO"], 7),
    ("WK3l-15k", [r"EN\_F", "FR"], 9),
]
assert len(blocks) == len(specs)

tables = []
original_cells = []
reflowed_cells = []
structure = []
for index, (block, (dataset, languages, method_count)) in enumerate(zip(blocks, specs), 1):
    rows = []
    for line in block.splitlines():
        line = line.strip()
        if NUMBER.search(line) and line.endswith(r"\\"):
            cells = [part.strip() for part in line[:-2].split("&")]
            assert len(cells) == 2 + 3 * len(languages), (dataset, cells)
            rows.append(cells)
    assert len(rows) == method_count
    title = (
        f"Language-level results on {dataset} (\\%, seed 17)."
        if dataset != "DWY" else "Knowledge-graph-level results on DWY (\\%, seed 17)."
    )
    avg_header = r"\shortstack{AVG MRR\\(FR only)}" if dataset == "WK3l-15k" else "AVG MRR"
    headers = ["Method", "Metric"] + languages + [avg_header]
    table = [
        r"\begin{minipage}{\linewidth}",
        r"\captionof{table}{" + title + "}",
        r"\label{tab:portrait-language-" + str(index) + "}",
        r"\fontsize{9}{10.2}\selectfont",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{tabularx}{\linewidth}{@{}L{28mm}L{13mm}*{" + str(len(languages) + 1) + r"}{C}@{}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row_index, cells in enumerate(rows):
        method, values, average = cells[0], cells[1:-1], cells[-1]
        original_cells.extend(cells[1:])
        if row_index:
            table.append(r"\midrule" if "ASRC" in method else r"\addlinespace[2pt]")
        for metric_index, metric in enumerate(["H@1", "H@10", "MRR"]):
            value_cells = [values[3 * language_index + metric_index] for language_index in range(len(languages))]
            reflowed_cells.extend(value_cells)
            method_cell = r"\multirow{3}{*}{" + method + "}" if metric_index == 0 else ""
            average_cell = r"\multirow{3}{*}{" + average + "}" if metric_index == 0 else ""
            if metric_index == 0:
                reflowed_cells.append(average)
            table.append(" & ".join([method_cell, metric] + value_cells + [average_cell]) + r" \\")
    table.extend([r"\bottomrule", r"\end{tabularx}", r"\end{minipage}"])
    tables.append("\n".join(table))
    structure.append({"dataset": dataset, "languages": languages, "methods": method_count})

# Preserve every displayed result and its existing best/second-best markup.
assert len(original_cells) == 378
assert Counter(original_cells) == Counter(reflowed_cells)

notes_page_1 = r"""\begin{tablenotesblock}
\textit{Notes.} Seed 17 only, with no SD. All scores are percentages. Bold and underline mark the best and second-best available values for each KG and metric, and for AVG MRR, ranked before rounding. AVG is the equal-KG macro MRR computed before rounding. JP denotes Japanese in E-PKG. Single-seed values may differ from the main table's three-seed means.
\par\smallskip
\textit{Evaluation and availability.} Full-candidate filtered tail ranking uses ascending entity-ID ties; filtering is train+valid for core datasets and all for WK3l. All displayed results are complete seed-17 runs, verified against saved ranks. Missing baselines were trained with fixed recipes and validation-only checkpoint selection; existing ASRC and baseline results were reused.
\end{tablenotesblock}"""

notes_page_2 = r"""\begin{tablenotesblock}
\textit{Notes.} All scores are percentages from seed 17. Bold and underline denote the best and second-best available values for each KG and metric, and for AVG MRR, ranked before rounding. EN\_F scores are supplemental; FR is the original primary target, and WK3l AVG remains FR-only.
\par\smallskip
\textit{Coverage.} ASRC reuses its FR-validation-selected checkpoint. TransE (teacher) uses the source-teacher recipe (margin 4, batch 1024) in both languages. English ATransN uses FR-to-EN transfer. Graph methods use a WK3l data-format adaptation with fixed recipes; one FR-validation-selected checkpoint serves both languages.
\par\smallskip
\textit{Split audit.} The original EN\_F test has 40,700 queries: 31 occur in training and 15 in validation. Original-split scores are shown; diagnostics on the remaining 40,654 queries are supplied separately. The original split and filtering sets are unchanged.
\end{tablenotesblock}"""

table_text = "\n\n".join([
    tables[0], r"\par\vspace{7mm}", tables[1], notes_page_1,
    r"\newpage", tables[2], r"\par\vspace{7mm}", tables[3], notes_page_2,
]) + "\n"

main = r"""% !TEX program = xelatex
% Separate portrait edition. The landscape source and PDF are not modified.
\documentclass[10pt,a4paper]{article}
\usepackage[left=18mm,right=18mm,top=17mm,bottom=17mm,footskip=9mm]{geometry}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\usepackage{booktabs,array,tabularx,multirow,caption}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={Language and KG results (seed 17): portrait paper edition},pdfauthor={}}
\captionsetup[table]{font=small,labelfont=bf,labelsep=period,justification=raggedright,singlelinecheck=false,skip=5pt,hypcap=false}
\renewcommand{\thetable}{S\arabic{table}}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\newenvironment{tablenotesblock}{\par\vspace{4pt}\begin{minipage}{\linewidth}\fontsize{8.5}{10}\selectfont}{\end{minipage}\par}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}
\pagestyle{plain}
\raggedbottom
\begin{document}
\input{tables_portrait.tex}
\end{document}
"""

(HERE / "main.tex").write_text(main, encoding="utf-8")
(HERE / "tables_portrait.tex").write_text(table_text, encoding="utf-8")
(HERE / "LAYOUT_CHECK.json").write_text(json.dumps({
    "numeric_cell_count": len(original_cells),
    "all_styled_cells_preserved": Counter(original_cells) == Counter(reflowed_cells),
    "original_bold_numeric_cells": sum(cell.startswith(r"\textbf{") for cell in original_cells),
    "original_underlined_numeric_cells": sum(cell.startswith(r"\underline{") for cell in original_cells),
    "dataset_structure": structure,
    "layout": "A4 portrait; metrics as rows; four supplementary tables across two pages",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Created portrait sources in {HERE}; preserved {len(original_cells)} styled numeric cells.")
