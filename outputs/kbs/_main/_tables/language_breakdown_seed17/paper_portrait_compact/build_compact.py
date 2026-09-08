"""Portrait edition retaining the landscape table's row/column structure."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
snapshot = HERE / "original_table_snapshot.tex"
if not snapshot.exists():
    protected = [
        "main.tex", "table_language_kg_seed17.tex",
        "All_Datasets_Language_KG_Seed17.pdf",
        "All_Datasets_Language_KG_Seed17_Portrait.pdf",
    ]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_files": {
            name: hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest()
            for name in protected
        },
    }
    snapshot.write_bytes((ORIGINAL / "table_language_kg_seed17.tex").read_bytes())
    (HERE / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

original_table = snapshot.read_text(encoding="utf-8")
table = original_table.replace(r"\setlength{\tabcolsep}{2pt}", r"\setlength{\tabcolsep}{0.75pt}")
table = table.replace(r"L{33mm}", r"L{22mm}")
table = table.replace(r"\fontsize{8.2}{9.2}", r"\fontsize{8.2}{10.2}")
assert original_table.count(r"L{33mm}") == table.count(r"L{22mm}") == 4
assert re.findall(r"\b\d+\.\d{2}\b", original_table) == [
    x for x in re.findall(r"\b\d+\.\d{2}\b", table.replace("{0.75pt}", "{2pt}"))
]

main = r"""% !TEX program = xelatex
% Separate compact portrait layout; preserves the original landscape table structure.
\RequirePackage{fix-cm}
\documentclass[10pt,a4paper]{article}
\usepackage[left=15mm,right=15mm,top=17mm,bottom=17mm]{geometry}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\usepackage{booktabs,array,tabularx,caption}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={Single-seed language and KG results: compact portrait edition},pdfauthor={}}
\captionsetup[table]{font=small,labelfont=bf,labelsep=period,justification=raggedright,singlelinecheck=false,skip=4pt}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\textfloatsep}{0pt}
\begin{document}
\input{table_language_kg_seed17.tex}
\end{document}
"""
(HERE / "main.tex").write_text(main, encoding="utf-8")
(HERE / "table_language_kg_seed17.tex").write_text(table, encoding="utf-8")
print(f"Created compact portrait sources: {HERE}")
