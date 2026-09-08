#!/usr/bin/env python3
"""Enrich the downloaded literature pack and generate citation-manager files."""

from __future__ import annotations

import csv
import json
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from literature_pipeline import CONTACT, OUT, ROOT, crossref_record, request_json


MANUAL_AUTHORS = {
    "imkgc_appendix": ["Jiawei Sheng", "Taoyu Su", "Weiyi Yang", "Linghui Wang", "Yongxiu Xu", "Tingwen Liu"],
    "similarity_flooding": ["Zequn Sun", "Jiacheng Huang", "Xiaozhou Xu", "Qijin Chen", "Weijun Ren", "Wei Hu"],
    "incomplete_kga": ["Vinh Van Tong", "Thanh Trung Huynh", "Thanh Tam Nguyen", "Hongzhi Yin", "Quoc Viet Hung Nguyen", "Quyet Thang Huynh"],
    "astarnet": ["Zhaocheng Zhu", "Xinyu Yuan", "Mikhail Galkin", "Sophie Xhonneux", "Ming Zhang", "Maxime Gazeau", "Jian Tang"],
    "best_metric": ["Ashutosh Soni", "Peizhong Ju", "Atilla Eryilmaz", "Ness B. Shroff"],
    "graph_shapley_utility": ["Hongliang Chi", "Qiong Wu", "Zhengyi Zhou", "Yao Ma"],
}

MANUAL_VENUES = {
    "dmkgc": "Proceedings of the ACM Web Conference 2026",
    "efficient_multilingual_sharing": "Findings of the Association for Computational Linguistics: EMNLP 2025",
    "dual_spirals": "Findings of the Association for Computational Linguistics: ACL 2024",
    "daea": "Proceedings of the 31st International Conference on Computational Linguistics",
    "global_local_mkgc": "2024 4th International Symposium on Computer Technology and Information Science",
    "imkgc_appendix": "AAAI 2026 Supplementary Material",
    "similarity_flooding": "Proceedings of the 40th International Conference on Machine Learning",
    "incomplete_kga": "IEEE International Conference on Data Engineering",
    "astarnet": "Advances in Neural Information Processing Systems 36",
    "graph_shapley_utility": "The Thirteenth International Conference on Learning Representations",
    "transe": "Advances in Neural Information Processing Systems 26",
    "complex_jmlr": "Journal of Machine Learning Research",
    "msda_moe": "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    "msds": "IEEE Transactions on Knowledge and Data Engineering",
    "multisource_theory": "Advances in Neural Information Processing Systems 31",
    "open_world_eval": "Advances in Neural Information Processing Systems 35",
    "flock": "The Fourteenth International Conference on Learning Representations",
    "mhyper": "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics",
    "rank_metrics_framework": "Graph Learning Benchmarks Workshop 2022",
    "automated_source_selection": "Pattern Recognition",
    "mtranse": "Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence",
    "multiview_ea": "Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence",
    "fustkgc": "Neurocomputing",
    "noniid_graph_transfer": "Proceedings of the AAAI Conference on Artificial Intelligence",
}

MANUAL_ENTRY_TYPES = {
    "dmkgc": "inproceedings", "imkgc": "inproceedings", "efficient_multilingual_sharing": "inproceedings",
    "dual_spirals": "inproceedings", "daea": "inproceedings", "global_local_mkgc": "inproceedings",
    "similarity_flooding": "inproceedings", "incomplete_kga": "inproceedings", "astarnet": "inproceedings",
    "graph_shapley_utility": "inproceedings", "transe": "inproceedings", "complex_jmlr": "article",
    "msda_moe": "inproceedings", "msds": "article", "multisource_theory": "inproceedings",
    "open_world_eval": "inproceedings", "flock": "inproceedings", "mhyper": "inproceedings",
    "imkgc_appendix": "misc", "rank_metrics_framework": "inproceedings",
    "automated_source_selection": "article",
    "mtranse": "inproceedings",
    "multiview_ea": "inproceedings", "fustkgc": "article", "noniid_graph_transfer": "inproceedings",
}


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def crossref_authors(cr: dict[str, Any] | None) -> list[str]:
    out: list[str] = []
    for author in (cr or {}).get("author", []):
        name = " ".join(filter(None, [clean_text(author.get("given")), clean_text(author.get("family"))]))
        if name:
            out.append(name)
    return out


def openalex_authors(oa: dict[str, Any] | None) -> list[str]:
    return [
        a["author"]["display_name"]
        for a in (oa or {}).get("authorships", [])
        if isinstance(a, dict) and isinstance(a.get("author"), dict) and a["author"].get("display_name")
    ]


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    oa = None
    if row.get("openalex_id"):
        oa = request_json(f"{row['openalex_id']}?mailto={urllib.parse.quote(CONTACT)}")
    cr = None
    doi = row.get("doi") or ""
    if doi and not doi.lower().startswith("10.48550/arxiv"):
        cr = crossref_record(doi)

    authors = row.get("authors") or MANUAL_AUTHORS.get(row["key"]) or crossref_authors(cr) or openalex_authors(oa)
    venue = MANUAL_VENUES.get(row["key"]) or clean_text((cr or {}).get("container-title")) or clean_text(row.get("venue"))
    if not venue:
        venue = clean_text((((oa or {}).get("primary_location") or {}).get("source") or {}).get("display_name"))
    publisher = clean_text((cr or {}).get("publisher"))
    volume = clean_text((cr or {}).get("volume"))
    issue = clean_text((cr or {}).get("issue"))
    pages = clean_text((cr or {}).get("page"))
    publication_type = clean_text((cr or {}).get("type")) or clean_text((oa or {}).get("type"))
    publication_date = clean_text((oa or {}).get("publication_date"))
    if not publication_date and cr:
        parts = ((cr.get("published-print") or cr.get("published-online") or {}).get("date-parts") or [[]])[0]
        if parts:
            publication_date = "-".join(str(x).zfill(2) if i else str(x) for i, x in enumerate(parts))

    out = dict(row)
    out.update({
        "authors": authors or [],
        "venue": venue,
        "publisher": publisher,
        "volume": volume,
        "issue": issue,
        "pages": pages,
        "publication_type": publication_type,
        "publication_date": publication_date,
        "metadata_version": 2,
    })
    return out


def bib_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def entry_type(row: dict[str, Any]) -> str:
    if row["key"] in MANUAL_ENTRY_TYPES:
        return MANUAL_ENTRY_TYPES[row["key"]]
    kind = (row.get("publication_type") or "").lower()
    venue = (row.get("venue") or "").lower()
    if "proceedings" in kind or any(token in venue for token in ["conference", "proceedings", "neurips", "iclr", "ijcai", "emnlp", "acl "]):
        return "inproceedings"
    if "journal" in kind or (venue and "arxiv" not in venue and not row.get("arxiv")):
        return "article"
    return "misc"


def bib_entry(row: dict[str, Any], include_file: bool) -> str:
    kind = entry_type(row)
    fields: list[tuple[str, Any]] = [
        ("title", "{" + row["title"] + "}"),
        ("author", " and ".join(row.get("authors") or [])),
    ]
    if row.get("venue"):
        fields.append(("booktitle" if kind == "inproceedings" else "journal", row["venue"]))
    elif row.get("arxiv"):
        fields.append(("journal", f"arXiv preprint arXiv:{row['arxiv']}"))
    fields += [
        ("year", row.get("year")),
        ("volume", row.get("volume")),
        ("number", row.get("issue")),
        ("pages", row.get("pages")),
        ("publisher", row.get("publisher")),
        ("doi", row.get("doi")),
        ("eprint", row.get("arxiv")),
        ("archivePrefix", "arXiv" if row.get("arxiv") else None),
        ("url", row.get("landing_url") or row.get("source_url")),
        ("annote", row.get("relevance")),
    ]
    if include_file and row.get("local_path"):
        path = (ROOT / row["local_path"]).as_posix()
        fields.append(("file", path))
    body = [f"  {key} = {{{bib_escape(str(value))}}}" for key, value in fields if value not in (None, "", [])]
    return f"@{kind}{{{row['key']},\n" + ",\n".join(body) + "\n}"


def ris_entry(row: dict[str, Any]) -> str:
    kind = entry_type(row)
    ty = {"inproceedings": "CPAPER", "article": "JOUR", "misc": "PREPRINT"}[kind]
    lines = [f"TY  - {ty}", f"ID  - {row['key']}", f"TI  - {row['title']}"]
    lines += [f"AU  - {author}" for author in row.get("authors") or []]
    for tag, key in [("PY", "year"), ("T2", "venue"), ("VL", "volume"), ("IS", "issue"), ("SP", "pages"), ("DO", "doi"), ("UR", "landing_url"), ("N1", "relevance")]:
        if row.get(key):
            lines.append(f"{tag}  - {row[key]}")
    if row.get("local_path"):
        lines.append(f"L1  - {(ROOT / row['local_path']).as_posix()}")
    lines.append("ER  -")
    return "\n".join(lines)


def main() -> int:
    manifest_path = OUT / "literature_manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    enriched: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(enrich, row): row["key"] for row in rows}
        for future in as_completed(futures):
            key = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                item = next(dict(r) for r in rows if r["key"] == key)
                item["metadata_error"] = f"{type(exc).__name__}: {exc}"
            enriched.append(item)
            print(f"[metadata] {key}", flush=True)
    enriched.sort(key=lambda r: (r["theme"], 0 if r["importance"] == "core" else 1, -(r.get("year") or 0), r["title"]))
    manifest_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "references.bib").write_text("\n\n".join(bib_entry(r, False) for r in enriched) + "\n", encoding="utf-8")
    (OUT / "references_zotero.bib").write_text("\n\n".join(bib_entry(r, True) for r in enriched) + "\n", encoding="utf-8")
    (OUT / "references.ris").write_text("\n\n".join(ris_entry(r) for r in enriched) + "\n", encoding="utf-8")

    columns = ["key", "theme", "importance", "title", "authors", "year", "publication_date", "venue", "publisher", "volume", "issue", "pages", "publication_type", "doi", "arxiv", "status", "local_path", "size_bytes", "sha256", "oa_status", "landing_url", "source_url", "relevance", "error"]
    with (OUT / "literature_catalog.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in enriched:
            out = {key: row.get(key, "") for key in columns}
            out["authors"] = "; ".join(row.get("authors") or [])
            writer.writerow(out)

    no_authors = [r for r in enriched if not r.get("authors")]
    no_venue = [r for r in enriched if not r.get("venue") and not r.get("arxiv")]
    report = [
        "# 引用元数据质量报告", "",
        f"- 总条目：{len(enriched)}", f"- 缺作者：{len(no_authors)}", f"- 缺正式 venue 且无 arXiv：{len(no_venue)}", "",
        "BibTeX 和 RIS 已生成；含 DOI 的条目建议导入 Zotero 后再执行一次“从 DOI 更新元数据”。", "",
    ]
    if no_authors:
        report += ["## 缺作者", ""] + [f"- `{r['key']}` — {r['title']}" for r in no_authors] + [""]
    if no_venue:
        report += ["## 缺 venue", ""] + [f"- `{r['key']}` — {r['title']}" for r in no_venue] + [""]
    (OUT / "citation_metadata_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"DONE: {len(enriched)} records; missing authors={len(no_authors)}; missing venue={len(no_venue)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
