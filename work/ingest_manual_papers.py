#!/usr/bin/env python3
"""Ingest user-supplied PDFs into the curated literature pack.

The source files are copied (not moved), validated by PDF signature and size,
matched to manifest keys explicitly, and all local citation indexes are rebuilt
without making network requests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import enrich_literature_metadata as citations
import literature_pipeline as pipeline


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "literature"
MANIFEST = OUT / "literature_manifest.json"

SOURCE_BY_KEY = {
    "global_local_mkgc": Path(
        r"C:\Users\evan\Downloads\Multilingual_Knowledge_Graph_Completion_based_on_Global-Local_Structure_Encoding.pdf"
    ),
    "fustkgc": Path(r"C:\Users\evan\Downloads\1-s2.0-S0925231226000871-main.pdf"),
    "kgc_model_calibration": Path(r"C:\Users\evan\Downloads\1179_Using_Model_Calibration_t.pdf"),
    "sim_fewshot": Path(r"C:\Users\evan\Downloads\3589557.pdf"),
    "kgc_eval_protocol": Path(r"C:\Users\evan\Downloads\3442381.3449856.pdf"),
    "msds": Path(
        r"C:\Users\evan\Downloads\MSDS_A_Novel_Framework_for_Multi-Source_Data_Selection_Based_Cross-Network_Node_Classification.pdf"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pdf(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size < 10_000:
        raise ValueError(f"PDF is unexpectedly small: {path}")
    with path.open("rb") as handle:
        if b"%PDF-" not in handle.read(1024):
            raise ValueError(f"PDF signature not found: {path}")


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row["theme"],
            0 if row["importance"] == "core" else 1,
            -(row.get("year") or 0),
            row["title"],
        ),
    )


def rebuild_indexes(rows: list[dict[str, Any]]) -> None:
    rows = sort_rows(rows)
    MANIFEST.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT / "references.bib").write_text(
        "\n\n".join(citations.bib_entry(row, False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (OUT / "references_zotero.bib").write_text(
        "\n\n".join(citations.bib_entry(row, True) for row in rows) + "\n",
        encoding="utf-8",
    )
    (OUT / "references.ris").write_text(
        "\n\n".join(citations.ris_entry(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    columns = [
        "key", "theme", "importance", "title", "authors", "year",
        "publication_date", "venue", "publisher", "volume", "issue", "pages",
        "publication_type", "doi", "arxiv", "status", "local_path", "size_bytes",
        "sha256", "oa_status", "landing_url", "source_url", "relevance", "error",
    ]
    with (OUT / "literature_catalog.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            item = {key: row.get(key, "") for key in columns}
            item["authors"] = "; ".join(row.get("authors") or [])
            writer.writerow(item)

    downloaded = [row for row in rows if row.get("status") == "downloaded"]
    missing = [row for row in rows if row.get("status") != "downloaded"]
    core_missing = [row for row in missing if row.get("importance") == "core"]

    checksum_lines = [
        f"{row['sha256']}  {str(row['local_path']).replace(os.sep, '/')}"
        for row in downloaded
    ]
    (OUT / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    manual_lines = [
        "# 需要手动下载的文献",
        "",
        "下列条目没有找到可直接、合法访问的公开 PDF。优先打开 DOI/论文主页；若你通过学校或机构有访问权限，可下载后交给本项目归档。",
        "",
        f"共 {len(missing)} 篇，其中核心文献 {len(core_missing)} 篇。",
        "",
    ]
    for row in missing:
        manual_lines.extend(
            [
                f"## {row.get('year') or '年份未知'} · {row['title']}",
                "",
                f"- 级别：{'核心' if row['importance'] == 'core' else '补充'}",
                f"- 主题：`{row['theme']}`",
                f"- DOI：{('https://doi.org/' + row['doi']) if row.get('doi') else '未找到'}",
                f"- arXiv：{('https://arxiv.org/abs/' + row['arxiv']) if row.get('arxiv') else '未找到'}",
                f"- 论文页：{row.get('landing_url') or '未找到'}",
                f"- 建议目录：`papers/{row['theme']}/`",
                f"- 原因：{row.get('error') or '下载失败'}",
                "",
            ]
        )
    (OUT / "manual_download_links.md").write_text("\n".join(manual_lines), encoding="utf-8")

    themes: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        themes.setdefault(row["theme"], []).append(row)
    readme = [
        "# QURA-Cert 文献包",
        "",
        "本目录围绕多领域知识图谱补全、负迁移与源选择、查询级反事实效用、KGC 基线与评测协议整理。PDF 来自公开来源或用户提供的合法个人副本。",
        "",
        "## 汇总",
        "",
        f"- 候选文献：{len(rows)} 篇",
        f"- 已获得并验证 PDF：{len(downloaded)} 篇",
        f"- 需要手动下载：{len(missing)} 篇",
        f"- 核心文献缺失：{len(core_missing)} 篇",
        "",
        "## 文件",
        "",
        "- `references.bib`：便携版 BibTeX，可直接用于 LaTeX/JabRef。",
        "- `references_zotero.bib`：含本机 PDF 绝对路径，导入 Zotero 时可一并关联附件。",
        "- `references.ris`：适合 EndNote、NoteExpress、Zotero 等引用管理器。",
        "- `literature_catalog.csv`：选题、重要性、DOI/arXiv、下载状态、相关性说明。",
        "- `manual_download_links.md`：仍未取得 PDF 的 DOI 和论文页。",
        "- `literature_manifest.json`：完整机器可读记录，含下载与人工归档记录。",
        "- `checksums.sha256`：PDF 完整性校验。",
        "- `citation_metadata_report.md`：作者和 venue 完整性检查结果。",
        "",
        "## 主题统计",
        "",
        "| 主题 | 总数 | 已下载 | 缺失 |",
        "|---|---:|---:|---:|",
    ]
    for theme, items in sorted(themes.items()):
        count = sum(item.get("status") == "downloaded" for item in items)
        readme.append(f"| `{theme}` | {len(items)} | {count} | {len(items) - count} |")
    readme.extend(
        [
            "",
            "## 使用建议",
            "",
            "先读 `01_core_multidomain_kgc` 建立 Related Work 主线，再读 `03_negative_transfer_and_source_selection` 与 `04_counterfactual_utility_and_risk` 做新颖性边界；实验实现与评价协议集中在 `02_kgc_backbones_and_reasoning` 和 `05_evaluation_surveys_and_systems`。",
            "",
        ]
    )
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")

    no_authors = [row for row in rows if not row.get("authors")]
    no_venue = [row for row in rows if not row.get("venue") and not row.get("arxiv")]
    report = [
        "# 引用元数据质量报告",
        "",
        f"- 总条目：{len(rows)}",
        f"- 缺作者：{len(no_authors)}",
        f"- 缺正式 venue 且无 arXiv：{len(no_venue)}",
        "",
        "BibTeX 和 RIS 已生成；含 DOI 的条目建议导入 Zotero 后再执行一次“从 DOI 更新元数据”。",
        "",
    ]
    if no_authors:
        report.extend(["## 缺作者", "", *[f"- `{row['key']}` — {row['title']}" for row in no_authors], ""])
    if no_venue:
        report.extend(["## 缺 venue", "", *[f"- `{row['key']}` — {row['title']}" for row in no_venue], ""])
    (OUT / "citation_metadata_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_key = {row["key"]: row for row in rows}
    seeds = {seed.key: seed for seed in pipeline.SEEDS}

    for key, source in SOURCE_BY_KEY.items():
        if key not in by_key or key not in seeds:
            raise KeyError(f"Unknown literature key: {key}")
        validate_pdf(source)
        row = by_key[key]
        seed = seeds[key]
        destination = pipeline.PAPERS / row["theme"] / pipeline.safe_filename(
            seed, row["title"], row.get("year")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256(source)
        if destination.exists():
            validate_pdf(destination)
            if sha256(destination) != source_hash:
                raise FileExistsError(f"Refusing to overwrite a different PDF: {destination}")
        else:
            shutil.copy2(source, destination)

        attempt = {
            "url": str(source),
            "result": "manually supplied by user; copied and validated",
        }
        attempts = list(row.get("attempts") or [])
        if attempt not in attempts:
            attempts.append(attempt)
        row.update(
            {
                "status": "downloaded",
                "local_path": str(destination.relative_to(ROOT)),
                "size_bytes": destination.stat().st_size,
                "sha256": source_hash,
                "source_url": str(source),
                "error": "",
                "attempts": attempts,
            }
        )
        print(f"[ingested] {key}: {destination.name} ({destination.stat().st_size:,} bytes)")

    rebuild_indexes(rows)
    downloaded = sum(row.get("status") == "downloaded" for row in rows)
    print(f"DONE: {downloaded}/{len(rows)} PDFs archived; {len(rows) - downloaded} still missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
