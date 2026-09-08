"""Compact 2 x 8 MRR panels from the seed-17 language/KG table.

Only plots saved results; does not train or evaluate any model.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SOURCE = ROOT / "outputs/kbs/_main/_tables/language_breakdown_seed17/language_kg_seed17.csv"
PDF_SOURCE = SOURCE.parent / "All_Datasets_Language_KG_Seed17_Portrait_Compact.pdf"
STEM = "Language_KG_MRR_Seed17_2x8"
rows = list(csv.DictReader(SOURCE.open(encoding="utf-8-sig")))
lookup = {(r["dataset"], r["kg"], r["method"]): r for r in rows}
assert len(lookup) == len(rows)

METHODS = ["TransE", "DistMult", "RotatE", "LSMGA", "DMKGC", "IMKGC",
           "ATransN", "TransE (teacher)", "ASRC"]
COLORS = ["#7394b8", "#77b7af", "#ccba72", "#a9c98b", "#b5a5ca", "#deb08f",
          "#97a0aa", "#6a9a77", "#d65335"]
PALETTE = dict(zip(METHODS, COLORS))
GROUPS = [
    (0, 0, 5, "DBP-5L", "#365d7c"),
    (0, 5, 8, "DWY", "#675488"),
    (1, 0, 6, "E-PKG", "#507961"),
    (1, 6, 8, "WK3l-15k", "#90643c"),
]
PANELS = [
    ("dbp5l", "el", "EL"), ("dbp5l", "en", "EN"),
    ("dbp5l", "es", "ES"), ("dbp5l", "fr", "FR"),
    ("dbp5l", "ja", "JA"), ("dwy", "db", "DBpedia"),
    ("dwy", "wk", "Wikidata"), ("dwy", "yg", "YAGO"),
    ("depkg", "de", "DE"), ("depkg", "es", "ES"),
    ("depkg", "fr", "FR"), ("depkg", "it", "IT"),
    ("depkg", "jp", "JP"), ("depkg", "uk", "UK"),
    ("wk3l", "en_f", "EN_F (suppl.)"), ("wk3l", "fr", "FR (primary)"),
]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5,
    "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6.5,
    "axes.linewidth": .55, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "svg.hashsalt": "asrc-language-kg-2x8-seed17",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
fig, axes = plt.subplots(2, 8, figsize=(12.8, 4.25), sharey=True)
fig.subplots_adjust(left=.038, right=.992, bottom=.115, top=.817, wspace=.18, hspace=.50)
records = []
for i, (ds, kg, label) in enumerate(PANELS):
    ax = axes.flat[i]
    methods = [m for m in METHODS if (ds, kg, m) in lookup]
    assert len(methods) == (9 if ds == "wk3l" else 7)
    values = []
    for m in methods:
        r = lookup[(ds, kg, m)]
        assert r["seed"] == "17" and r["status"] == "completed"
        value = float(r["mrr"]) * 100
        assert np.isfinite(value) and 0 <= value <= 100
        values.append(value)
        records.append({"panel": i+1, "dataset": ds, "kg": kg, "method": m,
                        "seed": 17, "mrr_percent": value,
                        "source_result": r["source_result"], "source_rank": r["source_rank"]})
    x = np.arange(len(methods))
    ax.bar(x, values, width=.80, color=[PALETTE[m] for m in methods],
           edgecolor="white", linewidth=.35, zorder=3)
    for xx, yy, method in zip(x, values, methods):
        ax.text(xx, yy+1.8, f"{yy:.2f}", rotation=90,
                ha="center", va="bottom", fontsize=5.8,
                fontweight="bold" if method == "ASRC" else "normal",
                color="#ab3922" if method == "ASRC" else "#3a424a")
    ax.set_ylim(0, 100)
    ax.set_xlim(-.6, len(methods)-.4)
    ax.set_xticks([])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", length=2, width=.45, pad=2)
    ax.set_title(label, pad=5)
    ax.grid(axis="y", color="#d9dfe4", linestyle="--", linewidth=.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#9aa4ad")
    ax.spines["bottom"].set_color("#9aa4ad")
    if i % 8 == 0:
        ax.set_ylabel("MRR (%)", labelpad=5)

for row, start, end, label, color in GROUPS:
    p0, p1 = axes[row, start].get_position(), axes[row, end-1].get_position()
    y = p0.y1 + .053
    fig.text((p0.x0+p1.x1)/2, y+.002, label,
             ha="center", va="bottom", color=color, fontsize=9.5, fontweight="bold")
    fig.add_artist(plt.Line2D([p0.x0, p1.x1], [y-.009, y-.009],
                             transform=fig.transFigure, color=color, linewidth=.8))

handles = [Patch(facecolor=PALETTE[m], label=m) for m in METHODS]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.51, .995),
           ncols=9, frameon=False, handlelength=1.2, handleheight=.85,
           columnspacing=1.4, handletextpad=.5, fontsize=8)
fig.text(.515, .042,
         "Seed 17  |  Bars follow legend order within each panel  |  EN_F: supplemental; FR: primary  |  No error bars",
         ha="center", va="center", fontsize=7, color="#555f69")

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
clipped = []
for text in fig.findobj(matplotlib.text.Text):
    if not text.get_visible() or not text.get_text():
        continue
    b = text.get_window_extent(renderer)
    if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
        clipped.append(text.get_text())
assert not clipped, clipped
assert len(records) == 116

fig.savefig(OUT / (STEM+".pdf"), metadata={"Title":"MRR by language and KG, seed 17", "Author":""})
fig.savefig(OUT / (STEM+".svg"))
fig.savefig(OUT / (STEM+".png"), dpi=600)
fig.savefig(OUT / (STEM+"_preview.png"), dpi=180)
with (OUT / "plot_data.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader(); writer.writerows(records)
manifest = {
    "source_csv": str(SOURCE), "source_pdf": str(PDF_SOURCE),
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "layout": "2 rows x 8 columns", "metric": "MRR (%)", "seed": 17,
    "panel_count": 16, "bar_count": len(records), "y_limits": [0,100],
    "figure_size_cm": [12.8*2.54,4.25*2.54], "error_bars": False,
    "panel_order": PANELS, "canvas_text_clipping": clipped,
    "notes": "Full-precision CSV values. No new experiments. WK3l EN_F is supplemental; AVG rows are omitted."
}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "README.zh.md").write_text("""# 语言／KG 单种子 MRR：2 行 × 8 列

上排：DBP-5L 5 个语言、DWY 3 个 KG。下排：E-PKG 6 个语言、WK3l-15k 2 个语言。

每个核心 KG 保留 7 个方法，WK3l 保留 9 个方法，共 16 个面板、116 根柱子。方法按共用图例顺序排列；ATransN 与 TransE (teacher) 只出现在 WK3l。红色为 ASRC。

数据来自用户指定 PDF 配套的 language_kg_seed17.csv，MRR 完整精度乘 100 后绘制，标签显示两位小数。seed 17 单种子，无误差条；纵轴统一 0–100%。EN_F 为补充评测，FR 为主目标；不添加 AVG 面板。

输出 PDF、可编辑 SVG、600 dpi PNG、轻量预览、绘图源代码和数据 CSV。画布约 32.5 × 10.8 cm，每个坐标区域约 3.4 × 3.0 cm，可用矢量文件调整整体大小。

只生成本目录内的绘图文件；未修改原始数据或现有图表，未启动实验。
""", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
plt.close(fig)
