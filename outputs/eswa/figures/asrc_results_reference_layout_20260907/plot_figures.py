"""Draw ASRC Figures 3--6 from frozen results; no model execution.

Run with Python + matplotlib + numpy + Pillow.  All plotted measurements are
read from reproduction/sota/paper_support/RESULTS.json.  The four paper
references are recorded in reference_map.json and README.zh.md.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SOURCE = ROOT / "reproduction/sota/paper_support/RESULTS.json"
CELLS = ROOT / "outputs/kbs/_main/_tables/cells_results.csv"
FROZEN_PDF = ROOT / "outputs/kbs/_main/_tables/KBS_Main_Text_Tables.20260907_final.pdf"
DATA = json.loads(SOURCE.read_text(encoding="utf-8"))
TABLES = {t: {r["key"]: r for r in rows} for t, rows in DATA["tables"].items()}
DATASETS = ["dbp5l", "depkg", "dwy", "wk3l"]
NAMES = ["DBP-5L", "E-PKG", "DWY", "WK3l-15k"]
SEEDS = [17, 29, 43]
assert DATA["seed_order"] == SEEDS
PLOT_RECORDS: list[dict] = []
LAYOUT_CHECKS: list[dict] = []
FIGURES: list[dict] = []

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9.0,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.65,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "asrc-20260907-reference-layout",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": True,
})


def stat(table: str, key: str, metric: str) -> dict:
    s = TABLES[table][key]["stats"][metric]
    values = np.array(s["seeds"], dtype=float)
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError((table, key, metric, "invalid seed values"))
    if not np.isclose(values.mean(), s["mean"], rtol=1e-11, atol=1e-10):
        raise ValueError((table, key, metric, "mean mismatch"))
    if not np.isclose(values.std(ddof=1), s["sd"], rtol=1e-9, atol=1e-10):
        raise ValueError((table, key, metric, "sample SD mismatch"))
    return s


def record(fig: int, panel: str, table: str, key: str, metric: str,
           unit: str, derivation: str = "stored mean and sample SD") -> dict:
    s = stat(table, key, metric)
    PLOT_RECORDS.append({
        "figure": fig, "panel": panel, "table": table, "source_key": key,
        "metric": metric, "unit": unit, "value": s["mean"],
        "sd": s["sd"], **{f"seed_{k}": v for k, v in zip(SEEDS, s["seeds"])},
        "derivation": derivation,
    })
    return s


def series(fig: int, panel: str, table: str, keys: list[str],
           metric: str, unit: str) -> tuple[np.ndarray, np.ndarray]:
    values = [record(fig, panel, table, k, metric, unit) for k in keys]
    return np.array([v["mean"] for v in values]), np.array([v["sd"] for v in values])


def derived_record(fig, panel, key, metric, value, unit, derivation, sd="", seeds=None):
    PLOT_RECORDS.append({
        "figure": fig, "panel": panel, "table": "derived", "source_key": key,
        "metric": metric, "unit": unit, "value": float(value), "sd": sd,
        **{f"seed_{k}": v for k, v in zip(SEEDS, seeds if seeds is not None else [""]*3)},
        "derivation": derivation,
    })


def frame(ax, *, grid="both", linestyle="-", color="#acacac"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=color, linewidth=0.45, linestyle=linestyle, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#333333")


def panel_label(fig, ax, text, y=0.035):
    p = ax.get_position()
    fig.text(p.x0+p.width/2, y, text, ha="center", va="bottom",
             fontfamily="Times New Roman", fontsize=10.5)


def errorline(ax, x, y, sd, color, marker, label, **kw):
    return ax.errorbar(x, y, yerr=sd, color=color, marker=marker,
        markersize=3.6, linewidth=0.95, markeredgewidth=0.55,
        capsize=2.0, capthick=0.65, elinewidth=0.65, label=label, **kw)


def write_figure(fig, number, stem, caption, reference):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fw, fh = fig.bbox.width, fig.bbox.height
    clipped = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fw+1 or box.y1 > fh+1:
            clipped.append(text.get_text())
    if clipped:
        raise ValueError(f"Figure {number}: text outside canvas: {clipped}")
    LAYOUT_CHECKS.append({"figure": number, "text_outside_canvas": clipped})
    metadata = {"Title": f"ASRC Figure {number}", "Author": "", "Subject": caption}
    fig.savefig(OUT / f"{stem}.pdf", metadata=metadata)
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}_preview.png", dpi=200)
    FIGURES.append({"number": number, "stem": stem, "caption": caption,
                    "reference": reference, "figure": fig})


def figure3():
    """Wang et al. Fig. 3: two side-by-side line panels and inset legends."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(left=0.072, right=0.985, top=0.965, bottom=0.225, wspace=0.27)
    x = np.arange(4)
    ax = axes[0]
    frame(ax)
    configs = [
        ("All queries", "T4", [d+".shared" for d in DATASETS], "#0000ff", "o"),
        ("Aligned heads", "T6", [d+".aligned" for d in DATASETS], "#ff0000", "^"),
        ("Unaligned heads", "T6", [d+".unaligned" for d in DATASETS], "#00a33a", "s"),
    ]
    offsets = {
        "All queries": [(4,-14), (2,8), (0,7), (-2,10)],
        "Aligned heads": [(5,7), (0,7), (0,8), (-2,7)],
        "Unaligned heads": [(1,8), (0,-14), (0,-14), (-2,-14)],
    }
    for label, table, keys, color, marker in configs:
        y, sd = series(3,"a",table,keys,"delta","percentage points")
        errorline(ax,x,y,sd,color,marker,label)
        for i, (xx, yy) in enumerate(zip(x,y)):
            ax.annotate(f"{yy:+.2f}",(xx,yy),xytext=offsets[label][i],
                        textcoords="offset points",ha="center",fontsize=7.6,color=color,
                        bbox={"facecolor":"white","edgecolor":"none","alpha":.85,"pad":.25})
    ax.axhline(0,color="#222222",linestyle="--",linewidth=0.75,zorder=1)
    ax.set(xlim=(-.20,3.25),ylim=(-7.5,38.0),ylabel="MRR gain (percentage points)",xlabel="Dataset")
    ax.set_xticks(x,NAMES)
    ax.set_yticks([0,10,20,30])
    ax.legend(loc="upper right",frameon=True,fancybox=False,edgecolor="#666666",
              framealpha=1,borderpad=.35,handlelength=1.9,labelspacing=.35)
    panel_label(fig,ax,"(a) Shared minus Independent")

    ax=axes[1]
    frame(ax)
    ptr, ptr_sd=series(3,"b","T4",[d+".shared" for d in DATASETS],"ptr","percent")
    ntr, ntr_sd=series(3,"b","T4",[d+".shared" for d in DATASETS],"ntr","percent")
    unchanged=[]
    unchanged_sd=[]
    for d in DATASETS:
        p=stat("T4",d+".shared","ptr")
        n=stat("T4",d+".shared","ntr")
        a=100-np.array(p["seeds"])-np.array(n["seeds"])
        unchanged.append(a.mean()); unchanged_sd.append(a.std(ddof=1))
        derived_record(3,"b",d+".shared","unchanged",a.mean(),"percent",
                       "per seed: 100 - PTR - NTR; then mean and sample SD",a.std(ddof=1),a)
    errorline(ax,x,ptr,ptr_sd,"#ff0000","^","Improved (PTR)")
    errorline(ax,x,np.array(unchanged),np.array(unchanged_sd),"#333333","s","Unchanged")
    errorline(ax,x,ntr,ntr_sd,"#0000ff","o","Worsened (NTR)")
    for xx,yy in zip(x,ntr):
        ax.annotate(f"{yy:.2f}",(xx,yy),xytext=(0,-13),textcoords="offset points",
                    ha="center",fontsize=7.6,color="#0000ff")
    ax.set(xlim=(-.20,3.25),ylim=(0,70),ylabel="Query proportion (%)",xlabel="Dataset")
    ax.set_xticks(x,NAMES)
    ax.set_yticks([0,10,20,30,40,50,60,70])
    ax.legend(loc="upper right",frameon=True,fancybox=False,edgecolor="#666666",
              framealpha=1,borderpad=.35,handlelength=1.9,labelspacing=.35)
    panel_label(fig,ax,"(b) Query-level ranking outcomes")
    caption=("Alignment-sharing gains and query-level outcomes across four datasets. "
        "(a) Shared minus Independent MRR for all queries and the predefined head-alignment groups. "
        "(b) Proportions of improved, unchanged, and worsened query rankings for Shared versus Independent. "
        "Markers and error bars denote the three-seed mean and sample standard deviation. Differences are "
        "computed within each seed before averaging. Group metrics are macro-averaged over nonempty KGs; "
        "DBP-5L unaligned heads cover four KGs and 643 queries. Lines connect categorical datasets as visual "
        "guides. E-PKG results here compare Shared and Independent, although ASRC uses Independent there.")
    write_figure(fig,3,"figure3_sharing_transfer",caption,"wang2019")


def figure4():
    """SynergyKGC Fig. 4(a): pastel grouped bars, labels, grey guide, stars."""
    fig,ax=plt.subplots(figsize=(7.2,4.0))
    fig.subplots_adjust(left=.087,right=.985,top=.80,bottom=.15)
    ax.set_axisbelow(True)
    ax.grid(axis="both",color="#cccccc",linestyle="--",linewidth=.6)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_linewidth(.8)
    variants=["independent","relation_only","entity_only","shared","no_reciprocal","no_n3"]
    labels=["Independent","Relation only","Entity only","Full shared","w/o reciprocal","w/o N3"]
    colors=["#8aa4bf","#a8dcb0","#ffe49b","#f2b2b5","#c5b3de","#a7d4d9"]
    metrics=["h1","h10","mrr"]
    metric_names=["Hits@1","Hits@10","MRR"]
    x=np.arange(3)
    width=.107
    offsets=(np.arange(6)-2.5)*width
    val=np.empty((6,3)); err=np.empty((6,3))
    for i,(v,label,c) in enumerate(zip(variants,labels,colors)):
        for j,m in enumerate(metrics):
            s=record(4,"a","T5","dbp5l."+v,m,"percent")
            val[i,j]=s["mean"];err[i,j]=s["sd"]
        ax.bar(x+offsets[i],val[i],width=width*.96,color=c,edgecolor="white",linewidth=.4,
               yerr=err[i],capsize=1.8,error_kw={"elinewidth":.6,"capthick":.6,"ecolor":"#4d4d4d"},
               label=label,zorder=3)
        for xx,yy,ss in zip(x+offsets[i],val[i],err[i]):
            ax.text(xx,yy+ss+3.2,f"{yy:.2f}",ha="center",va="bottom",rotation=90,
                    fontsize=8.1,zorder=5)
    for j in range(3):
        xp=x[j]+offsets
        ax.plot(xp,val[:,j],color="#5d5d5d",marker="o",markersize=3,linewidth=.85,zorder=4)
        best=int(np.argmax(val[:,j]))
        ax.scatter([xp[best]],[val[best,j]],marker="*",s=85,facecolors="white",
                   edgecolors="#ff0000",linewidths=.9,zorder=6)
    handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,labels)]
    handles += [Line2D([0],[0],color="#5d5d5d",marker="o",markersize=3,label="Value guide"),
                Line2D([0],[0],color="none",marker="*",markeredgecolor="#ff0000",
                       markerfacecolor="white",markersize=8,label="Best")]
    legend=ax.legend(handles=handles,ncols=4,loc="lower center",bbox_to_anchor=(.5,1.10),
        frameon=True,fancybox=False,edgecolor="#666666",handlelength=1.6,columnspacing=1.3,
        borderpad=.4,labelspacing=.5)
    ax.set(xlim=(-.52,2.52),ylim=(0,108),ylabel="Score (%)")
    ax.set_xticks(x,metric_names,fontweight="bold",fontsize=10)
    ax.set_yticks(np.arange(0,101,20))
    ax.set_ylabel("Score (%)",fontweight="bold")
    fig.text(.535,.035,"Component ablation on DBP-5L",ha="center",va="bottom",
             fontfamily="Times New Roman",fontsize=11)
    caption=("Component ablation on DBP-5L under the fixed training recipe. All six configurations "
        "are shown for Hits@1, Hits@10, and MRR. Pastel bars and vertical value labels follow the layout "
        "of SynergyKGC Figure 4(a); grey lines only connect values within each metric. Error bars are sample "
        "standard deviations across seeds 17, 29, and 43. Hollow stars mark the highest measured mean in "
        "each metric, which is the configuration without reciprocal augmentation. Its MRR exceeds Full "
        "shared by 0.3021 percentage points; the paired sample SD of that difference is 0.1803.")
    write_figure(fig,4,"figure4_component_ablation",caption,"synergykgc")


def figure5():
    """DMKGC Fig. 4: single wide grouped bar panel, legend above the box."""
    with plt.rc_context({"font.family":"Times New Roman","font.size":10,
                        "axes.labelsize":11,"xtick.labelsize":9.5,"ytick.labelsize":9.5,
                        "legend.fontsize":10}):
        fig,ax=plt.subplots(figsize=(7.2,2.8))
        fig.subplots_adjust(left=.085,right=.988,top=.85,bottom=.26)
        frame(ax,grid="y",color="#c7c7c7")
        conditions=["clean","train50","align50","noise10"]
        names=["Clean","50% training facts","50% alignment links","10% endpoint replacements"]
        x=np.arange(4)
        for mode,offset,color,label in [("independent",-.15,"#b03a79","Independent"),
                                       ("shared",.15,"#2e8bad","Shared")]:
            y,sd=series(5,"a","T7",["dbp5l."+c+"."+mode for c in conditions],"mrr","percent")
            ax.bar(x+offset,y,width=.29,color=color,edgecolor="white",linewidth=.4,
                   yerr=sd,capsize=2,error_kw={"elinewidth":.6,"capthick":.6},label=label,zorder=3)
            for xx,yy,ss in zip(x+offset,y,sd):
                ax.text(xx,yy+ss+1.0,f"{yy:.2f}",ha="center",va="bottom",fontsize=9.2)
        for c in conditions:
            si=stat("T7","dbp5l."+c+".independent","mrr")
            ss=stat("T7","dbp5l."+c+".shared","mrr")
            delta=np.array(ss["seeds"])-np.array(si["seeds"])
            derived_record(5,"caption","dbp5l."+c,"shared_minus_independent",delta.mean(),
                           "percentage points","within-condition per-seed Shared MRR - Independent MRR",
                           delta.std(ddof=1),delta)
        ax.set(xlim=(-.5,3.5),ylim=(0,84),ylabel="AVG-MRR (%)",xlabel="Input condition on DBP-5L")
        ax.set_xticks(x,names)
        ax.set_yticks([0,20,40,60,80])
        ax.legend(loc="lower center",bbox_to_anchor=(.5,1.01),ncols=2,frameon=True,
                  fancybox=False,edgecolor="#bdbdbd",borderpad=.22,columnspacing=3,handlelength=1.8)
        caption=("Mean completion performance under fixed DBP-5L input perturbations. Independent and "
            "Shared are compared under clean inputs, 50% retained training-fact groups, 50% retained "
            "alignment links, and 10% alignment endpoint replacements. Error bars show the sample SD "
            "across three training seeds. One perturbation seed (20260906) is fixed across runs. Independent "
            "results for the two alignment perturbations reuse the clean Independent model. The corresponding "
            "within-condition Shared-minus-Independent MRR advantages are 32.38, 32.96, 20.67, and 24.56 "
            "percentage points. Filtering continues to use the original public training facts.")
        write_figure(fig,5,"figure5_input_perturbations",caption,"dmkgc")


def figure6():
    """A*Net Fig. 6: two side-by-side panels, blue/orange twin-axis lines."""
    with plt.rc_context({"font.family":"Times New Roman","font.size":9.2,
                        "axes.labelsize":9.4,"xtick.labelsize":8.2,"ytick.labelsize":8.3,
                        "legend.fontsize":8.0}):
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.25))
        fig.subplots_adjust(left=.072,right=.93,top=.955,bottom=.235,wspace=.63)
        x=np.arange(4)
        metrics=["params_m","qps","p50_ms","p95_ms"]
        independent={}; asrc={}
        for m in metrics:
            unit={"params_m":"million parameters","qps":"queries/s","p50_ms":"ms","p95_ms":"ms"}[m]
            independent[m],_=series(6,"inputs","T8",[d+".independent" for d in DATASETS],m,unit)
            asrc[m],_=series(6,"inputs","T8",[d+".asrc" for d in DATASETS],m,unit)
        savings=100*(1-asrc["params_m"]/independent["params_m"])
        throughput=asrc["qps"]/independent["qps"]
        p50=asrc["p50_ms"]/independent["p50_ms"]
        p95=asrc["p95_ms"]/independent["p95_ms"]
        for j,d in enumerate(DATASETS):
            derived_record(6,"a",d,"parameter_reduction",savings[j],"percent",
                           "100 * (1 - ASRC parameter count / Independent parameter count)")
            for m,arr,panel in [("qps",throughput,"a"),("p50_ms",p50,"b"),("p95_ms",p95,"b")]:
                derived_record(6,panel,d,m+"_ratio",arr[j],"ratio",
                               "ASRC three-seed mean / Independent three-seed mean; ratio of means")
        blue="#1f77b4"; orange="#ff7f0e"
        ax=axes[0]; right=ax.twinx()
        frame(ax)
        l1,=ax.plot(x,savings,color=blue,marker=".",markersize=4,linewidth=1.0,label="Parameter reduction")
        l2,=right.plot(x,throughput,color=orange,marker=".",markersize=4,linewidth=1.0,label="Throughput ratio")
        ax.set(xlim=(-.15,3.2),ylim=(0,80),ylabel="Parameter reduction (%)",xlabel="Dataset")
        right.set(ylim=(.85,1.25),ylabel="Throughput ratio (×)")
        ax.set_yticks([0,20,40,60,80]);right.set_yticks([.9,1.0,1.1,1.2])
        ax.set_xticks(x,["DBP-5L","E-PKG*","DWY","WK3l-15k"])
        right.axhline(1,color=orange,linewidth=.55,linestyle=":",alpha=.65)
        for xx,yy in zip(x,savings):
            ax.annotate(f"{yy:.2f}",(xx,yy),xytext=(0,6),textcoords="offset points",
                        ha="center",color=blue,fontsize=7.6)
        for xx,yy in zip(x,throughput):
            right.annotate(f"{yy:.3f}×",(xx,yy),xytext=(0,-14),textcoords="offset points",
                           ha="center",color=orange,fontsize=7.6)
        ax.legend(handles=[l1,l2],loc="upper left",fancybox=False,frameon=True,
                  edgecolor="#aaaaaa",borderpad=.35,handlelength=1.45,labelspacing=.4)
        panel_label(fig,ax,"(a) Parameters and throughput")

        ax=axes[1];right=ax.twinx()
        frame(ax)
        l1,=ax.plot(x,p50,color=blue,marker=".",markersize=4,linewidth=1.0,label="p50 latency ratio")
        l2,=right.plot(x,p95,color=orange,marker=".",markersize=4,linewidth=1.0,label="p95 latency ratio")
        ax.set(xlim=(-.15,3.2),ylim=(.90,1.23),ylabel="p50 latency ratio (×)",xlabel="Dataset")
        right.set(ylim=(.90,1.23),ylabel="p95 latency ratio (×)")
        ax.set_yticks([.9,1.0,1.1,1.2]);right.set_yticks([.9,1.0,1.1,1.2])
        ax.set_xticks(x,["DBP-5L","E-PKG*","DWY","WK3l-15k"])
        ax.axhline(1,color="#777777",linewidth=.6,linestyle=":")
        for xx,yy in zip(x,p50):
            ax.annotate(f"{yy:.3f}×",(xx,yy),xytext=(0,7),textcoords="offset points",
                        ha="center",color=blue,fontsize=7.6)
        for xx,yy in zip(x,p95):
            right.annotate(f"{yy:.3f}×",(xx,yy),xytext=(0,-14),textcoords="offset points",
                           ha="center",color=orange,fontsize=7.6)
        ax.legend(handles=[l1,l2],loc="upper left",fancybox=False,frameon=True,
                  edgecolor="#aaaaaa",borderpad=.35,handlelength=1.45,labelspacing=.4)
        panel_label(fig,ax,"(b) Ranking latency")
        caption=("Parameter savings and warmed full-candidate ranking-cost ratios relative to Independent. "
            "(a) Percentage parameter reduction (left axis) and ASRC/Independent throughput ratio (right axis). "
            "(b) ASRC/Independent p50 (left axis) and p95 (right axis) latency ratios, using identical scales. "
            "A throughput ratio above one is faster; a latency ratio above one is slower. Ratios are computed "
            "from the three-seed means in Table 8, with no error bars on the derived ratios; raw per-seed "
            "variability is reported in the table. Profiling uses one RTX 2080 Ti, warmed ranking, batch size "
            "1 for latency and 256 for throughput. E-PKG* reuses the Independent model and measurements. "
            "Lines connect categorical datasets as visual guides.")
        write_figure(fig,6,"figure6_ranking_costs",caption,"astarnet")


REFERENCES={
    "wang2019": {
        "title":"Characterizing and Avoiding Negative Transfer",
        "figure":"Figure 3(a,b)","pdf_page":6,"printed_page":"11298",
        "local_pdf":"literature/papers/03_negative_transfer_and_source_selection/2019_negative_transfer_characterizing_and_avoiding_negative_transfer.pdf",
        "url":"https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Characterizing_and_Avoiding_Negative_Transfer_CVPR_2019_paper.html",
        "page_png":"wang2019_fig3_page.png","crop_png":"wang2019_fig3_reference.png",
        "crop_pixels":[85,553,486,728],
        "layout":"Two equal horizontal line panels; boxed axes, full grid, inset legends, colored lines and markers, bottom subfigure labels.",
        "changes":"Replace source/target adaptation curves with categorical dataset results: MRR differences by query group and query-ranking outcome proportions. Add saved three-seed SD bars and numeric labels."
    },
    "synergykgc": {
        "title":"SynergyKGC: Reconciling Topological Heterogeneity in Knowledge Graph Completion via Topology-Aware Synergy",
        "figure":"Figure 4(a)","pdf_page":6,"printed_page":"6 (PDF)",
        "local_pdf":"literature/papers/01_core_multidomain_kgc/2026_synergykgc_synergykgc_reconciling_topological_heterogeneity_in_knowledge_graph_completion_via_topology_awa.pdf",
        "url":"https://arxiv.org/abs/2602.10845",
        "page_png":"synergykgc_fig4_page.png","crop_png":"synergykgc_fig4a_reference.png",
        "crop_pixels":[110,93,616,470],
        "layout":"Grouped pastel bars, dashed grid, vertical values, grey connecting lines, hollow red best-value stars, boxed legend above the panel.",
        "changes":"Use all six DBP-5L ablations and three available metrics; one dataset means no dataset hatching; stars follow the actual per-metric best mean. No radar panel is borrowed."
    },
    "dmkgc": {
        "title":"Conditional Diffusion Guided Knowledge Transfer for Multi-Domain Knowledge Graph Completion",
        "figure":"Figure 4","pdf_page":7,"printed_page":"3750",
        "local_pdf":"literature/papers/01_core_multidomain_kgc/2026_dmkgc_conditional_diffusion_guided_knowledge_transfer_for_multi_domain_knowledge_graph_completion.pdf",
        "url":"https://doi.org/10.1145/3774904.3792252",
        "page_png":"dmkgc_fig4_page.png","crop_png":"dmkgc_fig4_reference.png",
        "crop_pixels":[541,145,944,292],
        "layout":"One wide grouped-bar panel; magenta/teal colors from the original palette; legend above; thin box; horizontal grid; values above bars; Times-style text.",
        "changes":"Four observed perturbation categories and two models replace three entity-coverage levels and six methods. Use a zero-based MRR axis and saved sample SD bars."
    },
    "astarnet": {
        "title":"A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs",
        "figure":"Figure 6","pdf_page":9,"printed_page":"9",
        "local_pdf":"literature/papers/02_kgc_backbones_and_reasoning/2023_astarnet_a_net_a_scalable_path_based_reasoning_approach_for_knowledge_graphs.pdf",
        "url":"https://arxiv.org/abs/2206.04798",
        "page_png":"astarnet_fig6_page.png","crop_png":"astarnet_fig6_reference.png",
        "crop_pixels":[185,383,515,520],
        "layout":"Two horizontal twin-axis line panels, blue and orange series, thin grey grid, upper-left boxed legends, Times-style text.",
        "changes":"Use four categorical datasets; plot parameter savings and throughput ratio, then p50 and p95 ratios, all from saved Table 8 measurements. Ratios use Independent as denominator."
    },
}


def package():
    for r in REFERENCES.values():
        im=Image.open(OUT/"references"/r["page_png"])
        im.crop(tuple(r["crop_pixels"])).save(OUT/"references"/r["crop_png"])
        r["sha256"]=hashlib.sha256((ROOT/r["local_pdf"]).read_bytes()).hexdigest()
    (OUT/"reference_map.json").write_text(json.dumps(REFERENCES,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"plot_data.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(PLOT_RECORDS[0]))
        writer.writeheader();writer.writerows(PLOT_RECORDS)
    with PdfPages(OUT/"ASRC_Figures_3-6.pdf",metadata={"Title":"ASRC Figures 3–6","Author":""}) as pdf:
        for spec in FIGURES: pdf.savefig(spec["figure"])
    captions="\n\n".join(f"Figure {s['number']}. {s['caption']}" for s in FIGURES)
    (OUT/"captions.en.txt").write_text(captions+"\n",encoding="utf-8")
    manifest={
        "status":"rendered","scope":"Existing result plotting only; no training or evaluation executed.",
        "source":str(SOURCE),"source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "cells_csv":str(CELLS),"cells_sha256":hashlib.sha256(CELLS.read_bytes()).hexdigest(),
        "frozen_table_pdf":str(FROZEN_PDF),
        "seeds":SEEDS,"plot_record_count":len(PLOT_RECORDS),
        "aggregation":"Use stored KG-macro metrics; seed arithmetic mean and sample SD (ddof=1).",
        "figure6_ratios":"Ratio of reported three-seed means, not mean of per-seed ratios.",
        "matplotlib_version":matplotlib.__version__,"layout_checks":LAYOUT_CHECKS,
        "figures":[{k:v for k,v in s.items() if k!="figure"} for s in FIGURES],
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    build_preview()
    build_readme()
    build_contact_sheet()


def build_preview():
    articles=[]
    chinese={3:"共享收益与查询级影响",4:"组件消融",5:"输入扰动",6:"参数与排序成本"}
    for s in FIGURES:
        r=REFERENCES[s["reference"]]
        stem=s["stem"]
        articles.append(f'''<article id="figure{s['number']}">
          <h2>Figure {s['number']} · {chinese[s['number']]}</h2>
          <p class="source">参考：<a href="{r['url']}">{html.escape(r['title'])}</a>，
          {r['figure']}，PDF 第 {r['pdf_page']} 页。</p>
          <div class="pair"><section><span class="label">论文参考图</span>
          <img class="ref" src="references/{r['crop_png']}" alt="Reference figure"></section>
          <section><span class="label">ASRC · 已替换为真实实验数据</span>
          <a href="{stem}.png"><img src="{stem}_preview.png" alt="ASRC Figure {s['number']}"></a></section></div>
          <div class="links"><a href="{stem}.pdf">矢量 PDF</a><a href="{stem}.svg">可编辑 SVG</a>
          <a href="{stem}.png">600 dpi PNG</a></div>
          <details><summary>英文图注与统计口径</summary><p>{html.escape(s['caption'])}</p></details>
          </article>''')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>ASRC Figures 3–6 · 参考版式对照</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f5f6f8;color:#202935;font:15px/1.65 Arial,"Microsoft YaHei",sans-serif}
    main{max-width:1380px;margin:auto;padding:34px}h1{font-size:27px;margin:0 0 8px}h2{font-size:21px;margin:0 0 5px}
    header p{margin:6px 0;color:#566170}nav{display:flex;gap:16px;margin:18px 0 28px;flex-wrap:wrap}
    a{color:#176d9b;text-decoration:none}a:hover{text-decoration:underline}article{padding:25px;margin:24px 0;background:white;border:1px solid #dce1e7;border-radius:8px}
    .source{font-size:13px;color:#596878}.pair{display:grid;grid-template-columns:0.78fr 1.22fr;gap:22px;align-items:center;margin:18px 0}
    section{min-width:0;border:1px solid #e4e8ee;padding:12px;background:white}.label{display:block;font-size:12px;color:#6e7885;margin-bottom:12px}
    img{display:block;width:100%;height:auto}.links{display:flex;gap:18px;font-size:13px}details{font-size:13px;color:#596878;margin-top:15px}
    details p{max-width:1000px}footer{color:#78838c;font-size:12px;padding:20px 0}@media(max-width:850px){main{padding:18px}.pair{grid-template-columns:1fr}}
    </style><main><header><h1>ASRC · 四张结果图与参考版式</h1><p>Figure 3–6 | seeds 17 / 29 / 43 | 原始汇总数据驱动 | PDF / SVG / 600 dpi PNG</p>
    <nav><a href="ASRC_Figures_3-6.pdf">打开四图合并 PDF</a><a href="plot_data.csv">绘图数据</a><a href="plot_figures.py">绘图源代码</a><a href="README.zh.md">来源与口径说明</a></nav></header>'''
    doc+='\n'.join(articles)
    doc+='''<footer>参考缩略图用于版式对照；提交用的四张图由 Matplotlib 依据 ASRC 已有实验数据重新绘制。</footer></main></html>'''
    (OUT/"preview.html").write_text(doc,encoding="utf-8")


def build_readme():
    rows=[]
    for s in FIGURES:
        r=REFERENCES[s["reference"]]
        rows.append(f"| Figure {s['number']} | {r['title']} | {r['figure']} | {r['pdf_page']} |")
    content="""# ASRC Figure 3–6：参考论文版式绘图

本目录包含四张正式结果图。按本轮“严格参照参考论文版式”的要求，图型以原图为准：Figure 3 为双面板折线；Figure 4 为分组柱状；Figure 5 为单面板分组柱状；Figure 6 为双面板双轴折线。上一轮规划中的森林图、堆叠条形图不用于这一版。

## 文件

- `ASRC_Figures_3-6.pdf`：四页矢量合并预览。
- `figure3_*.pdf` 至 `figure6_*.pdf`：独立矢量 PDF，可用于 LaTeX。
- 同名 `.svg`：文字保留为文本的可编辑矢量文件。
- 同名 `.png`：600 dpi 图片；`*_preview.png` 为轻量预览。
- `preview.html`：原论文参考图与新图并排展示。
- `plot_figures.py`：可复现绘图脚本；只读取现有结果，不启动实验。
- `plot_data.csv`：完整精度绘图数据、种子值与派生公式。
- `captions.en.txt`：英文图注。
- `reference_map.json`：原论文完整路径、URL、精确图号、PDF 页码、版式说明与文件 SHA-256。
- `manifest.json`：数据来源及哈希、统计口径、导出清单。

## 精确参考位置

| 新图 | 参考论文 | 原图 | PDF 页码 |
|---|---|---|---|
"""+"\n".join(rows)+"""

复用了参考图的面板排列、绘图类型、图例位置、配色风格和标注方式；数据序列数与横纵坐标按现有 ASRC 结果调整。Figure 4 仅参考 SynergyKGC 的 Figure 4(a)，没有复制它的雷达图。原文截图仅保存在 `references/` 中供版式核对；四张正式图不嵌入原文图像。

## 数据与统计口径

数据源：`G:/zhishitupui/reproduction/sota/paper_support/RESULTS.json`，以 `outputs/kbs/_main/_tables/KBS_Main_Text_Tables.20260907_final.pdf` 和 `cells_results.csv` 为当前发布口径。没有重跑模型或重新评测。

- Figure 3：Table 4 与 Table 6。左图为 Shared−Independent 的 MRR 百分点差；右图为查询改善、持平、恶化比例。持平比例逐种子按 `100−PTR−NTR` 计算。全部数据均比较 Shared 和 Independent，E-PKG 并未改写成 ASRC 自比较。分组由查询头实体是否有已知对齐定义；各组按非空 KG 宏平均，DBP-5L 未对齐组为 643 条查询、4 个非空 KG。
- Figure 4：Table 5 全部六个配置、MRR/Hits@1/Hits@10。空心星号标记每个指标的真实最高均值：三个指标均为去掉互逆增强的配置。MRR 提升使用完整精度得到 `+0.3020926459655682` 个百分点；没有把已有组件包装成全都不可缺少。
- Figure 5：Table 7 四种已评测离散输入条件。独立模型在两个对齐扰动条件下复用 clean 结果；三个训练种子共享同一个扰动种子 20260906。减少训练输入后，过滤仍保留原公开训练事实。柱状图从零起画。
- Figure 6：Table 8。参数节省为 `100×(1−P_ASRC/P_Independent)`；吞吐量、p50、p95 比值均为 `ASRC 三种子均值 / Independent 三种子均值`。吞吐比大于 1 表示更快；延迟比大于 1 表示更慢。此图不对“均值之比”伪造误差条，原始三种子 SD 保留在 Table 8 与 `plot_data.csv`。E-PKG* 复用 Independent 模型与测量。
- Figure 3–5 的误差条为种子 17、29、43 的样本标准差（ddof=1），不是置信区间。差值先在种子内计算，再求均值和 SD。
- Figure 3 和 Figure 6 的横轴是数据集类别，折线仅用于连接类别点，没有插值或连续趋势含义。

Table 1–8 与已有实验文件保持原样；未显示为曲线的完整绝对成本、内存、分组样本量和其他指标仍在原表中。

## 复现

安装有 Matplotlib、NumPy 和 Pillow 的 Python 可直接运行：

```powershell
& 'C:\\Users\\evan\\AppData\\Local\\Programs\\Python\\Python314\\python.exe' -B 'G:\\zhishitupui\\outputs\\eswa\\figures\\asrc_results_reference_layout_20260907\\plot_figures.py'
```

脚本使用 Arial 与 Times New Roman；其他系统可在 rcParams 中替换字体。标准科学绘图工具生成，输出为可编辑矢量图。
"""
    (OUT/"README.zh.md").write_text(content,encoding="utf-8")


def build_contact_sheet():
    canvas=Image.new("RGB",(1800,1320),"#f5f6f8")
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf",27)
    for i,s in enumerate(FIGURES):
        xx=22+(i%2)*890;yy=22+(i//2)*650
        draw.rounded_rectangle((xx,yy,xx+865,yy+625),radius=8,fill="white",outline="#d9dee4",width=1)
        draw.text((xx+18,yy+13),f"Figure {s['number']}",font=font,fill="#283547")
        im=Image.open(OUT/(s["stem"]+"_preview.png")).convert("RGB")
        im.thumbnail((835,555))
        canvas.paste(im,(xx+(865-im.width)//2,yy+55+(555-im.height)//2))
    canvas.save(OUT/"contact_sheet.png")


if __name__=="__main__":
    figure3();figure4();figure5();figure6();package()
    print(json.dumps({"output":str(OUT),"figures":len(FIGURES),"plotted_records":len(PLOT_RECORDS),
                      "layout_checks":LAYOUT_CHECKS},ensure_ascii=False,indent=2))
    plt.close("all")
