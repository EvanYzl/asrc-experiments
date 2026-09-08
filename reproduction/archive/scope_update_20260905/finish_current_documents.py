"""Finish current entry points and verify this archived scope migration."""
import csv
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT=Path('G:/zhishitupui')
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'
TABLES=ROOT/'outputs/kbs_main_tables'
ARCHIVE=Path(__file__).resolve().parent


def write(path, content):
    old=ARCHIVE/'before'/path.relative_to(ROOT)
    if path.exists() and not old.exists():
        old.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,old)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content,encoding='utf-8')


write(ROOT/'reproduction/README.md', '''# 基线的本机复现

当前只运行并填写 Table 1–3 的非 QURA 项。外部基线共 9 个：TransE、DistMult、RotatE、ATransN、LSMGA、DMKGC、IMKGC、AlignKGC、SS-AGA；内部对照为 Target-only 和 Uniform-All。Table 3 仅包含 AlignKGC、SS-AGA 在 DBP-5L 和 E-PKG 上的本地复现。

正式实验固定随机种子 17、29、43，覆盖各方法对应的全部目标 KG。每个完整三种子组通过原始查询排名、公开划分、验证选模、检查点与代码哈希审计后才填写。表格性能值只采用当前批次的本机复现结果。

- [实时状态与执行清单](runs/strict_baselines_20260905/STATUS.zh.md)
- [正式执行与绘图数据计划](strict_baselines/EXECUTION_PLAN.zh.md)
- [运行目录说明](runs/strict_baselines_20260905/README.zh.md)
- [当前表格 PDF](../outputs/kbs_main_tables/KBS_Main_Text_Tables.pdf)
- [逐格数值与来源](../outputs/kbs_main_tables/cells_results.csv)
- [中间结果字段](strict_baselines/ARTIFACT_SCHEMA.zh.md)

唯一活动队列为 `runs/strict_baselines_20260905/manifest.json`，单 GPU 顺序运行。已完成独立任务保留，支持完整续训的适配器从检查点恢复。历史源码和失败记录保留用于审计，不进入当前实验清单或填表来源。

QURA、Table 4–8、预算控制暂缓；后续匹配预算依据冻结的 QURA 验证集访问量确定。
''')
write(ROOT/'reproduction/BASELINE_REPRODUCTION_REPORT.md', '''# 基线复现报告

本阶段的正式报告由 [实时实验状态](runs/strict_baselines_20260905/STATUS.zh.md)、[当前表格](../outputs/kbs_main_tables/KBS_Main_Text_Tables.pdf) 和 [逐格来源 CSV](../outputs/kbs_main_tables/cells_results.csv) 共同构成。范围为 Table 1–3 的非 QURA 项，外部基线共 9 个。

每个性能单元格对应完整目标 KG 和随机种子 17、29、43 的本机结果。先按 KG 宏平均，再报告三种子均值和样本标准差。Table 3 的 AlignKGC 和 SS-AGA 使用注明的方法特定协议。未经验收的结果保持待填。

审计证据位于 `runs/strict_baselines_20260905/results/table_fill_audit.json`。原始查询、排名、选中模型、训练曲线、代码与数据哈希均保留，数据集统计可从冻结的公开数据清单重新计算。

完整实施方案和绘图数据要求见 [执行计划](strict_baselines/EXECUTION_PLAN.zh.md)。当前完成情况以实时状态和表格为准。
''')
p=ROOT/'baselines/README.md'
t=p.read_text(encoding='utf-8')
t=t.replace('目前只完成代码下载和完整性检查，尚未安装环境、修改代码或运行实验。', '下表记录下载时的源码状态；当前本机复现进度见 `../reproduction/runs/strict_baselines_20260905/STATUS.zh.md`，实际运行实现以各任务配置中的源码哈希为准。')
write(p,t)
p=ROOT/'reproduction/ATRANSN_PROTOCOL_AUDIT.md'
write(p,p.read_text(encoding='utf-8').replace('全部十个基线','本阶段的基线'))
p=ROOT/'reproduction/queue/baseline_full_manifest.json'
legacy=json.loads(p.read_text(encoding='utf-8'))
legacy['suite']='frozen-9-baselines-representative-full-data-seed2020'
legacy['scope_note']='Legacy representative audit only. The current three-seed full-domain queue is reproduction/runs/strict_baselines_20260905/manifest.json.'
write(p,json.dumps(legacy,ensure_ascii=False,indent=2)+'\n')
p=ROOT/'refine-logs/LITERATURE_COMPARISON_LEDGER.md'
t=p.read_text(encoding='utf-8')
start=t.index('### 4.1 冻结的 9 个外部基线')
end=t.index('## 5.',start)
t=t[:start]+'''### 4.1 当前的 9 个外部基线

QURA-Cert 和内部对照不计入外部基线数。当前 Table 1–3 全部性能值采用本机完整三种子复现；本台账的文献数值仅供背景与出处核对。

| # | Method | 本阶段数据 | 进入方式 |
|---:|---|---|---|
| 1 | TransE | DBP/E/DWY/WK3l | Table 2，本机三种子 |
| 2 | DistMult | DBP/E/DWY/WK3l | Table 2，本机三种子 |
| 3 | RotatE | DBP/E/DWY/WK3l | Table 2，本机三种子 |
| 4 | AlignKGC | DBP/E | Table 3，方法特定协议，本机三种子 |
| 5 | SS-AGA | DBP/E | Table 3，方法特定协议，本机三种子 |
| 6 | LSMGA | DBP/E/DWY | Table 2，本机三种子 |
| 7 | ATransN | WK3l | Table 2，本机三种子 |
| 8 | DMKGC | DBP/E/DWY | Table 2，本机三种子 |
| 9 | IMKGC | DBP/E | Table 2，本机三种子；DWY 划分不兼容 |

具体命令、当前状态和逐格来源以正式队列与 `outputs/kbs_main_tables/cells_results.csv` 为准。

'''+t[end:]
t=t.replace('以下数字全部保留以方便引用，但只有标为 `strict candidate` 的行可进入严格已刊面板；`historical` 行必须与自有受控结果视觉分区。', '以下文献数字仅保留用于出处核对，不填入当前性能表。')
write(p,t)
write(ROOT/'outputs/paper_experiment_tables/README.zh.md', '# 当前实验表格\n\n当前正文表格统一位于 [kbs_main_tables](../kbs_main_tables/README.zh.md)，预览见 [KBS_Main_Text_Tables.pdf](../kbs_main_tables/KBS_Main_Text_Tables.pdf)。\n')

# Publish a fresh source package containing current files only.
entries=[TABLES/name for name in ['main.tex','preamble.tex','values.tex','cells_template.csv','cells_results.csv','README.zh.md','table_manifest.json','validation_report.json','build.ps1','validate_document.py','validate_current.py','KBS_Main_Text_Tables.pdf']]
entries+=sorted((TABLES/'tables').glob('*.tex'))
package=ROOT/'outputs/KBS_Main_Text_Tables_LaTeX.zip'
with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
    for p in entries:
        z.write(p,str(Path('kbs_main_tables')/p.relative_to(TABLES)))

with (ARCHIVE/'before/outputs/kbs_main_tables/cells_results.csv').open(encoding='utf-8-sig',newline='') as f:
    before={r['cell_id']:r for r in csv.DictReader(f) if r['value'] and '.kens.' not in r['cell_id']}
with (TABLES/'cells_results.csv').open(encoding='utf-8-sig',newline='') as f:
    after={r['cell_id']:r for r in csv.DictReader(f)}
for key,row in before.items():
    for field in ['value','standard_deviation','seed','run_id','checkpoint','source_type']:
        assert row[field]==after[key][field],(key,field)
assert len(before)==51
print(json.dumps({'preserved_filled_cells':len(before),'source_package_files':len(entries),'source_package':str(package)}))
