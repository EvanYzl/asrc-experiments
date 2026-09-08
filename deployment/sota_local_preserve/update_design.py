"""Update the active idea and Table 2 design; retain the archived original idea."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
canonical=ROOT/'work/ideaspark/_run/multidomain-kgc-local/_2/phase4'
legacy=ROOT/'work/ideaspark_run/multidomain-kgc-local_2/phase4'
text=r'''# 对齐共享的互逆复数分解：Table 2 实验方案

**当前候选：** Alignment-selective Reciprocal ComplEx（ASRC）。状态：种子 17 按数据集顺序验收；复用已完成初筛，仅使用单种子结果，未宣称 SOTA 或方法创新。

## 修改依据与版本

2026-09-06 根据当前任务，将目标收敛为 Table 2 的十二项指标。原 QURA-Cert 的完整 idea、Markdown、TeX 和三页 PDF 保存在 `reproduction/sota/history/20260906T042743Z/phase4`。原方案的仅目标骨干为 128 维归一化 TransE；已冻结等权教师的 seed 17 验证宏 MRR 为 DBP-5L 29.08%、E-PKG 46.48%、DWY 17.80%。这说明首先需要改善骨干。尚未训练效用门控，不能据此断言门控无效。

当前重设计采用已有文献中的 ComplEx、互逆关系训练及 N3 正则化，并验证显式对齐共享的跨图谱作用。它是一个可检验的工程候选，不将已有组件包装为新理论，也不继承原方案尚未验证的安全证书或资源优势声明。

## 输入与模型

只读取固定公开训练三元组与用户已有实体对齐链接。四数据集的实体 ID、关系 ID、训练/验证/测试划分及十六份清单保持不变。不读取文本特征、不增加外部训练数据、不用留出三元组建图。

按文件名和实体 ID 的固定顺序处理给定对齐。用并查集为一致对齐实体分配共享表示；若合并会使同一 KG 中两个不同候选实体变成同一表示，则拒绝该条合并并记录。不会依据验证或测试正确答案修改对齐规则。独立对照保留各 KG 的单独实体和关系表示。

核心数据已有统一关系字典，因此共享模型沿用该字典。WK3l 的两个关系字典分别编号，使用可逆偏移保持互不混淆；其原始 ID 在保存查询与评测时恢复。训练为每个原始事实加入一条反向事实，并为反向关系分配独立参数。反向事实完全由训练事实生成。

采用复数实体及关系向量，评分为

$$s(h,r,t)=\operatorname{Re}\sum_j e_{h,j}w_{r,j}\overline{e_{t,j}}.$$

训练对每个查询，在所属 KG 的全部候选实体上计算交叉熵，并对头、关系、尾复数向量的模施加三次方正则项。首轮设置：复数秩 256（512 个实数）、N3 系数 0.01、Adagrad 学习率 0.1、批量 512、初始化标准差 0.001，最多 40 epoch，每 5 epoch 验证，连续 5 次未改善则早停。首轮六项配置已完成。按验证宏 MRR，DBP-5L、DWY、WK3l 采用共享表示，E-PKG 采用独立表示。按照用户最新目标，只使用种子 17，先 DBP-5L，再 E-PKG、DWY、WK3l。复用已完成的各数据集初筛，不重复训练；每次只推进当前数据集。旧目标下已完成的 29、43 种子验证产物仅作为历史留存，不参与本阶段验收。

## 选模、冻结与验收

全部候选使用原 `val_select`，过滤训练集加 `val_select` 的其他正确尾实体；不得使用 `val_cert` 或测试标签选模。实体全集、尾实体预测、同分按实体 ID 升序，与严格基线一致。DBP-5L、E-PKG、DWY 分别对 5、6、3 个 KG 等权宏平均；WK3l 只评测 FR，EN_F 为支持源。以验证宏 MRR 选择检查点，同分保留较早检查点。

先用种子 17 小预算验证筛选；任何时刻只对一个数据集搜索方法。验证证据充分后，冻结该数据集的模型结构、超参数、训练预算、检查点及比较参考。候选训练程序不加载测试三元组，独立评测程序只对这个固定检查点执行一次测试。测试后锁定结果，不依据测试重新调参或选模型，再迁移到下一数据集。四套数据保持同一 ASRC 算法定义，允许只凭新数据集验证集调整共享开关等超参数。最终测试：前三套用 train+valid 过滤，WK3l 用 all 过滤；保持当前 gold、全候选与同分规则。

比较保留 Table 2 全部原有外部方法。本阶段不补基线，不补其他种子；直接复用本地复现。非星号参考使用未四舍五入原值；用户最新星号参考单列保存，不能标为复现数值。参考 MRR/H@1/H@10（%）为 DBP-5L 50.47/37.31/约78.88*；E-PKG 46.40*/约33.03*/约69.55*；DWY 44.22/33.02/约64.97*；WK3l 42.11/31.89/64.86。WK3l 三种图方法仍未知，不能视为已经超过。比较口径和重复次数差异必须披露。

验收为十二项中至少十项严格大于相应最大基准，其余每项落后不超过 0.5 个百分点。若仍有未知或暂代基线，仅能报告达到暂定门槛。每个数据集争取三项超过，但不擅自将总体门槛提高为 12/12。若锁定的测试使总体门槛不可能满足，立即停止等待，不依据测试重试。四数据集单种子结果达到门槛且全部表格、idea、设计、日志和原始结果回传核验后，报告“单种子达到暂定门槛”，立即停止本任务实验，不自动补基线、补种子或继续提高分数。

## 执行与原始记录

复用服务器已验收环境和六张 RTX 2080 Ti。独立任务按空闲显存调度，队列保存命令、种子、PID、GPU、退出码及结果路径。每步前后追加 `SOTA_LOG.md`；失败尝试保留。验证逐查询排名、完整精度的统计、训练曲线、检查点、输入/代码哈希及同步清单构成验收证据。Table 4–8 不属于本任务，不新增实验。

## 参考文献

- Lacroix, Usunier, Obozinski. Canonical Tensor Decomposition for Knowledge Base Completion. ICML 2018. https://proceedings.mlr.press/v80/lacroix18a.html
- Chen et al. Multilingual Knowledge Graph Completion via Ensemble Knowledge Transfer. Findings EMNLP 2020. https://aclanthology.org/2020.findings-emnlp.290/
'''
preamble=(legacy/'idea.std.zh.tex').read_text(encoding='utf-8').split(r'\begin{document}')[0]
def escape(s):
    return s.replace('\\','\\textbackslash{}').replace('&','\\&').replace('%','\\%').replace('_','\\_').replace('#','\\#')
lines=[]
for line in text.splitlines():
    if line.startswith('# '):lines.append(r'\section*{'+escape(line[2:])+'}')
    elif line.startswith('## '):lines.append(r'\subsection*{'+escape(line[3:])+'}')
    elif line.startswith('$$'):lines.append(r'\['+line[2:-2]+r'\]')
    else:lines.append(escape(line.replace('**','').replace('`',''))+'\n')
tex=preamble+'\\begin{document}\n'+'\n'.join(lines)+'\n\\end{document}\n'
for folder in [canonical,legacy]:
    (folder/'idea.std.zh.md').write_text(text,encoding='utf-8')
    (folder/'idea.std.zh.tex').write_text(tex,encoding='utf-8')
design=ROOT/'outputs/kbs/_main/_tables/EXPERIMENT_DESIGN.zh.md'
design.write_text(text,encoding='utf-8')
table=ROOT/'outputs/kbs/_main/_tables/tables/t02.tex'
s=table.read_text(encoding='utf-8').replace(r'\textbf{QURA-Cert}',r'\textbf{ASRC (candidate)}')
table.write_text(s,encoding='utf-8')
print('Updated canonical and legacy Chinese idea sources, canonical Table 2 design; original archive retained.')
