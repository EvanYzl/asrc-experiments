"""Versioned Chinese idea and claim-to-experiment design, before and after the study."""
import datetime
import json
import re
import shutil
from pathlib import Path
from frozen_data import ROOT
base=ROOT/'reproduction/sota';phase=base/'paper_support';plan=json.loads((phase/'PLAN.json').read_text());tables=ROOT/'outputs/kbs/_main/_tables'
result=json.loads((phase/'RESULTS.json').read_text()) if (phase/'RESULTS.json').exists() else None
stamp=datetime.datetime.now(datetime.timezone.utc);tag=stamp.strftime('%Y%m%d_%H%M%S');refine=ROOT/'refine-logs';refine.mkdir(exist_ok=True)
status=result['status'] if result else '已预注册；正在进行仅验证集训练'
sections=[('ASRC：对齐共享的效用、负迁移与计算成本','四个数据集的主表三种子确认已完成。当前工作扩展论文证据链，主表配置与结果继续冻结。ASRC 是 Alignment-selective Reciprocal ComplEx 的简称；“selective”表示由验证集选择数据集级共享或独立表示，不表示查询级门控。'),
('修改依据与版本','原 QURA-Cert 方案尚未训练效用门控，其风险证书、LOSO 效用标签、查询取源和门控成本均未产生可用结果。原 128 维 TransE 教师验证宏 MRR 为 DBP-5L 29.08%、E-PKG 46.48%、DWY 17.80%；这些结果支持先改善骨干，但不能推出未训练门控无效。原方案与本轮修改前的源文件、PDF 和表格均已归档。ASRC 采用已有 ComplEx、互逆增强和 N3 组件，当前贡献定位是可复核的参数共享研究，不把已有组件或工程组合包装为新理论。'),
('输入与参数共享','使用固定清单中的公开训练事实和原有实体对齐，不引入文本、外部知识或留出事实。按文件名与实体 ID 的稳定顺序对给定链接做并查集合并；若合并会使同一 KG 的两个不同候选实体共用表示，则拒绝并记录。共享表示始终保持每个 KG 内的候选可区分性。核心数据使用原有全局关系字典；WK3l 的英法关系分别偏移，不假设两套关系 ID 对应。独立对照为每个 KG 分配独立实体和关系参数。'),
('评分与训练','ComplEx 评分为 Re(sum_j e_hj r_j conjugate(e_tj))。复数秩为 256，即每个向量 512 个实数分量。训练对每条原始事实添加互逆事实和独立逆关系；每个批次在当前 KG 的全部尾实体候选上计算交叉熵，再加系数 0.01 的复数模长 N3 正则。采用 Adagrad，学习率 0.1，批量 512，初始化标准差 0.001。最多 40 个 epoch，每 5 个 epoch 在 val_select 上评估，连续 5 次无改善停止，验证宏 MRR 相同保留最早检查点。'),
('冻结主表与评测口径','DBP-5L、DWY、WK3l-15k 使用共享表示，E-PKG 使用独立表示；选择来自此前验证实验，本轮不重新选择。训练种子固定 17、29、43。所有查询使用目标 KG 全候选尾预测，分数相同按实体 ID 升序。验证过滤为 train+val_select；核心数据测试过滤为 train+valid，WK3l 为 all。先等权 KG 宏平均，再计算三种子均值与样本标准差。核心数据共 14 个评测 KG，WK3l 仅法语作为目标，英语提供训练支持。val_cert 未用于调参。'),
('主表已有结果','MRR / H@1 / H@10，单位均为 %，三种子均值±样本标准差：DBP-5L 74.88±0.03 / 65.78±0.05 / 89.29±0.07；E-PKG 56.41±0.21 / 43.99±0.33 / 76.17±0.06；DWY 63.70±0.02 / 54.43±0.06 / 78.58±0.14；WK3l-15k 51.52±0.23 / 41.92±0.15 / 70.63±0.39。12 项超过用户提供的暂定参考线。仍有暂代数值、未知 WK3l 图方法和部分基线重复次数不一致，因此只能称“达到暂定门槛”，不能宣称完整可比 SOTA。'),
('两个待检验主张','C1：固定互逆 ComplEx/N3 骨干之后，对齐参数共享产生可量化的查询收益与伤害，数据集级验证选择具有经验依据。C2：训练期参数共享会改变模型大小及全候选排序成本，其方向和幅度需要实际测量。两项均为经验主张；不预设所有数据集、所有查询或扰动条件均有收益。'),
('Table 4：同骨干迁移对照','四数据集各比较 Independent、Always shared 和冻结 ASRC。按同种子同查询配对秩，报告 MRR、相对 Independent 的 MRR 差、负迁移率 NTR=P(r>r_I)、正迁移率 PTR=P(r<r_I)，以及平均倒数排名伤害 E[max(1/r_I−1/r,0)]。后四项先逐 KG 统计再宏平均。按用户反馈改为每数据集一行：并列独立/共享 MRR，所有迁移项均比较共享与独立，末列指明冻结 ASRC 模式。省略定义上为零的自比较和完全重复的 ASRC 数值行，完整原始统计仍保留。共享即使平均提高 MRR，也可能伤害部分查询，不能解释成逐查询安全保证。'),
('Table 5：组件消融','DBP-5L 比较完整共享、全部独立、仅实体共享、仅关系共享、去互逆增强、去 N3。除被移除组件外，训练和选模规则不变；去互逆时重复原方向事实，匹配每轮优化步数，逆关系参数仍分配且计入参数量。该对照回答固定训练预算下的增强作用，不是最优单向模型的超参数搜索。报告三种子 MRR/H@1/H@10、相对完整模型的 MRR 差及参数量。若消融更好，也如实保留，不依据其测试分数替换主表。'),
('Table 6：预定义查询分组','依据干净对齐并查集定义查询头实体是否跨 KG 对齐，不看测试尾答案来定义组。四数据集同时报告已对齐与未对齐两组的查询数、独立与共享 MRR、差值和 NTR。每组仅对非空 KG 等权平均；空组为 N/A，不按零填充。完整保留两组，不能只挑有利语言或查询；组均值不必通过按查询数加权还原主表宏均值。'),
('Table 7：固定输入扰动','仅 DBP-5L，比较干净输入、每 KG 保留 50% 唯一训练事实组、每文件保留 50% 唯一对齐链接，以及将 10% 链接的第二端点确定性替换。扰动种子固定为 20260906，在三个训练种子间保持相同；所有派生索引、链接、哈希和冲突计数落盘。重复事实整体保留或移除。原始正例过滤集合仍完整固定，仅优化事实减少，因此此表研究训练信息减少，不声称严格零辅助信息的少样本设置。合成端点替换不等于经人工验证的真实错误链接；去重和碰撞拒绝后的实际数量均披露。共享模式不随扰动重新选择，独立模型不使用对齐，所以对齐减少/替换条件复用干净独立结果。'),
('Table 8：实测排序成本','训练与测试队列结束后，在无其他 GPU 作业的一张 RTX 2080 Ti 11GB 上串行测量。使用每 KG 固定 val_select 前 256 个查询，3 轮预热和 10 轮计时，FP32、4 CPU 线程。batch=1 测 p50/p95，batch=256 测吞吐；计时含查询张量构造、全候选打分、正例过滤、gold rank、稳定 top-10 和结果回传 CPU，排除模型/数据加载及磁盘写出。逐 KG 分位数/吞吐先宏平均，再做三种子统计。显存为 PyTorch 已分配峰值，RSS 为 0.1 秒采样进程峰值，各种子取 KG 最大峰值。完整报告参数量、显存、RSS、延迟和吞吐，不推断未测的冷启动、训练全生命周期或取源成本。'),
('预算、复用与停止规则','本轮共注册 48 个模型配置：14 次训练复用，34 次新训练；12 个已锁测试复用，36 个新配置各测试一次。先做 16 项输入一致性检查与 8 个单 epoch 验证运行检查，后者不计入论文。六卡并行独立训练和评测，串行执行 21 个独立计时配置。预计 2–4 GPU 小时，属于预算估计而非丢弃慢实验的理由。所有配置在新测试前登记并冻结，只按验证 MRR 选检查点，保留失败和负结果。完成表格、idea、设计、日志、原始结果与双端核验后停止，不追加主表优化或提高门槛。'),
('外部比较与局限','Table 1–3 的已有真实数据和比较项完整保留。Table 3 缺少的 SS-AGA 项、Table 2 尚未完成的外部比较继续如实标记，独立基线任务不由本轮重复启动。Table 5/7 的细粒度机制与扰动证据只来自 DBP-5L，不能自动推广到其余三个数据集。ASRC 的共享/独立二选一由此前验证小预算确定；本轮做确认性描述，不能将观察到的测试差异用于再选方案。'),
('参考依据','Lacroix, Usunier, Obozinski (ICML 2018), Canonical Tensor Decomposition for Knowledge Base Completion：互逆训练、张量分解及正则化的既有基础。https://proceedings.mlr.press/v80/lacroix18a.html 。Chen et al. (Findings EMNLP 2020), Multilingual Knowledge Graph Completion via Ensemble Knowledge Transfer：跨语言共享及对齐不一致的研究背景。https://aclanthology.org/2020.findings-emnlp.290/ 。')]
findings=[]
if result:
    t=result['tables']
    for r in t['T4']:
        if r['variant']=='shared':findings.append(f"{r['dataset']}：共享相对独立的 MRR 差为 {r['stats']['delta']['mean']:+.2f} 个百分点，NTR 为 {r['stats']['ntr']['mean']:.2f}%。")
    for r in t['T5']:
        if r['variant'] not in ['shared','independent']:findings.append(f"DBP-5L {r['variant']}：相对完整模型 MRR 差 {r['stats']['delta']['mean']:+.2f} 个百分点。")
    for r in t['T7']:
        if r['condition']!='clean' and r['mode']=='shared':findings.append(f"DBP-5L {r['condition']}：共享模型 MRR {r['stats']['mrr']['mean']:.2f}%，相对干净输入 {r['stats']['delta']['mean']:+.2f} 个百分点。")
    if t['T8']:
        for ds in ['dbp5l','depkg','dwy','wk3l']:
            rr={r['variant']:r for r in t['T8'] if r['dataset']==ds};a=rr['asrc']['stats'];b=rr['independent']['stats']
            findings.append(f"{ds} 排序成本：ASRC 参数 {a['params_m']['mean']:.2f}M（独立 {b['params_m']['mean']:.2f}M），p50 {a['p50_ms']['mean']:.2f} ms（独立 {b['p50_ms']['mean']:.2f} ms），吞吐 {a['qps']['mean']:.1f} queries/s（独立 {b['qps']['mean']:.1f}）。")
    interpretation='DBP-5L 仅关系共享几乎不能还原完整收益，主要增益依赖实体共享；仅实体共享仍落后完整模型约 1.01 个百分点。去掉互逆增强反而提高测试均值约 0.30 个百分点，因此本轮不支持“互逆增强是性能提升的必要条件”；该负面组件证据保留，冻结主表不据此更换。E-PKG 的已对齐头组共享收益为 +2.56 个百分点，未对齐头组为 −3.16 个百分点，提示整体小幅负收益掩盖了组间差异。E-PKG ASRC 的 NTR=0 仅因为它复用 Independent，不能当作查询级安全证书。对齐减半与 10% 合成端点替换使 DBP-5L 共享 MRR 分别降低约 11.71 和 7.82 个百分点，说明方法依赖对齐覆盖和质量。'
    sections.insert(-2,('本轮结果与解释','\n\n'.join(findings)+'\n\n'+interpretation+'\n\n以上均来自预注册的固定对照。三种子标准差反映本次训练重复性，不能替代跨数据集外推或显著性检验。排序成本方面，DBP-5L 参数从 33.89M 降至 16.06M，但 p50 几乎相同；WK3l 的共享模型在本次固定顺序计时中 p50 更高、吞吐略低，不能宣称普遍加速。DBP 输入审计显示 37,592 条唯一对齐；减半后保留 18,795 条，噪声条件替换 3,755 个端点并出现 4,464 次冲突拒绝，实际扰动计数与派生输入均保留。原始结果和逐查询配对数据保留在 paper_support 中。'))
md='# '+sections[0][0]+'\n\n状态：'+status+'。\n\n'+sections[0][1]+'\n\n'+'\n\n'.join('## '+title+'\n\n'+body for title,body in sections[1:])+'\n'
idea=ROOT/'work/ideaspark/_run/multidomain-kgc-local/_2/phase4';history=phase/'document_versions'/tag;history.mkdir(parents=True)
(history/'idea.std.zh.md').write_text(md,encoding='utf-8');shutil.copy2(history/'idea.std.zh.md',idea/'idea.std.zh.md')
def escape(s):
    return ''.join({'\\':r'\textbackslash{}','%':r'\%','_':r'\_','&':r'\&','#':r'\#','{':r'\{','}':r'\}','$':r'\$','^':r'\textasciicircum{}','~':r'\textasciitilde{}'}.get(c,c) for c in s)
preamble=(ROOT/plan['old_version_archive']/'idea/idea.std.zh.tex').read_text(encoding='utf-8').split(r'\begin{document}')[0]
preamble+=r'\usepackage[hidelinks]{hyperref}'+'\n'
tex=preamble+r'\begin{document}'+'\n'
for i,(title,body) in enumerate(sections):
    tex+=('\\section*' if i==0 else '\\subsection*')+'{'+escape(title)+'}\n\n'
    # URL tokens use breakable \url markup; prose stays Chinese.
    pieces=re.split(r'(https://\S+)',body)
    prose=''.join('\\url{'+x+'}' if x.startswith('https://') else escape(x) for x in pieces)
    if title=='评分与训练':
        prose=prose.replace(escape('Re(sum_j e_hj r_j conjugate(e_tj))'),r'\(\operatorname{Re}\!\left(\sum_{j=1}^{d} e_{h,j}r_j\overline{e_{t,j}}\right)\)')
        prose+='\n'+r'\[\mathcal{L}_{B}=\frac{1}{|B|}\sum_{(h,r,t)\in B}\left[-\log\frac{\exp s(h,r,t)}{\sum_{u\in\mathcal{E}_{\mathrm{KG}}}\exp s(h,r,u)}+\lambda\sum_{j=1}^{d}\left(|e_{h,j}|^3+|r_j|^3+|e_{t,j}|^3\right)\right],\quad \lambda=0.01.\]'+'\n'
    tex+=prose+'\n\n'
tex+=r'\end{document}'+'\n';(history/'idea.std.zh.tex').write_text(tex,encoding='utf-8');shutil.copy2(history/'idea.std.zh.tex',idea/'idea.std.zh.tex')
design='# ASRC 支持实验设计（Table 4–8）\n\n状态：'+status+'。本轮用户授权扩展其他表格；Table 2 冻结，外部基线不重复启动。\n\n'
design+='## Claim–evidence map\n\n| 主张 | 必要实验 | 支持证据 | 负结果的解释 |\n|---|---|---|---|\n| C1 共享收益与伤害可量化 | Table 4 同骨干配对；Table 5 组件；Table 6 分组；Table 7 扰动 | 固定三种子、配对原始秩，所有条件同时报告 | 平均无增益、局部高 NTR、扰动失败均限制共享适用范围，不删行、不重选主表 |\n| C2 共享改变模型和排序成本 | Table 8 固定查询实测 | 参数量、显存/RSS、延迟和吞吐完整报告 | 参数更少未必延迟更低；若无加速，撤回速度主张 |\n\n'
design+='\n\n'.join('## '+title+'\n\n'+body for title,body in sections[3:] if title!='参考依据')
design+='\n\n## Run matrix 与预算\n\n| 模型块 | 配置数 | 复用训练 | 新训练 | 新测试 |\n|---|---:|---:|---:|---:|\n| 四数据集独立/共享，三种子 | 24 | 14 | 10 | 12 |\n| DBP 四个组件消融，三种子 | 12 | 0 | 12 | 12 |\n| DBP train50 双模式、align50/noise10 共享 | 12 | 0 | 12 | 12 |\n| 合计 | 48 | 14 | 34 | 36 |\n\n另有 16 项 CPU 输入检查、8 项单 epoch GPU 实现检查，不进入论文；21 个串行计时任务仅访问验证查询。每个正式训练最高 40 epoch，全候选 CE；队列为六卡一作业一卡，按空闲显存分配且同阶段不复用忙卡。计时阶段与训练/测试互斥。\n\n## 可复核文件\n\nPLAN.json 记录全部配方；TRAINING_SOURCE_FREEZE.json、EVALUATION_FREEZE.json 固定源码、清单和检查点；每个训练保存 config/result、学习曲线、best.pt、派生输入和验证秩；每个测试用 TEST_OPENED.json 防止重开并保存原始三元组、三种过滤秩及 top-10。RESULTS.json 保存逐种子统计和原始文件哈希。SOTA_LOG.md 记录每项实验前后、PID、GPU 和结果路径。旧文档归档路径：'+plan['old_version_archive']+'。\n'
tracker='# ASRC 实验跟踪\n\n阶段：'+status+'。\n\n- 主表：四数据集 12 个三种子结果已完成并锁定。\n- 新范围：Table 4–8，共 34 新训练、36 新单次测试、21 串行计时；14 训练和 12 锁定测试复用。\n- 输入检查：INPUT_CHECKS.json；阶段状态：wave1、smoke_queue、wave2、test_queue 各 state.json。\n- 结果汇总：paper_support/RESULTS.json；负结果必须保留。\n- 待外部任务完成：现有 Table 2/3 中未知基线，不通过本轮新增基线作业处理。\n'
if result:tracker+='\n## 已观测结果\n\n'+'\n'.join('- '+f for f in findings)+'\n'
for name,content in [('EXPERIMENT_PLAN',design),('EXPERIMENT_TRACKER',tracker)]:
    version=refine/f'{name}_{tag}.md';version.write_text(content,encoding='utf-8');shutil.copy2(version,refine/f'{name}.md')
    with (ROOT/'MANIFEST.md').open('a',encoding='utf-8') as f:f.write(f'| {stamp.isoformat()} | experiment-plan | {version.relative_to(ROOT).as_posix()} | experiment | ASRC Table 4–8：版本化计划/跟踪，保留主表冻结与全部负结果 |\n')
(tables/'EXPERIMENT_DESIGN.zh.md').write_text(design,encoding='utf-8')
print(json.dumps({'status':status,'version':tag,'idea':str(idea/'idea.std.zh.tex'),'plan':str(refine/'EXPERIMENT_PLAN.md')},ensure_ascii=False))
