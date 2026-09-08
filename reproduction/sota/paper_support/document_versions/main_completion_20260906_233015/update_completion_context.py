"""Record the fixed Table 2 comparison and update the current document status."""
import csv
import datetime
import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path

ROOT = Path('G:/zhishitupui')
TABLES = ROOT / 'outputs/kbs/_main/_tables'
BASE = ROOT / 'reproduction/language_table_seed17/main_completion_20260907'
ARCHIVE = Path(__file__).resolve().parent
IDEA = ROOT / 'work/ideaspark/_run/multidomain-kgc-local/_2/phase4'
LEGACY = ROOT / 'work/ideaspark_run/multidomain-kgc-local_2/phase4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


with (TABLES / 'cells_results.csv').open(encoding='utf-8-sig', newline='') as stream:
    rows = list(csv.DictReader(stream))
comparison = {}
for dataset in ('dbp', 'epkg', 'dwy', 'wk3l'):
    for metric in ('mrr', 'h1', 'h10'):
        selected = [r for r in rows if r['cell_id'].startswith(f'T2.{dataset}.')
                    and r['cell_id'].endswith('.' + metric)]
        assert all(r['value'] and r['source_type'] == 'rerun' and r['seed'] == '17;29;43'
                   for r in selected), selected
        ours = next(r for r in selected if '.asrc.' in r['cell_id'])
        baselines = [r for r in selected if r != ours]
        best = max(Decimal(r['value']) for r in baselines)
        gap = (Decimal(ours['value']) - best) * 100
        comparison[f'{dataset}.{metric}'] = {
            'ours': ours['value'], 'baseline_max': str(best),
            'baseline_cells': [r['cell_id'] for r in baselines if Decimal(r['value']) == best],
            'gap_percentage_points': str(gap), 'strict_win': gap > 0,
            'within_half_point': gap >= Decimal('-0.5'),
            'pending_cells': [], 'baseline_rows': baselines, 'ours_row': ours,
        }
wins = sum(c['strict_win'] for c in comparison.values())
passed = wins >= 10 and all(c['within_half_point'] for c in comparison.values())
assert wins == 12 and passed
report = {
    'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status': 'fixed_table2_threshold_met',
    'claim_zh': '在固定 Table 2 比较范围内，三种子均值 12/12 指标严格领先，达到预设门槛。',
    'scope': 'All pre-existing Table 2 comparison rows; no rows removed. Method-specific Table 3 remains outside this frozen comparison.',
    'comparison_frozen': True, 'pending_table2_cells': [], 'provisional_reference_cells': [],
    'strict_wins': wins, 'total_metrics': 12, 'threshold_met': passed,
    'threshold': 'At least 10 strict wins; every other gap >= -0.5 percentage points',
    'comparison_precision': 'Decimal of all stored unrounded means; percent only for display',
    'protocol': 'Fixed input/split/recipes; validation-only checkpoint selection; previously locked test outcomes reused; seeds 17/29/43 mean and sample SD',
    'csv_sha256': sha(TABLES / 'cells_results.csv'),
    'new_results_sha256': sha(BASE / 'PUBLICATION_RESULTS.json'),
    'original_asrc_results_sha256': sha(ROOT / 'reproduction/sota/three_seed/RESULTS.json'),
    'remaining_table3_cells': [r['cell_id'] for r in rows if r['table_id'] == 'T3' and not r['value']],
    'columns': comparison,
    'scope_stop': 'No new training, inference, or SOTA optimization launched in this publication. The separately registered SS-AGA/E-PKG queue continues.',
}
write_json(BASE / 'CURRENT_COMPARISON.json', report)
write_json(ARCHIVE / 'CURRENT_COMPARISON.json', report)
manifest_path = TABLES / 'table_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['date'] = '2026-09-07'
manifest['current_comparison'] = (BASE / 'CURRENT_COMPARISON.json').relative_to(ROOT).as_posix()
manifest['current_comparison_sha256'] = sha(BASE / 'CURRENT_COMPARISON.json')
manifest['table2_complete'] = True
manifest['table2_strict_wins'] = wins
manifest['table2_threshold_met'] = passed
manifest['comparison_scope'] = report['scope']
write_json(manifest_path, manifest)

old_main = '12 项超过当前已完成复现项的逐列最高值。WK3l 三种图方法仍未知，因此只能称“达到暂定门槛”，不能宣称完整可比 SOTA。'
new_main = '固定 Table 2 的全部比较项现已完成本地三种子复现，暂代值与未知项均已补齐。按未四舍五入的均值逐列比较，ASRC 在 12/12 项指标上严格领先，达到至少 10 项严格领先、其余每项落后不超过 0.5 个百分点的预设门槛。该结论限于已冻结的比较项、数据划分、输入和评测协议。'
old_graph = '当前比较记录已用这些复现均值替换对应暂代值；ASRC 仍在 12 个已有参考指标上严格领先，但 WK3l 三种图方法仍未知，结论仍限于“达到暂定门槛”。本阶段补种子实验结束，不再追加优化。'
new_graph = '随后，独立补齐任务完成 WK3l 的三种图方法：LSMGA 为 34.50±0.80 / 21.88±0.88 / 60.14±0.98，DMKGC 为 40.26±0.18 / 28.29±0.16 / 65.22±0.20，IMKGC 为 37.33±0.27 / 23.81±0.25 / 64.40±0.29。WK3l 按已冻结协议仅报告 FR 目标，采用 all 过滤。Table 2 已无未知比较项；以完整精度均值重新冻结逐列最大参考值后，ASRC 为 12/12 项严格领先。主表补齐与验收结束，不再追加优化。'
old_limits = 'Table 1–3 的已有真实数据和比较项完整保留。Table 3 缺少的 SS-AGA 项、Table 2 尚未完成的外部比较继续如实标记，独立基线任务不由本轮重复启动。'
new_limits = 'Table 1–3 的已有真实数据和比较项完整保留。Table 2 已补齐；Table 3 的 SS-AGA/DBP-5L 三种子结果为 7.81±0.44 / 3.47±0.33 / 15.36±0.66（MRR / H@1 / H@10，%）。SS-AGA 使用已登记的 train 过滤、supporter 验证事实和重新生成的 mBERT 标签特征，保留在原定方法专属比较中，不纳入 Table 2 排名。Table 3 仅 SS-AGA/E-PKG 尚待原独立队列完成。'
for path in [IDEA / 'idea.std.zh.md', IDEA / 'idea.std.zh.tex', TABLES / 'EXPERIMENT_DESIGN.zh.md']:
    text = path.read_text(encoding='utf-8')
    for old, new in [(old_main, new_main), (old_graph, new_graph), (old_limits, new_limits)]:
        assert text.count(old) == 1, (path, old)
        if path.suffix == '.tex':
            new = new.replace('%', r'\%')
        text = text.replace(old, new)
    if path.name == 'EXPERIMENT_DESIGN.zh.md':
        text += '\n## 2026-09-07 主表完成与本次回填\n\n'
        text += '本次仅核验已有输出，不启动训练或推理。独立 39 项队列的 21 项已完成并回传，复用 WK3l 三个 seed17，合计审计 24 项原始结果。新增 Table 2 的 9 个 WK3l 三种子均值与标准差，以及 Table 3 的 SS-AGA/DBP-5L 三项指标和种子记录。Table 2 所有指标以未舍入值确定最优和次优，显示保留两位小数。\n\n'
        text += '可追溯证据位于 reproduction/language_table_seed17/main_completion_20260907：PUBLICATION_RESULTS.json、PUBLICATION_SOURCE_RECEIPT.json、publication_acceptance/、CURRENT_COMPARISON.json；原始返回结果保存在 returned_jobs/，WK3l 原 seed17 保存在 reproduction/language_table_seed17/jobs/。旧比较快照仍保留在 graph_three_seed/CURRENT_COMPARISON.json，本次完整比较以此处的新快照为准。SS-AGA/E-PKG 尚有 6 项运行、12 项排队，继续使用此前冻结的 25 轮预算；本次不调整其配置。\n'
    path.write_text(text, encoding='utf-8')
for ext in ('md', 'tex'):
    shutil.copy2(IDEA / f'idea.std.zh.{ext}', LEGACY / f'idea.std.zh.{ext}')
print(json.dumps({'strict_wins': wins, 'table2_complete': True,
                  'baseline_maxima': {k: v['baseline_cells'] for k, v in comparison.items()}}, ensure_ascii=False))
