"""Verify the completed, emphasized PDF against the unrounded table data."""
from pathlib import Path
from collections import Counter
import datetime as dt
import hashlib
import json
import re
import shutil
import pymupdf

ROOT=Path('G:/zhishitupui')
BASE=Path(__file__).resolve().parent
TABLE=ROOT/'outputs/kbs/_main/_tables/language_breakdown_seed17'
data=json.loads((TABLE/'language_kg_seed17.json').read_text(encoding='utf-8'))
expected=[]
for rows in data['rows'].values():
    for row in rows:
        assert row['status']=='completed'
        for group in ['per_kg','supplemental_kg_results']:
            for v in row.get(group,{}).values():expected.extend(f'{v[m]*100:.2f}' for m in ['h1','h10','mrr'])
        expected.append(f"{row['macro']['mrr']*100:.2f}")
assert len(expected)==378
tex=(TABLE/'table_language_kg_seed17.tex').read_text(encoding='utf-8')
assert not re.search(r' & (?:NR|P|\\textit\{n/a\})(?: & | \\\\)',tex)
bold_expected=len(re.findall(r'\\textbf\{[0-9]+\.[0-9]{2}\}',tex))
underline_expected=len(re.findall(r'\\underline\{[0-9]+\.[0-9]{2}\}',tex))
columns=[]
blocks=re.findall(r'\\begin\{tabularx\}.*?\\end\{tabularx\}',tex,re.S)
assert len(blocks)==4
for (ds,rows),block in zip(data['rows'].items(),blocks):
    keys=['en_f','fr'] if ds=='wk3l' else list(rows[0]['per_kg'])
    for kg,m in [(kg,m) for kg in keys for m in ['h1','h10','mrr']]+[('AVG','mrr')]:
        values=[]
        for row in rows:
            v=row['macro'] if kg=='AVG' else row['supplemental_kg_results'][kg] if kg=='en_f' else row['per_kg'][kg]
            values.append((row['method'],v[m]))
        levels=sorted({v for _,v in values},reverse=True)
        idx=3*len(keys) if kg=='AVG' else 3*keys.index(kg)+['h1','h10','mrr'].index(m)
        for method,v in values:
            label=r'\textbf{ASRC}' if method=='ASRC' else method
            line=next(line for line in block.splitlines() if line.startswith(label+' & '))
            cell=line.split(' & ')[idx+1].removesuffix(' \\\\').strip();number=f'{v*100:.2f}'
            wanted=r'\textbf{'+number+'}' if v==levels[0] else r'\underline{'+number+'}' if len(levels)>1 and v==levels[1] else number
            assert cell==wanted,(ds,kg,m,method)
        columns.append({'dataset':ds,'kg':kg,'metric':m,'best':[n for n,v in values if v==levels[0]],'second':[n for n,v in values if len(levels)>1 and v==levels[1]]})
pdf=TABLE/'build/main.pdf';doc=pymupdf.open(pdf);assert len(doc)==1
page=doc[0];content=page.get_text()
actual=re.findall(r'(?<![0-9.])[0-9]+\.[0-9]{2}(?![0-9.])',content)
assert Counter(actual)==Counter(expected)
assert not re.search(r'Overfull|Missing character|Float too large',(TABLE/'build/main.log').read_text(encoding='utf-8',errors='replace'))
bold_actual=[]
for block in page.get_text('dict')['blocks']:
    for line in block.get('lines',[]):
        for span in line['spans']:
            if 'bold' in span['font'].lower():bold_actual.extend(re.findall(r'[0-9]+\.[0-9]{2}',span['text']))
underlines=[item for drawing in page.get_drawings() if .35<drawing['width']<.45 for item in drawing['items'] if item[0]=='l' and abs(item[1].y-item[2].y)<.05 and 10<abs(item[2].x-item[1].x)<30]
assert len(bold_actual)==bold_expected and len(underlines)==underline_expected
boxes=[b[:4] for b in page.get_text('blocks') if b[6]==0]
bbox=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
assert bbox[0]>=20 and bbox[1]>=15 and bbox[2]<=page.rect.width-20 and bbox[3]<=page.rect.height-15
page.get_pixmap(matrix=pymupdf.Matrix(1.8,1.8),alpha=False).save(TABLE/'preview_page_1.png');doc.close()
shutil.copy2(pdf,TABLE/'All_Datasets_Language_KG_Seed17.pdf')
receipt={'status':'passed','timestamp':dt.datetime.now(dt.timezone.utc).isoformat(),'seed':17,'pages':1,
         'numeric_cells_verified':378,'missing_cells':0,'bold_cells':bold_expected,'underlined_cells':underline_expected,
         'warnings':[],'content_bbox':bbox,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
         'pdf':str(TABLE/'All_Datasets_Language_KG_Seed17.pdf')}
for path in [TABLE/'PDF_CHECK.json',BASE/'DELIVERY_VERIFICATION.json']:
    path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(TABLE/'HIGHLIGHT_AUDIT.json').write_text(json.dumps({**receipt,'ranking':'unrounded values; best bold, second distinct value underlined; exact ties share the same style','columns':columns},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
readme='''# 全数据集语言／KG 单种子汇总表

四套数据集的语言／KG 结果已汇总，固定 seed 17，共 378 个数值，无待填数值。每列按未四舍五入的成绩比较，最优加粗、次优加下划线，原值并列共享标记。

本次补充训练包括英语 TransE、DistMult、RotatE，法语 TransE 教师，FR→EN 的 ATransN，以及 WK3l 上 LSMGA、DMKGC、IMKGC，共 8 个任务。IMKGC/E-PKG 复用原本正在运行的 seed17 任务完成后的结果。所有原先已填成绩保留。

TransE (teacher) 一行现在汇总英法两个源教师，均采用 margin 4、batch 1024 的教师配置，与普通 TransE 分开。ATransN 的 EN_F 为 FR→EN 迁移；FR 仍为原始 EN→FR 实验。图方法用 FR 验证集选择同一权重，并评测英法两侧。

EN_F 仍属补充评测，WK3l 的 AVG 继续只统计 FR。原始划分和评测规则保持固定：全候选尾实体排序，WK3l 使用 all 过滤，原值按确定性的实体 ID 处理并列。其他三套数据保持 train+valid 过滤。只用验证集选择权重。

EN_F 原始测试含 31 条训练重叠、15 条验证重叠查询；表内为原划分成绩，排除这 46 条后的 40,654 条查询诊断见 reproduction/language_table_seed17/EN_F_overlap_excluded.csv。

main.tex 为编译入口；build_table.py 重新渲染当前已核验结果，不启动训练。实验配置、隔离代码、进程、训练日志、权重、原始排名和回填校验保存在 G:/zhishitupui/reproduction/language_table_seed17。PDF_CHECK.json、DATA_AUDIT.json 和 HIGHLIGHT_AUDIT.json 记录数值与版面核验。版本备份见 versions。

本表为单种子语言分解补充表，不替代主表的三种子比较或验收。
'''
(TABLE/'README.zh.md').write_text(readme,encoding='utf-8')
(TABLE/'AVAILABILITY.zh.md').write_text('# 当前可用性\n\n全部 378 个数值已填满并核验。没有尚需启动的训练任务。详情见 README.zh.md，原始结果见 reproduction/language_table_seed17。\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))
