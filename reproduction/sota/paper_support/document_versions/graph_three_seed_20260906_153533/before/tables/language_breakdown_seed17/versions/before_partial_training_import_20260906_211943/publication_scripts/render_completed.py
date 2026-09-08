"""Re-render accepted results without training, inference, or importing new runs."""
from pathlib import Path
import ast
import json

ROOT=Path('G:/zhishitupui')
TABLE=ROOT/'outputs/kbs/_main/_tables/language_breakdown_seed17'


def main():
    data=json.loads((TABLE/'language_kg_seed17.json').read_text(encoding='utf-8'))
    assert data.get('completion_batch') and all(r['status']=='completed' for rs in data['rows'].values() for r in rs)
    script=(TABLE/'build_table.py').read_text(encoding='utf-8');constants={}
    for node in ast.parse(script).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['SPECS','METRICS']:
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    constants['SPECS']['wk3l']=('WK3l-15k',[('en_f',r'EN\_F (supplemental)'),('fr','FR (primary)')])
    ns=dict(constants,OUT=TABLE,method_rows=data['rows'],supplement=True)
    render=script[script.index('def column_value('):script.index('for rel, item in raw_sources.items():')]
    exec(compile(render,str(TABLE/'build_table.py')+' [completed results]','exec'),ns)
    path=TABLE/'table_language_kg_seed17.tex';tex=path.read_text(encoding='utf-8')
    old='TransE (EN teacher) is the reused ATransN source teacher (margin 4, batch 1024), with no FR/AVG entry; it is not an English-target ATransN result.'
    new='TransE (teacher) uses the source-teacher recipe (margin 4, batch 1024) in both languages. English ATransN uses FR-to-EN transfer. Graph methods use a WK3l data-format adaptation with fixed recipes; one FR-validation-selected checkpoint serves both languages.'
    assert old in tex;tex=tex.replace(old,new)
    old='P: seed-17 result pending; NR: no accepted result; \\textit{n/a}: not applicable. All filled values were verified against saved ranks and source JSON. English ASRC inference was added using frozen weights; no model was retrained.'
    new='All displayed results are complete seed-17 runs, verified against saved ranks. Missing baselines were trained with fixed recipes and validation-only checkpoint selection; existing ASRC and baseline results were reused.'
    assert old in tex;path.write_text(tex.replace(old,new),encoding='utf-8')


if __name__=='__main__':main()
