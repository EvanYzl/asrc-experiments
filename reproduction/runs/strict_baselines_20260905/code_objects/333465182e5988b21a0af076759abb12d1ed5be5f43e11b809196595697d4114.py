"""Independently replay corrected KEnS votes and frozen target checkpoint scores."""
import argparse
import json
from pathlib import Path

from audit_kens_predictions import RUN,audit_one,sha256

REVISION='kens-transe-paper-loss-v2'


def audit_group(group):
    rows=[]
    for path in sorted(Path(group).glob('*/result.json')):
        result=json.loads(path.read_text(encoding='utf-8'))
        if result.get('status')!='completed' or result.get('training_revision')!=REVISION:continue
        receipt=RUN/'results/kens_v2_prediction_checks'/f'{path.parent.parent.name}_{result["kg"]}.json'
        config=json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
        vote_hash=next(v for k,v in config['source_hashes'].items() if k.replace('\\','/').endswith('/src/ensemble.py'))
        checkpoint_hashes={k.replace('\\','/'):v for k,v in result['checkpoints'].items()}
        key={'query_sha256':result['query_sha256'],'checkpoint_sha256':checkpoint_hashes[f'model/{result["kg"]}.h5'],
             'original_vote_source_sha256':vote_hash,'weight_sha256':result['weight_sha256']}
        old=json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {}
        if old.get('passed') and all(old.get(k)==v for k,v in key.items()):row=old
        else:
            row=audit_one(path)
            assert all(row[k]==v for k,v in key.items() if k!='weight_sha256')
            assert sha256(path.parent/'ensemble_weights.npz')==key['weight_sha256']
            row.update(passed=True,training_revision=REVISION,full_data=result['full_data'],run_id=path.parent.parent.name,**key)
            receipt.parent.mkdir(parents=True,exist_ok=True)
            receipt.write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8')
        rows.append(row)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--group');args=parser.parse_args()
    groups=[Path(args.group)] if args.group else sorted((RUN/'jobs').glob('kens_v2_*'))
    rows=[row for group in groups for row in audit_group(group)]
    summary={'status':'passed' if rows else 'waiting','audited_kg_runs':len(rows),'audited_queries':sum(row['queries'] for row in rows),'rows':rows,
             'interpretation':'Vote/checkpoint consistency is checked independently of the training-formula regression checks. Scores and head-return rates are descriptive; no score threshold selects a recipe.'}
    if not args.group:
        (RUN/'results/kens_v2_prediction_audit.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
