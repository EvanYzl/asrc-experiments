"""Replay released KEnS voting and inspect the observed low Hits@1 result.

This audit never alters trained weights, nominees, or the published metric.
Run in the isolated KEnS environment (NumPy and h5py; no TensorFlow import).
"""
import ast
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'reproduction/runs/strict_baselines_20260905'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def reference_vote(source):
    tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='filt_voting_with_model_weight')
    namespace={'defaultdict':defaultdict}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace[node.name]


def audit_one(path):
    result=json.loads(path.read_text(encoding='utf-8'))
    out=path.parent;ds=result['dataset'];kg=result['kg']
    config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    source_hashes={name.replace('\\','/'):value for name,value in config['source_hashes'].items()}
    original='reproduction/sources/KEnS/src/ensemble.py'
    source=RUN/'code_objects'/(source_hashes[original]+'.py')
    assert sha256(source)==source_hashes[original]
    vote=reference_vote(source)
    train=np.loadtxt(ROOT/f'data/raw/dmkgc/dataset{ds}/kg/{kg}-train.tsv',delimiter='\t',dtype=np.int64,ndmin=2)
    known={}
    for h,r,t in train:known.setdefault((int(h),int(r)),set()).add(int(t))
    with np.load(out/'test_queries.npz') as q,np.load(out/'ensemble_weights.npz') as w:
        truth=q['triples'];ids=q['candidate_ids'];scores=q['candidate_scores'];langs=q['languages'].tolist()
        assert np.array_equal(w['languages'],q['languages'])
        weights=w['weights'];mismatches={n:0 for n in [1,3,10]}
        final={n:q[f'final_top{n}'] for n in mismatches}
        for i,(h,r,t) in enumerate(truth):
            choices=[[(int(e),float(s)) for e,s in zip(ee,ss) if e>=0] for ee,ss in zip(ids[i],scores[i])]
            blocked=known.get((int(h),int(r)),set())-{int(t)}
            for n in mismatches:
                ordered=vote([c[:n] for c in choices],n,blocked,weights[int(h)].tolist())
                observed=final[n][i];observed=observed[observed>=0].tolist()
                mismatches[n]+=observed!=[int(e) for e,s in ordered]
        assert not any(mismatches.values()),(path,mismatches)
        # Independently evaluate 32 fixed queries from the saved target model.
        # Direct subtraction avoids relying on the batched distance kernel.
        checkpoint=out/'model'/f'{kg}.h5'
        matrices=[]
        with h5py.File(checkpoint) as f:
            f['model_weights'].visititems(lambda name,obj:matrices.append(obj[...]) if isinstance(obj,h5py.Dataset) else None)
        nentities=weights.shape[0]
        entity=next(a for a in matrices if a.shape[0]==nentities)
        relation=next(a for a in matrices if a.shape[0]!=nentities)
        assert len(matrices)==2 and entity.shape[1]==relation.shape[1]==300
        sample=np.sort(np.random.default_rng(20260905).choice(len(truth),min(32,len(truth)),replace=False))
        target_index=langs.index(kg);top1_mismatch=0;max_score_delta=0.
        for i in sample:
            h,r,t=truth[i];dist=np.linalg.norm(entity-(entity[h]+relation[r]),axis=1)
            selected=ids[i,target_index];selected=selected[selected>=0]
            top1_mismatch+=int(np.argmin(dist))!=int(selected[0])
            max_score_delta=max(max_score_delta,float(np.max(np.abs(-dist[selected]-scores[i,target_index,:len(selected)]))))
            assert (np.diff(dist[selected])>=-2e-5).all()
            assert dist[selected[-1]]<=np.partition(dist,len(selected)-1)[len(selected)-1]+2e-5
        assert top1_mismatch==0,(path,top1_mismatch)
        assert max_score_delta<.002,(path,max_score_delta)
        observed_head=(final[1][:,0]==truth[:,0])
        self_loop=truth[:,0]==truth[:,2]
        return {'dataset':ds,'kg':kg,'seed':result['seed'],'queries':len(truth),
            'h1':float(q['hit_1'].mean()),'h3':float(q['hit_3'].mean()),'h10':float(q['hit_10'].mean()),
            'head_return_rate':float(observed_head.mean()),'self_loop_rate':float(self_loop.mean()),
            'h1_matches_self_loop_mask':bool(np.array_equal(q['hit_1'],self_loop)),
            'original_vote_mismatch_count':sum(mismatches.values()),'checkpoint_sample_queries':len(sample),
            'checkpoint_top1_mismatch_count':top1_mismatch,'max_checkpoint_score_delta':max_score_delta,
            'query_sha256':sha256(out/'test_queries.npz'),'checkpoint_sha256':sha256(checkpoint),
            'original_vote_source_sha256':sha256(source)}


def main():
    rows=[]
    for group in sorted((RUN/'jobs').glob('kens_*/result.json')):
        result=json.loads(group.read_text(encoding='utf-8'))
        if result.get('status')!='completed' or not result.get('full_data'):continue
        for kg in sorted(result['per_kg']):rows.append(audit_one(group.parent/kg/'result.json'))
    assert rows,'No full KEnS runs available'
    out=RUN/'results';out.mkdir(exist_ok=True)
    payload={'status':'passed','model_or_protocol_changed':False,
        'checks':'All saved query votes match the original voting function; fixed sampled target scores match saved H5 checkpoints.',
        'interpretation':'Low Hits@1 is retained as measured. Head-entity returns and self-loop counts are reported for diagnosis; no head exclusion or result-driven retraining is applied.',
        'rows':rows}
    (out/'kens_prediction_audit.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    with (out/'kens_prediction_audit.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# KEnS 预测复核','',
        '全部已完成的正式运行逐查询重放了作者投票函数，并从 H5 检查点直接重算每域固定抽取的 32 个查询；候选第一名和投票结果均一致。',
        '本次 TransE 配方出现频繁返回查询头实体的现象。低 Hits@1 保留为实测值，不删除头实体或据此更换配方。Hits@1 的标准差为零也保留原值。','',
        '| 数据集 | KG | 种子 | Hits@1 (%) | Hits@10 (%) | 返回头实体 (%) | 自环查询 (%) |',
        '|---|---|---|---|---|---|---|']
    for r in rows:lines.append(f"| {r['dataset']} | {r['kg']} | {r['seed']} | {r['h1']*100:.4f} | {r['h10']*100:.4f} | {r['head_return_rate']*100:.4f} | {r['self_loop_rate']*100:.4f} |")
    lines+=['','字段、检查点和逐查询文件哈希见 `kens_prediction_audit.json`；可按相同测试行序继续开展案例与失效模式分析。']
    (out/'KENS_PREDICTION_AUDIT.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','audited_kg_runs':len(rows),'audited_queries':sum(r['queries'] for r in rows),
        'all_head_returns':all(r['head_return_rate']==1 for r in rows),
        'all_h1_equal_self_loops':all(r['h1_matches_self_loop_mask'] for r in rows)}))


if __name__=='__main__':main()
