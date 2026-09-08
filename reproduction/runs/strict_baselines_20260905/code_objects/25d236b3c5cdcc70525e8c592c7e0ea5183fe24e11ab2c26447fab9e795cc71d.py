"""Numerical and serialization checks for the KEnS training correction."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'reproduction/sources/KEnS'))
import numpy as np
import tensorflow as tf
from tensorflow import keras
import src.param as param
from src.model import create_knowledge_model,create_alignment_model,get_entity_layer,get_relation_layer
from src.negative_sampling import FilteredTailSampler,positive_keys
from src.validate import MultiModelTester,TestMode
from src.model import extend_seed_align_links
from src.weightlearning import rank_matrix_from_one_triple,weight_matrix_from_one_triple,learn_model_weights
import pandas as pd

RUN=ROOT/'reproduction/runs/strict_baselines_20260905'


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    param.knowledge='transe';param.dim=4;param.k=3;param.neg_per_pos=2;param.reg_scale=0;param.margin=.3
    np.random.seed(17);tf.random.set_seed(17)
    facts=np.array([[0,0,2],[0,0,3],[1,1,4],[2,0,6]],dtype=np.int64)
    x=[facts[:3,i:i+1].astype(np.int32) for i in range(3)]
    config=json.loads((RUN/'jobs/kens_dbp5l_s17/el/config.json').read_text())
    original_hash=next(v for k,v in config['source_hashes'].items() if k.replace('\\','/').endswith('/src/model.py'))
    spec=importlib.util.spec_from_file_location('kens_legacy_fixture',RUN/'code_objects'/(original_hash+'.py'))
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    legacy,_,_=original.create_knowledge_model(8,2)
    legacy_shape=list(legacy(x).shape);assert legacy_shape==[3,3]
    model,predictor,knn=create_knowledge_model(8,2,training_triples=facts)
    sampler=model.get_layer('generate_negative_t_samples')
    negative=model.get_layer('compute_loss').get_output_at(1)
    inspect_model=keras.Model(model.inputs,[model.output,sampler.output,negative])
    with tf.GradientTape() as tape:
        loss,neg,neg_dist=inspect_model(x)
        neg_objective=tf.reduce_mean(neg_dist)
    gradient=tf.convert_to_tensor(tape.gradient(neg_objective,get_relation_layer(model).weights[0])).numpy()
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient)>0
    entity=get_entity_layer(model).get_weights()[0];relation=get_relation_layer(model).get_weights()[0]
    q=entity[x[0].ravel()]+relation[x[1].ravel()]
    pos=np.linalg.norm(entity[x[2].ravel()]-q+1e-8,axis=1,keepdims=True)
    distance=np.linalg.norm(q[:,None,:]-entity[neg.numpy()]+1e-8,axis=-1).mean(1,keepdims=True)
    expected=np.maximum(pos-distance+param.margin,0)
    assert list(loss.shape)==[3,1]
    np.testing.assert_allclose(loss.numpy(),expected,rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(predictor(x[:2]).numpy()[:,0,:],q,rtol=0,atol=0)
    queries=np.repeat(facts[:1],4096,axis=0)
    drawn=sampler([queries[:,i:i+1] for i in range(3)]).numpy()
    assert set(np.unique(drawn))=={0,1,4,5,6,7},np.unique(drawn)
    assert not np.isin(drawn,[2,3]).any()
    with tempfile.TemporaryDirectory(prefix='kens-repair-',dir=str(RUN/'smoke')) as directory:
        file=Path(directory)/'roundtrip.h5';model.save(file)
        restored=keras.models.load_model(file,compile=False,custom_objects={'tf':tf,'param':param})
        np.testing.assert_array_equal(get_entity_layer(restored).get_weights()[0],entity)
        np.testing.assert_array_equal(get_relation_layer(restored).get_weights()[0],relation)
        assert list(restored(x).shape)==[3,1]
        another,_,_=create_knowledge_model(9,2,training_triples=facts)
        alignment=create_alignment_model(model,another)
        assert np.isfinite(alignment([x[0],x[0]]).numpy()).all()
    target=SimpleNamespace(lang='target',num_entity=8,model=model,kNN_finder=knn)
    support=SimpleNamespace(lang='support',num_entity=8,model=model,dict0to1={0:0,5:4})
    tester=MultiModelTester(target,[support])
    transferred=tester.predict(0,0,TestMode.Transfer,supporter_kg=support)
    assert {entity_id for entity_id,score in transferred}=={0,5}
    assert tester.predict(1,0,TestMode.Transfer,supporter_kg=support)==[]
    param.n_test=3
    prediction_frame=tester._MultiModelTester__test_and_record_results_transe_batched(facts[:3])
    assert all(e in {0,5} for pairs in prediction_frame['support'] for e,score in pairs)
    row=pd.Series({'t':2,'known_true_tails':[2,3],'a':[0,2,3],'b':[5,4]})
    rank=rank_matrix_from_one_triple('a',row,8,['a','b'],{'a':.5,'b':.5})
    weight=weight_matrix_from_one_triple(row,8)
    assert np.array_equal(rank[[2,3]],np.zeros((2,2))) and np.array_equal(weight[[2,3]],[0,0])
    assert rank[0,0]==-1 and rank[4,0]==1
    rng=np.random.default_rng(81);ranks=np.where(rng.random((100,6))<.65,1,-1)
    boosted=learn_model_weights(list('abcdef'),np.ones(100)/100,ranks)
    assert all(np.isfinite(v) and abs(v-1)>1e-9 for v in boosted.values())
    param.max_entity_scan=100
    source_vectors=np.eye(4,dtype=np.float32)
    target_vectors=(np.eye(4,dtype=np.float32)+.01).astype(np.float32)
    a=SimpleNamespace(num_entity=4,get_embedding_matrix=lambda:source_vectors.reshape(1,-1))
    b=SimpleNamespace(num_entity=4,get_embedding_matrix=lambda:target_vectors.reshape(1,-1))
    links=extend_seed_align_links(a,b,np.empty((0,2),dtype=np.int64))
    assert set(map(tuple,links))=={(i,i) for i in range(4)},links
    report={'passed':True,'legacy_loss_shape':legacy_shape,'corrected_loss_shape':list(loss.shape),
        'analytical_loss_max_abs_error':float(np.max(np.abs(loss.numpy()-expected))),
        'negative_loss_relation_gradient_norm':float(np.linalg.norm(gradient)),
        'negative_samples_checked':int(drawn.size),'last_entity_id_sampled':bool((drawn==7).any()),
        'known_positive_negatives':int(np.isin(drawn,[2,3]).sum()),
        'h5_roundtrip_and_alignment_shared_weights':'passed','legacy_model_source_sha256':original_hash,
        'unaligned_candidates_excluded':'passed','critical_true_true_pairs_excluded':'passed',
        'six_model_boosting_rounds':'passed','csls_mutual_best_neighbor_fixture':'passed',
        'paper':'https://aclanthology.org/2020.findings-emnlp.290.pdf','equations':[1,2]}
    (RUN/'results/kens_training_repair_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
