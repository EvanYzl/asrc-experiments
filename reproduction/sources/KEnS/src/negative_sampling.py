"""Uniform per-query corrupted tails, excluding known training positives."""
import numpy as np
import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package='KEnS')
class FilteredTailSampler(keras.layers.Layer):
    def __init__(self,num_entities,num_relations,positive_keys,num_negatives=1,**kwargs):
        super().__init__(**kwargs)
        self.num_entities=int(num_entities);self.num_relations=int(num_relations)
        self.num_negatives=int(num_negatives);self.positive_keys=[int(x) for x in positive_keys]
        keys=tf.constant(self.positive_keys,dtype=tf.int64)
        self.known=tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys,tf.ones_like(keys)),default_value=0)

    def call(self,inputs):
        head,relation,tail=[tf.cast(x,tf.int64) for x in inputs]
        shape=tf.stack([tf.shape(head)[0],self.num_negatives])
        def draw():return tf.random.uniform(shape,maxval=self.num_entities,dtype=tf.int64)
        def invalid(candidate):
            key=(head*self.num_relations+relation)*self.num_entities+candidate
            return tf.logical_or(self.known.lookup(key)>0,candidate==tail)
        candidate=draw()
        candidate,=tf.while_loop(lambda x:tf.reduce_any(invalid(x)),
            lambda x:(tf.where(invalid(x),draw(),x),),(candidate,),maximum_iterations=10000)
        tf.debugging.assert_equal(tf.reduce_any(invalid(candidate)),False,message='No valid corrupted tail sampled')
        return tf.cast(candidate,tf.int32)

    def get_config(self):
        return {**super().get_config(),'num_entities':self.num_entities,'num_relations':self.num_relations,
                'num_negatives':self.num_negatives,'positive_keys':self.positive_keys}


def positive_keys(triples,num_entities,num_relations):
    triples=np.unique(np.asarray(triples,dtype=np.int64).reshape(-1,3),axis=0)
    hr=triples[:,0]*num_relations+triples[:,1]
    _,counts=np.unique(hr,return_counts=True)
    if len(counts) and counts.max()>=num_entities:
        raise ValueError('A training query has no negative tail candidate')
    return (hr*num_entities+triples[:,2]).tolist()
