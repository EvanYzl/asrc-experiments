# coding=utf-8
from __future__ import division
def outer_factory():

    def inner_factory(ag__):
        tf__lam = lambda y_true, loss: ag__.with_function_scope(lambda lscope: loss, 'lscope', ag__.STD)
        return tf__lam
    return inner_factory