# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, inputs):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                (head, relation, tail) = [ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(x), ag__.ld(tf).int64), None, fscope) for x in ag__.ld(inputs)]
                shape = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(head),), None, fscope)[0], ag__.ld(self).num_negatives],), None, fscope)

                @ag__.autograph_artifact
                def draw():
                    with ag__.FunctionScope('draw', 'fscope_1', ag__.STD) as fscope_1:
                        do_return_1 = False
                        retval__1 = ag__.UndefinedReturnValue()
                        try:
                            do_return_1 = True
                            retval__1 = ag__.converted_call(ag__.ld(tf).random.uniform, (ag__.ld(shape),), dict(maxval=ag__.ld(self).num_entities, dtype=ag__.ld(tf).int64), fscope_1)
                        except:
                            do_return_1 = False
                            raise
                        return fscope_1.ret(retval__1, do_return_1)

                @ag__.autograph_artifact
                def invalid(candidate):
                    with ag__.FunctionScope('invalid', 'fscope_2', ag__.STD) as fscope_2:
                        do_return_2 = False
                        retval__2 = ag__.UndefinedReturnValue()
                        key = (ag__.ld(head) * ag__.ld(self).num_relations + ag__.ld(relation)) * ag__.ld(self).num_entities + ag__.ld(candidate)
                        try:
                            do_return_2 = True
                            retval__2 = ag__.converted_call(ag__.ld(tf).logical_or, (ag__.converted_call(ag__.ld(self).known.lookup, (ag__.ld(key),), None, fscope_2) > 0, ag__.ld(candidate) == ag__.ld(tail)), None, fscope_2)
                        except:
                            do_return_2 = False
                            raise
                        return fscope_2.ret(retval__2, do_return_2)
                candidate = ag__.converted_call(ag__.ld(draw), (), None, fscope)
                (candidate,) = ag__.converted_call(ag__.ld(tf).while_loop, (ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.ld(tf).reduce_any, (ag__.converted_call(ag__.ld(invalid), (ag__.ld(x),), None, fscope),), None, fscope)), ag__.autograph_artifact(lambda x: (ag__.converted_call(ag__.ld(tf).where, (ag__.converted_call(ag__.ld(invalid), (ag__.ld(x),), None, fscope), ag__.converted_call(ag__.ld(draw), (), None, fscope), ag__.ld(x)), None, fscope),)), (ag__.ld(candidate),)), dict(maximum_iterations=10000), fscope)
                ag__.converted_call(ag__.ld(tf).debugging.assert_equal, (ag__.converted_call(ag__.ld(tf).reduce_any, (ag__.converted_call(ag__.ld(invalid), (ag__.ld(candidate),), None, fscope),), None, fscope), False), dict(message='No valid corrupted tail sampled'), fscope)
                try:
                    do_return = True
                    retval_ = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(candidate), ag__.ld(tf).int32), None, fscope)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory