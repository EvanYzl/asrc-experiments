import numpy
import torch
import torch.autograd
from AlignKGC.kb import kb

class data_loader(object):
    """
    Does th job of batching a knowledge base and also generates negative samples with it.
    """
    def __init__(self, kb_, load_to_gpu, first_zero=True, loss=None, flag_add_reverse=None):
        """
        Duh..\n
        :param kb: the knowledge base to batch
        :param load_to_gpu: Whether the batch should be loaded to the gpu or not
        :param first_zero: Whether the first entity in the set of negative samples of each fact should be zero
        """
        self.kb : kb = kb_
        self.load_to_gpu = load_to_gpu
        self.first_zero = first_zero
        self.flag_add_reverse = flag_add_reverse
        self.loss = loss

    def sample(self, batch_size=1000, negative_count=10):
        """
        Generates a random sample from kb and returns them as numpy arrays.\n
        :param batch_size: The number of facts in the batch or the size of batch.
        :param negative_count: The number of negative samples for each positive fact.
        :return: A list containing s, r, o and negative s and negative o of the batch
        """
        indexes = numpy.random.randint(0, self.kb.facts.shape[0], batch_size)
        facts = self.kb.facts[indexes]
        s = numpy.expand_dims(facts[:, 0], -1)
        r = numpy.expand_dims(facts[:, 1], -1)
        o = numpy.expand_dims(facts[:, 2], -1)
        ns = numpy.random.randint(0, len(self.kb.entity_map), (batch_size, negative_count))#2))
        no = numpy.random.randint(0, len(self.kb.entity_map), (batch_size, negative_count))
        if self.first_zero:
            ns[:, 0] = len(self.kb.entity_map)-1
            no[:, 0] = len(self.kb.entity_map)-1
        return [s, r, o, ns, no]

    def sample_icml(self, batch_size=1000, negative_count=10):
        """
        Generates a random sample from kb and returns them as numpy arrays.\n
        :param batch_size: The number of facts in the batch or the size of batch.
        :param negative_count: The number of negative samples for each positive fact.
        :return: A list containing s, r, o and negative s and negative o of the batch
        """
        indexes = numpy.random.randint(0, self.kb.facts.shape[0], batch_size)
        facts = self.kb.facts[indexes]
        s_fwd = numpy.expand_dims(facts[:, 0], -1)
        r_fwd = numpy.expand_dims(facts[:, 1], -1)
        o_fwd = numpy.expand_dims(facts[:, 2], -1)

        if self.loss == "crossentropy_loss":
            ns_fwd = None; no_fwd = None
        else:
            ns_fwd = numpy.random.randint(0, self.kb.nonoov_entity_count, (batch_size, negative_count))
            no_fwd = numpy.random.randint(0, self.kb.nonoov_entity_count, (batch_size, negative_count))

        num_relations = len(self.kb.relation_map)
        r_rev = r_fwd + num_relations

        s = numpy.concatenate([s_fwd, o_fwd])
        r = numpy.concatenate([r_fwd, r_rev])
        o = numpy.concatenate([o_fwd, s_fwd])

        if self.loss == "crossentropy_loss":
            ns = None; no = None
        else:
            ns = numpy.concatenate([ns_fwd, no_fwd]) ##to do randomly generate ns_rev and no_rev
            no = numpy.concatenate([no_fwd, ns_fwd])
            if self.first_zero:
                ns[:, 0] = self.kb.nonoov_entity_count - 1
                no[:, 0] = self.kb.nonoov_entity_count - 1

        return [s, r, o, ns, no] 

    def tensor_sample(self, batch_size=1000, negative_count=10):
        """
        Generates a random sample from kb and returns them as torch tensors. Internally uses sampe\n
        :param batch_size: The number of facts in the batch or the size of batch.
        :param negative_count: The number of negative samples for each positive fact.
        :return: A list containing s, r, o and negative s and negative o of the batch
        """
        if self.flag_add_reverse:
            ls = self.sample_icml(batch_size, negative_count)
        else:
            ls = self.sample(batch_size, negative_count)
        if self.load_to_gpu:
            return [torch.autograd.Variable(torch.from_numpy(x).cuda()) for x in ls]
        else:
            return [torch.autograd.Variable(torch.from_numpy(x)) for x in ls]
