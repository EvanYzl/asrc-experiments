import numpy
import os
import random
import torch, torch.nn, torch.nn.utils
from typing import Dict

from AlignKGC import kb, evaluate
from AlignKGC.utils import intersect, union
from AlignKGC.alignkgc_base import AlignKgcBaseTrainer


class Jaccard(AlignKgcBaseTrainer):
    def __init__(self, **kwargs):
        super(Jaccard, self).__init__(**kwargs)
        self.ranker = evaluate.ranker(self.scoring_function,
                                      kb.union([self.dltrain.kb, self.dlvalid.kb]+
                                               [i[1].kb for i in self.dltestmap.values()]))
        #ugh i[0]=lang, i[1]=kb
        trainpath = os.path.join(self.dataset_root, 'train.txt')
        self.rel_aligns, _ = self.get_rel_align_imply(self.meta, trainpath,
                                    rmap=self.dltrain.kb.relation_map,
                                    jaccard=True)

    def get_rel_align_imply(self, meta, kgc_train_path, rmap, jaccard: bool):
        """
        :param jaccard: if true, use jaccard, otherwise use asymmetric
        similarity
        """

        rel_ent_pairs = dict()
        """ map from rel to SO-pairs(rel)
        key = grelid, val = listof( tupleof(gsubid, gobjid) ) """
        kgc_train_file=open(kgc_train_path)  # combined global train triples
        lines = kgc_train_file.readlines()
        for line in lines:
            a=line.split()
            if int(a[1]) not in rel_ent_pairs:
                rel_ent_pairs[int(a[1])] = list()
            rel_ent_pairs[int(a[1])].append((int(a[0]),int(a[2])))
        kgc_train_file.close()

        IoU_rel1 = dict()
        """ key = rel1,  val = listof( tuple4( iou, io1, io2, rel2 ) )
        rel1 and rel2 are in different languages.
        The val lists are sorted in decreasing order.
        iou = jaccard, io1 = intersect / set1, io2 = intersect / set2 """

        equiv_rel = dict()
        """ key = string langid,
        val = dictof( rel1 -> closest rel2 in langid ) """

        for lang in meta.langs:  # for each lang
            langid = str(meta.lang_to_lid(lang))
            equiv_rel[langid] = dict()
            for rel1 in rel_ent_pairs:  # rel1 isa grelid str
                if str(rel1)[0]!= langid:
                    continue
                # rel1 is of language lang
                IoU_rel1[rel1] = list()
                for rel2 in rel_ent_pairs:  # rel2 isa grelid str
                    if str(rel2)[0]!= langid:  # only if rel2 is not of lang
                        sc=len(intersect(rel_ent_pairs[rel1],rel_ent_pairs[rel2]))/len(union(rel_ent_pairs[rel1],rel_ent_pairs[rel2]))
                        sc1=len(intersect(rel_ent_pairs[rel1],rel_ent_pairs[rel2]))/len(rel_ent_pairs[rel1])
                        sc2=len(intersect(rel_ent_pairs[rel1],rel_ent_pairs[rel2]))/len(rel_ent_pairs[rel2])
                        # sc = jaccard, sc1, sc2 = asymm jaccard
                        if jaccard:
                            IoU_rel1[rel1].append((sc,sc,sc,rel2))
                        else:
                            IoU_rel1[rel1].append((sc1,sc2,sc,rel2))
                IoU_rel1[rel1].sort(reverse=True)  # most similar rel2 first

        for rel1 in IoU_rel1:  # for each rel1...
            for lang in meta.langs:
                langid = str(meta.lang_to_lid(lang))
                for sc1,sc2,sc,rel2 in IoU_rel1[rel1]:
                    if str(rel2)[0]== langid:
                        equiv_rel[langid][rel1] = (sc,sc1,sc2,rel2)
                        break  # only most similar rel2 in lang

        rel_align = dict()
        """ key = rel1 val = list( tuple2 ( iou, rel2 ) )
        for each lang, keep largest iou rel2 in list """
        implication = dict()

        for rel1 in IoU_rel1:
            rel_align[rmap[str(rel1)]]=[]
            implication[rmap[str(rel1)]]=[]
            for lang in meta.langs:
                langid = str(meta.lang_to_lid(lang))
                if langid==str(rel1)[0]:
                    continue  # only languages other than of rel1
                sc,sc1,sc2,rel2 = equiv_rel[langid][rel1]
                rel_align[rmap[str(rel1)]].append((sc, rmap[str(rel2)]))
                if sc1>=0.1:  ## MAGIC
                    implication[rmap[str(rel1)]].append((sc, rmap[str(rel2)]))
                    # curiously, (sc, rel2) not (sc1, rel2)

        return rel_align, implication

    def rel_alignment_loss(self, losstype="L1"):
        R_re = self.scoring_function.R_re
        R_im = self.scoring_function.R_im
        loss : float=0
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for rel in self.rel_aligns:
            for sc,rel2 in self.rel_aligns[rel]:
                if sc <= 0.01:  ## MAGIC
                    continue
                p1=torch.tensor([rel]).cuda()
                p2=torch.tensor([rel2]).cuda()
                if losstype=="L1":
                    loss=loss+((sc*R_re(p1)-sc*R_re(p2)).abs().mean()+
                               (sc*R_im(p1)-sc*R_im(p2)).abs().mean())
                elif losstype == "cos":
                    loss=loss+(sc*(1-cos(R_re(p1),R_re(p2)))+
                               sc*(1-cos(R_im(p1),R_im(p2))))
                else:
                    raise ValueError(losstype)
        return loss

    def step(self, loop_no) -> Dict[str, float]:
        """
        Use all ent as neg sample
        """
        flag_using_full_softmax = 0

        if self.negative_sample_count == 0 or self.loss.name == 'crossentropy_loss_AllNeg_subsample':  # use all ent as neg sample
            ns = None
            no = None
            nr = None
            s, r, o, _, _ = self.dltrain.tensor_sample(self.batch_size, 1)
            flag_using_full_softmax = 1
        else:
            s, r, o, ns, no = self.dltrain.tensor_sample(self.batch_size, self.negative_sample_count)

        flag = random.randint(1,10001)
        if flag>9950:
            flag_debug = 1
        else:
            flag_debug = 0
        fp  = self.scoring_function(s, r, o,  flag_debug=flag_debug+1)
        fno = self.scoring_function(s, r, no, flag_debug=flag_debug+1)
        fns = self.scoring_function(ns, r, o, flag_debug=flag_debug+1)
        fnr = self.scoring_function(s, nr, o, flag_debug=flag_debug+1)

        if self.regloss_coeff > 0:
            reg = self.regularizer(s, r, o)#, reg_val=3) #+ self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            if self.scoring_function.reg != 3:
                reg = reg/self.batch_size#/(self.batch_size*self.scoring_function.embedding_dim)
                ##dividing by dim size is a bit too much!!
        else:
            reg = 0  # TODO should be cast to tensor?
        reg_loss = self.regloss_coeff*reg

        kbc_loss = None
        if flag_using_full_softmax:  # use all ent as neg sample
            if self.loss.name == 'crossentropy_loss_AllNeg_subsample':
                kbc_loss = self.loss(s, fns, self.negative_sample_count) + \
                           self.loss(o, fno, self.negative_sample_count)
            else:
                if self.flag_add_reverse==0 or self.scoring_function.flag_avg_scores:
                    kbc_loss = self.loss(s, fns) + self.loss(o, fno)
                else:
                    kbc_loss = self.loss(o, fno)
        else:
            kbc_loss = self.loss(fp, fns) + self.loss(fp, fno)

        ea_loss = torch.zeros(1).cuda()  # pinned to zero
        ra_loss = torch.zeros(1).cuda()
        if numpy.random.randint(1000)<100:  ## MAGIC
            ra_loss = self.raloss_coeff * self.rel_alignment_loss(losstype="L1")
        total_loss = reg_loss + kbc_loss + ea_loss + ra_loss
        self.optim.zero_grad()
        total_loss.backward()
        if self.gradient_clip:
            torch.nn.utils.clip_grad_norm(self.scoring_function.parameters(),
                                          self.gradient_clip)
        self.optim.step()
        debug = ""
        if "post_epoch" in dir(self.scoring_function):
            debug = self.scoring_function.post_epoch()
        return { "total": total_loss.item(),
            "kbc": kbc_loss.item(), "ea": ea_loss.item(), "ra": ra_loss.item(),
            "reg": reg_loss.item()}

