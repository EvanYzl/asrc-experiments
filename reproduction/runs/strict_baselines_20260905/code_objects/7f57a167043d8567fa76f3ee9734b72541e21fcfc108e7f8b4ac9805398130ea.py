import os
import random
import torch, torch.nn, torch.nn.utils
from torch.autograd import Variable
from typing import Dict, List, Tuple

from AlignKGC import kb, evaluate
from AlignKGC.utils import update_imp_sc, get_rel_align_dict
from AlignKGC.alignkgc_base import AlignKgcBaseTrainer


class AlignKGCmBERT(AlignKgcBaseTrainer):
    def __init__(self, **kwargs):
        super(AlignKGCmBERT, self).__init__(**kwargs)
        self.a = 100  # MAGIC
        self.b = Variable(torch.tensor(90).cuda().float(), requires_grad=True)
        self.ranker = evaluate.ranker(self.scoring_function,
                        kb.union([self.dltrain.kb, self.dlvalid.kb]+
                                [i[1].kb for i in self.dltestmap.values()]))
        #ugh i[0]=lang, i[1]=kb
        trainpath = os.path.join(self.dataset_root, 'train.txt')
        self.rel_ent_pairs, self.yylang = get_rel_align_dict(self.meta, trainpath,
                                emap=self.dltrain.kb.entity_map,
                                rmap=self.dltrain.kb.relation_map)

        self.bert_align = self.get_bert_entity_alignments(entmap=self.dltrain.kb.entity_map)

    def rel_alignment_loss(self, losstype="L1", entity_bactrack=0):
        R_re = self.scoring_function.R_re
        R_im = self.scoring_function.R_im
        loss : float=0
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for lid in self.meta.lids():
            lidprefix = str(lid)
            for rel in self.equiv_rel[lidprefix]:
                sc,rel2 = self.equiv_rel[lidprefix][rel]
                if entity_bactrack==0:
                    sc=sc.detach()
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

        ea_loss=torch.zeros(1).cuda()
        ra_loss=torch.zeros(1).cuda()
        if loop_no%10==0:  # MAGIC
            if loop_no%5000==0 or (loop_no>13000 and loop_no%1000==0):
                self.equiv_rel = update_imp_sc(self.meta, self.rel_ent_pairs,
                                               self.yylang,
                                               self.scoring_function.E_re,
                                               self.scoring_function.E_im,
                                               self.a, self.b)
            if loop_no>13000 and loop_no%50==0:  # MAGIC
                    f=0.01* self.rel_alignment_loss(losstype="cos",entity_bactrack=1)
                    f.backward(retain_graph=True)
                    # print("b",self.b,self.b.grad)
                    self.b.data=self.b-0.8*self.b.grad.data  # MAGIC
                    self.b.grad.data.zero_()
                    # print("b.grad", self.b.grad)
            else :
                ra_loss=self.raloss_coeff* self.rel_alignment_loss(losstype="L1")
            if loop_no%50==0:
                ea_loss= self.ealoss_coeff * self.ent_alignment_loss(self.bert_align, self.scoring_function.E_re, self.scoring_function.E_im)
        total_loss = reg_loss + kbc_loss + ra_loss + ea_loss
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
