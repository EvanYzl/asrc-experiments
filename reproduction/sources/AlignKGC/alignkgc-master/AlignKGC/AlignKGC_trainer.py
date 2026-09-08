import os
import random
from typing import Dict
import torch, torch.nn, torch.nn.utils
from torch.autograd import Variable

from AlignKGC import kb, evaluate
from AlignKGC.utils import update_imp_sc, get_rel_align_dict
from AlignKGC.alignkgc_base import AlignKgcBaseTrainer


class AlignKGC(AlignKgcBaseTrainer):
    def __init__(self, **kwargs):
        super(AlignKGC, self).__init__(**kwargs)
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
        # Keep the expensive O(R^2) soft relation-alignment refresh bounded in
        # explicit smoke tests.  Five relations per language still exercises
        # every cross-language branch; full runs (MAX_SAM=0) are untouched.
        max_samples = kwargs.get("MAX_SAM", 0)
        if max_samples and max_samples > 0:
            per_lang_limit = max(1, min(5, max_samples // len(self.meta.langs)))
            pair_limit = max(2, min(8, max_samples))
            kept_per_lang = {str(lid): 0 for lid in self.meta.lids()}
            limited_pairs = {}
            limited_lang = {}
            for rel, pairs in self.rel_ent_pairs.items():
                lang = self.yylang[rel]
                if (lang in kept_per_lang and len(pairs) >= 2 and
                        kept_per_lang[lang] < per_lang_limit):
                    limited_pairs[rel] = pairs[:pair_limit]
                    limited_lang[rel] = lang
                    kept_per_lang[lang] += 1
            self.rel_ent_pairs = limited_pairs
            self.yylang = limited_lang
            print("MAX_SAM relation-alignment cap", kept_per_lang,
                  "relations", len(self.rel_ent_pairs))

    def rel_alignment_loss(self, losstype="L1", entity_bactrack=0):
        R_re = self.scoring_function.R_re
        R_im = self.scoring_function.R_im
        rel1s, rel2s, scores = [], [], []
        for lid in self.meta.lids():
            lidprefix = str(lid)
            for rel in self.equiv_rel[lidprefix]:
                sc,rel2 = self.equiv_rel[lidprefix][rel]
                if entity_bactrack==0:
                    sc=sc.detach()
                rel1s.append(rel)
                rel2s.append(rel2)
                scores.append(sc.reshape(()))
        device = R_re.weight.device
        if not rel1s:
            return torch.zeros((), device=device)
        p1 = torch.as_tensor(rel1s, dtype=torch.long, device=device)
        p2 = torch.as_tensor(rel2s, dtype=torch.long, device=device)
        sc = torch.stack(scores).reshape(-1)
        if losstype == "L1":
            scaled_re = sc[:, None] * (R_re(p1) - R_re(p2))
            scaled_im = sc[:, None] * (R_im(p1) - R_im(p2))
            return (scaled_re.abs().mean(dim=1) +
                    scaled_im.abs().mean(dim=1)).sum()
        if losstype == "cos":
            cos_re = torch.nn.functional.cosine_similarity(
                R_re(p1), R_re(p2), dim=1, eps=1e-6)
            cos_im = torch.nn.functional.cosine_similarity(
                R_im(p1), R_im(p2), dim=1, eps=1e-6)
            return (sc * (1 - cos_re) + sc * (1 - cos_im)).sum()
        raise ValueError(losstype)

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
        # ``fp`` is needed only by sampled-negative losses.  ``fnr`` was never
        # consumed by any loss in the released trainer.  Avoiding those two
        # forwards leaves the objective unchanged and removes an unnecessary
        # all-relation matrix multiplication in the full-softmax recipe.
        fp = None
        direct_subsample_loss = None
        can_score_subsampled = (
            flag_using_full_softmax and
            self.loss.name == 'crossentropy_loss_AllNeg_subsample' and
            hasattr(self.scoring_function, 'score_subsampled_entities') and
            not self.scoring_function.batch_norm and
            not self.scoring_function.clamp_v)
        if can_score_subsampled:
            # The released loss calls randperm once for head prediction and
            # then once for tail prediction.  No intervening model operation
            # consumes RNG in this configuration, so these are the identical
            # candidate sets without first materialising every entity score.
            entity_count = self.scoring_function.entity_count
            head_candidates = torch.randperm(entity_count)[
                :self.negative_sample_count].to(s.device)
            tail_candidates = torch.randperm(entity_count)[
                :self.negative_sample_count].to(s.device)
            head_pos, head_neg, tail_pos, tail_neg = \
                self.scoring_function.score_subsampled_entities(
                    s, r, o, head_candidates, tail_candidates)
            labels = torch.zeros(s.shape[0], dtype=torch.long, device=s.device)
            direct_subsample_loss = (
                self.loss.loss(torch.cat((head_pos[:, None], head_neg), dim=1),
                               labels)
                + self.loss.loss(torch.cat((tail_pos[:, None], tail_neg), dim=1),
                                 labels))
            fno = fns = None
        elif not flag_using_full_softmax:
            fp = self.scoring_function(s, r, o, flag_debug=flag_debug+1)
            fno = self.scoring_function(s, r, no, flag_debug=flag_debug+1)
            fns = self.scoring_function(ns, r, o, flag_debug=flag_debug+1)
        else:
            fno = self.scoring_function(s, r, no, flag_debug=flag_debug+1)
            fns = self.scoring_function(ns, r, o, flag_debug=flag_debug+1)

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
            if direct_subsample_loss is not None:
                kbc_loss = direct_subsample_loss
            elif self.loss.name == 'crossentropy_loss_AllNeg_subsample':
                kbc_loss = self.loss(s, fns, self.negative_sample_count) + \
                           self.loss(o, fno, self.negative_sample_count)
            else:
                if self.flag_add_reverse==0 or self.scoring_function.flag_avg_scores:
                    kbc_loss = self.loss(s, fns) + self.loss(o, fno)
                else:
                    kbc_loss = self.loss(o, fno)
        else:
            kbc_loss = self.loss(fp, fns) + self.loss(fp, fno)

        device = self.scoring_function.E_re.weight.device
        ea_loss = torch.zeros(1, device=device)  # pinned to zero
        ra_loss = torch.zeros(1, device=device)
        if loop_no%10==0:  # MAGIC
            if loop_no%5000==0 or (loop_no>13000 and loop_no%1000==0):
                self.equiv_rel = update_imp_sc(self.meta, self.rel_ent_pairs,
                                               self.yylang,
                                               self.scoring_function.E_re,
                                               self.scoring_function.E_im,
                                               self.a, self.b,
                                               # The released schedule reuses the
                                               # mapping built at batch 10000 for
                                               # the first ``b`` update at 13050.
                                               # Retain that selected-pair graph;
                                               # starting at 13000 is too late
                                               # because no mapping is built there.
                                               track_grad=loop_no >= 10000)
            if loop_no>13000 and loop_no%50==0:  # MAGIC
                    f=0.01* self.rel_alignment_loss(losstype="cos",entity_bactrack=1)
                    f.backward(retain_graph=True)
                    # print("b",self.b,self.b.grad)
                    self.b.data=self.b-0.8*self.b.grad.data  # MAGIC
                    self.b.grad.data.zero_()
                    # print("b.grad", self.b.grad)
            else :
                ra_loss=self.raloss_coeff* self.rel_alignment_loss(losstype="L1")
                # print("e=", ra_loss, "RA=", self.raloss_coeff)
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
        # One host transfer instead of five synchronizing ``.item()`` calls.
        values = torch.stack((total_loss.reshape(()), kbc_loss.reshape(()),
                              ea_loss.reshape(()), ra_loss.reshape(()),
                              reg_loss.reshape(()))).detach().cpu().tolist()
        return dict(zip(("total", "kbc", "ea", "ra", "reg"), values))
