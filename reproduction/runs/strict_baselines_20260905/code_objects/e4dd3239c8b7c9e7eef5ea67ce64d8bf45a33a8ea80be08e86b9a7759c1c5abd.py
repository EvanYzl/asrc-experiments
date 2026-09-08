import csv
import copy
import gc
import numpy
import time
import torch

from collections import defaultdict as dd

from torch.functional import Tensor

from AlignKGC import utils
from AlignKGC.kb import kb

class ranker(object):
    """
    A network that ranks entities based on a scoring function. It excludes entities that have already
    been seen in any kb to compute the ranking as in ####### cite paper here ########. It can be constructed
    from any scoring function/model from models.py
    """
    def __init__(self, scoring_function, all_kb, type_filter=None):
        """
        The initializer\n
        :param scoring_function: The model function to used to rank the entities
        :param all_kb: A union of all the knowledge bases (train/test/valid)
        :param type_filter: Passed when type filtering of entities needed during prediction
                            Should be a dictionary of the form - {'filename':<filename>,'mode':<mode>}
                            :<filename> is a string path of csv containing an entity to type mapping ('<id> <ent_name> <type_name>' on each line)
                            :<mode> can be either 'type' (restrict entity set to be of same type as gold entity), 
                             or 'typeset'(restrict entities to be of set of types corresponding to relation)
        """
        super(ranker, self).__init__()
        self.scoring_function = scoring_function
        self.all_kb = all_kb
        self.type_filter=False
        try:
            self.flag_add_reverse = scoring_function.flag_add_reverse
            self.flag_avg_scores  = scoring_function.flag_avg_scores
            print("##self.flag_avg_score", self.flag_avg_scores)
            print("##self.flag_add_reverse", self.flag_add_reverse)
        except:
            self.flag_add_reverse = 0


        print("Add reverse", self.flag_add_reverse)
        if type_filter:      #filter candidate entities based on types
            self.type_filter=True
            
            filename,mode=type_filter['filename'],type_filter['mode']
            self.type_filter_mode=mode
            
            self.id_typeint=self.get_typemap(filename)

            if(mode=='typeset'):
                #--for typeset--#
                self.rel_to_otypeset=dd(lambda: [])
                self.rel_to_stypeset=dd(lambda: [])
                for e1,r,e2 in all_kb.facts:
                    self.rel_to_otypeset[r].append(self.id_typeint[e2])
                    self.rel_to_stypeset[r].append(self.id_typeint[e1])

                for r in self.rel_to_stypeset.keys():
                    self.rel_to_stypeset[r]= set(utils.removeElements(self.rel_to_stypeset[r],3)) #filter out rare types
                    self.rel_to_otypeset[r]= set(utils.removeElements(self.rel_to_otypeset[r],3))
                #-----------#
            

            self.id_typeint=torch.from_numpy(numpy.array(self.id_typeint)).cuda() #load on gpu
            # print("Shape of id_typeint: ",self.id_typeint.shape)


        self.knowns_o = {} #o seen w/t s,r
        self.knowns_s = {} 
        print("building all known database from joint kb")
        for fact in self.all_kb.facts:
            if (fact[0], fact[1]) not in self.knowns_o:
                self.knowns_o[(fact[0], fact[1])] = set()
            self.knowns_o[(fact[0], fact[1])].add(fact[2])

            if (fact[2], fact[1]) not in self.knowns_s:
                self.knowns_s[(fact[2], fact[1])] = set()
            self.knowns_s[(fact[2], fact[1])].add(fact[0])


        print("converting to lists")
        for k in self.knowns_o:
            self.knowns_o[k] = list(self.knowns_o[k])
        for k in self.knowns_s:
            self.knowns_s[k] = list(self.knowns_s[k])
        print("done")

    def get_typemap(self,type_filename):
        """
        return a list L of type id's, with L[i] denoting type id for entity id i 
        """ 
        type_to_ids=dd(lambda: [])
        id_to_type=dd(str)
        ent_to_id={}#dd(int)
        for e,eid in self.all_kb.entity_map.items():
            # print(e)
            ent_to_id[e]=eid

        print("Number of entities in rem: {}".format(len(list(ent_to_id.keys()))))
        with open(type_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                ent_id=ent_to_id[row[0]]
                ent_type=row[2]
                # type_to_ids[ent_type].append(ent_id)
                id_to_type[ent_id]=ent_type

        id_typeint=[]
        typeint=dd(lambda: len(typeint)+1)

        for eid in range(len(list(ent_to_id.keys()))):
            typ=id_to_type[eid]
            id_typeint.append(typeint[typ])

        return id_typeint

    def get_knowns(self, e, r, flag_s_o=0):
        """
        computes and returns the set of all entites that have been seen as a fact (s, r, _) or (_, r, o)\n
        :param e: The head(s)/tail(o) of the fact
        :param r: The relation of the fact
        :param flag_s_o: whether e is s #s is fixed we try o
        :return: All entites o such that (s, r, o) has been seen in all_kb 
                 OR
                 All entites s such that (s, r, o) has been seen in all_kb 
        """
        if flag_s_o:
            ks = [self.knowns_o[(a, b)] for a,b in zip(e, r)]
        else:
            ks = [self.knowns_s[(a, b)] for a,b in zip(e, r)]

        lens = [len(x) for x in ks]
        max_lens = max(lens)
        ks = [numpy.pad(x, (0, max_lens-len(x)), 'edge') for x in ks]
        result= numpy.array(ks)
        return result

    def test_hard_reflexive_constraint(self, scores, s, minimum_value):
        return scores.scatter_(1, s, minimum_value)

    def filter_scores(self,scores,mode, s,r,o,flag_s_o):
        """
        if mode=='type', scores of all entities with type not matching gold type set to minimum.
        if mode=='typeset', scores of all entities with type not in typeset for the relation set to minimum.
        """
        if(mode=='typeset'):
            to_pred,type_set=[],[]
            if(flag_s_o):
                to_pred=o
                type_set=self.rel_to_otypeset
            else:
                to_pred=s
                type_set=self.rel_to_stypeset

            num_samples=scores.shape[0]

            for i in range(num_samples):
                typeid=int(self.id_typeint[int(to_pred[i]) ])

                all_types=set(self.id_typeint.tolist())
                all_types.remove(typeid)

                for typ in [j for j in all_types if j not in type_set[int(r[i])]]:
                    scores[i,:][torch.eq(self.id_typeint,typ)]=self.scoring_function.minimum_value

        elif(mode=='type'):
            to_pred=[]
            if(flag_s_o):
                to_pred=o
            else:
                to_pred=s

            num_samples=scores.shape[0]
            for i in range(num_samples):
                typeid=int(self.id_typeint[int(to_pred[i]) ])
                scores[i,:][torch.ne(self.id_typeint,typeid)]=self.scoring_function.minimum_value

        return
    #def filter_lang(self,scores,idx):
        
    def forward(self, s, r, o, knowns, flag_s_o=0,filt=[]):
        """
        Returns the rank of o for the query (s, r, _) as given by the scoring function\n
        :param s: The head of the query
        :param r: The relation of the query
        :param o: The Gold object of the query
        :param knowns: The set of all o that have been seen in all_kb with (s, r, _) as given by ket_knowns above
        :return: rank of o, score of each entity and score of the gold o
        """

        #pdb.set_trace()
        if flag_s_o:
            scores = self.scoring_function(s, r, None).data
            score_of_expected = scores.gather(1, o.data)
            #scores = test_hard_reflexive_constraint(scores, s, self.scoring_function.minimum_value) # making scores of all e2 same as e1 low
        else:
            scores = self.scoring_function(None, r, o).data
            score_of_expected = scores.gather(1, s.data)
            #scores = test_hard_reflexive_constraint(scores, o, self.scoring_function.minimum_value) # making scores of all e1 same as e2 low
        scores.scatter_(1, knowns, self.scoring_function.minimum_value)
        if(self.type_filter is True):
            self.filter_scores(scores,self.type_filter_mode,s,r,o,flag_s_o)
        if len(filt) != 0:
            scores=scores[:,filt]
        # print(len(scores[0]))
        greater = scores.ge(score_of_expected).float()
        # print(greater)
        equal = scores.eq(score_of_expected).float()
        rank = greater.sum(dim=1)+1+equal.sum(dim=1)/2.0
        return rank, scores, score_of_expected

    def forward_lx(self, s, r, o, knowns, flag_s_o=0, flag_avg=0):
        """
        Returns the rank of o for the query (s, r, _) as given by the scoring function\n
        :param s: The head of the query
        :param r: The relation of the query
        :param o: The Gold object of the query
        :param knowns: The set of all o that have been seen in all_kb with (s, r, _) as given by ket_knowns above
        :return: rank of o, score of each entity and score of the gold o
        """
        #pdb.set_trace()
        if flag_s_o:
            scores = self.scoring_function(s, r, None).data
            if flag_avg:
                r_rev = r + self.scoring_function.relation_count
                scores_inv = self.scoring_function(None, r_rev, s).data
                scores += scores_inv
            score_of_expected = scores.gather(1, o.data)
            #scores = test_hard_reflexive_constraint(scores, s, self.scoring_function.minimum_value) # making scores of all e2 same as e1 low
        else:

            # r_rev = r + (self.scoring_function.relation_count/2)
            r_rev = r + self.scoring_function.relation_count
            scores = self.scoring_function(o, r_rev, None).data
            if flag_avg:
                scores_inv = self.scoring_function(None, r, o).data
                scores += scores_inv
            score_of_expected = scores.gather(1, s.data)
            #scores = test_hard_reflexive_constraint(scores, o, self.scoring_function.minimum_value) # making scores of all e1 same as e2 low
        
        scores.scatter_(1, knowns, self.scoring_function.minimum_value)

        if(self.type_filter is True):
            self.filter_scores(scores,self.type_filter_mode,s,r,o,flag_s_o)
        
            
        greater = scores.ge(score_of_expected).float()
        equal = scores.eq(score_of_expected).float()
        rank = greater.sum(dim=1)+1+equal.sum(dim=1)/2.0
        return rank, scores, score_of_expected


@torch.no_grad()
def evaluate(name, ranker_: ranker, kb_: kb, batch_size, verbose=0, top_count=5,
             hooks=None, filt=[], save=0, save_text="", save_eval=None):
    """
    Evaluates an entity ranker on a knowledge base, by computing mean reverse rank, mean rank, hits 10 etc\n
    Can also print type prediction score with higher verbosity.\n
    :param name: A name that is displayed with this evaluation on the terminal
    :param ranker: The ranker that is used to rank the entites
    :param kb: The knowledge base to evaluate on. Must be augmented with type information when used with higher verbosity
    :param batch_size: The batch size of each minibatch
    :param verbose: The verbosity level. More info is displayed with higher verbosity
    :param top_count: The number of entities whose details are stored
    :param hooks: The additional hooks that need to be run with each mini-batch
    :param save_eval: Object with save(...) method to save to file the
     reciprocal rank for each test instance
    :return: A dict with the mrr, mr, hits10 and hits1 of the ranker on kb
    """
    if hooks is None:
        hooks = []
    totals = {"e2":{"mrr":0, "mr":0, "hits10":0, "hits1":0}, "e1":{"mrr":0, "mr":0, "hits10":0, "hits1":0}, "m":{"mrr":0, "mr":0, "hits10":0, "hits1":0}}
    start_time = time.time()
    facts = kb_.facts
    if(verbose>0):
        totals["correct_type"]={"e1":0, "e2":0}
        entity_type_matrix = kb_.entity_type_matrix.cuda()
        ##CPU mode
        # entity_type_matrix = kb.entity_type_matrix        
        for hook in hooks:
            hook.begin()
    for i in range(0, int(facts.shape[0]), batch_size):
        start = i
        end = min(i+batch_size, facts.shape[0])
        s = facts[start:end, 0]
        r = facts[start:end, 1]
        o = facts[start:end, 2]
        
        knowns_o = ranker_.get_knowns(s, r, flag_s_o=1)
        knowns_s = ranker_.get_knowns(o, r, flag_s_o=0)

        s = torch.autograd.Variable(torch.from_numpy(s).cuda().unsqueeze(1), requires_grad=False)
        r = torch.autograd.Variable(torch.from_numpy(r).cuda().unsqueeze(1), requires_grad=False)
        o = torch.autograd.Variable(torch.from_numpy(o).cuda().unsqueeze(1), requires_grad=False)
        knowns_s = torch.from_numpy(knowns_s).cuda()
        knowns_o = torch.from_numpy(knowns_o).cuda()

        ##CPU mode
        # s = torch.autograd.Variable(torch.from_numpy(s).unsqueeze(1), requires_grad=False)
        # r = torch.autograd.Variable(torch.from_numpy(r).unsqueeze(1), requires_grad=False)
        # o = torch.autograd.Variable(torch.from_numpy(o).unsqueeze(1), requires_grad=False)
        # knowns_s = torch.from_numpy(knowns_s)
        # knowns_o = torch.from_numpy(knowns_o)

        #pdb.set_trace()

        if ranker_.flag_add_reverse:
            ranks_o, scores_o, score_of_expected_o = ranker_.forward_lx(s, r, o, knowns_o, flag_s_o=1, flag_avg=ranker_.flag_avg_scores)
            ranks_s, scores_s, score_of_expected_s = ranker_.forward_lx(s, r, o, knowns_s, flag_s_o=0, flag_avg=ranker_.flag_avg_scores)
        else:
            ranks_o, scores_o, score_of_expected_o = ranker_.forward(s, r, o, knowns_o, flag_s_o=1,filt=filt)
            ranks_s, scores_s, score_of_expected_s = ranker_.forward(s, r, o, knowns_s, flag_s_o=0,filt=filt)

        #e1,r,?
        totals['e2']['mr'] += ranks_o.sum()
        totals['e2']['mrr'] += (1.0/ranks_o).sum()
        totals['e2']['hits10'] += ranks_o.le(11).float().sum()
        totals['e2']['hits1'] += ranks_o.eq(1).float().sum()
        #?,r,e2 
        totals['e1']['mr'] += ranks_s.sum()
        totals['e1']['mrr'] += (1.0/ranks_s).sum()
        totals['e1']['hits10'] += ranks_s.le(11).float().sum()
        totals['e1']['hits1'] += ranks_s.eq(1).float().sum()

        totals['m']['mr'] += (ranks_s.sum()+ranks_o.sum())/2.0
        totals['m']['mrr'] += ((1.0/ranks_s).sum()+(1.0/ranks_o).sum())/2.0
        totals['m']['hits10'] += (ranks_s.le(11).float().sum() + ranks_o.le(11).float().sum())/2.0
        totals['m']['hits1'] += (ranks_s.eq(1).float().sum() + ranks_o.eq(1).float().sum())/2.0

        if save_eval:
            save_eval.save(name, s.data, r.data, o.data, ranks_s, ranks_o)

        extra = ""
        if save or verbose > 0:
            scores_s.scatter_(1, s.data, score_of_expected_s)
            top_scores_s, top_predictions_s = scores_s.topk(top_count, dim=-1)
            if verbose > 0:
                top_predictions_type_s = torch.nn.functional.embedding(top_predictions_s, entity_type_matrix).squeeze(-1)
                expected_type_s = torch.nn.functional.embedding(s, entity_type_matrix).squeeze()
                correct_s = expected_type_s.eq(top_predictions_type_s[:, 0]).float()
                correct_count_s = correct_s.sum()
                totals["correct_type"]["e1"] += correct_count_s.item()#data[0]
                extra += "TP-s error %5.3f |" % (100*(1.0-totals["correct_type"]["e1"]/end)) #?,r,o
                for hook in hooks:
                    hook(s.data, r.data, o.data, ranks_s, top_scores_s, top_predictions_s, expected_type_s, top_predictions_type_s)
        
            scores_o.scatter_(1, o.data, score_of_expected_o)
            top_scores_o, top_predictions_o = scores_o.topk(top_count, dim=-1)
            if verbose > 0:
                top_predictions_type_o = torch.nn.functional.embedding(top_predictions_o, entity_type_matrix).squeeze(-1)
                expected_type_o = torch.nn.functional.embedding(o, entity_type_matrix).squeeze()
                correct_o = expected_type_o.eq(top_predictions_type_o[:, 0]).float()
                correct_count_o = correct_o.sum()
                totals["correct_type"]["e2"] += correct_count_o.item()#data[0]
                extra += "TP-o error %5.3f |" % (100*(1.0-totals["correct_type"]["e2"]/end)) #s,r,?
                for hook in hooks:
                    hook(s.data, r.data, o.data, ranks_o, top_scores_o, top_predictions_o, expected_type_o, top_predictions_type_o)

            if save:
                #with open("analysis_r_e1e2_e1Type_e2Type_e1Predictede2Predicted_e1PredictedType_e2PredictedType.csv","w") as f:    
                with open(save_text+"analysis_r_e1e2_e1Predictede2Predicted_e1Ranke2Rank.csv","a") as f:
                    #writer = csv.writer(f, delimiter='\t')
                    #print(r.data.shape,s.data.shape,o.data.shape,top_predictions_s.data.shape,top_predictions_o.data.shape)
                    #print("ready")
                    #print("r",r.data.cpu().numpy())
                    #print("s",s.data.cpu().numpy())
                    #print("o",o.data.cpu().numpy()[:,0])
                    #print("top_predictions_o",top_predictions_o.data.cpu().numpy())
                    #print("top_predictions_s",top_predictions_s.data.cpu().numpy()[:,0])
                    #print("ranks_s",ranks_s.data.cpu().numpy().shape);
                    #print("ranks_s",ranks_s.data.cpu().numpy()[:]);exit()
                    for r_w,s_w,o_w,ts_w,to_w, r_s_w, r_o_w in zip(r.data.cpu().numpy()[:,0],s.data.cpu().numpy()[:,0],o.data.cpu().numpy()[:,0],top_predictions_s.data.cpu().numpy()[:,0],top_predictions_o.data.cpu().numpy()[:,0],ranks_s.data.cpu().numpy()[:],ranks_o.data.cpu().numpy()[:]):
                        #print("##",[r_w,s_w,o_w,ts_w,to_w])
                        f.write(("\t").join([str(ele) for ele in [r_w,s_w,o_w,ts_w,to_w,r_s_w, r_o_w]])+"\n")
                    f.flush()
                    f.close();
        """
        utils.print_progress_bar(end, facts.shape[0],
                                 "Eval on %s" % name,
                                 (("|M|  mrr:%3.2f|h10:%3.2f%"
                                   "%|h1:%3.2f|time %5.0f|") %
                                  (100.0*totals['e2']['mrr']/end,
                                   100.0*totals['e2']['hits10']/end,
                                   100.0*totals['e2']['hits1']/end,
                                   time.time()-start_time)) + extra,
                                 color="green")
        """
    # for batch i
    print("_EVAL_\t[%s]\tmrr:%3.2f\th10:%3.2f\th1:%3.2f" %
          (name, 100.0*totals['e2']['mrr']/end,
           100.0*totals['e2']['hits10']/end,
           100.0*totals['e2']['hits1']/end))
    # print("__EVAL__DONE__")

    gc.collect()
    torch.cuda.empty_cache()
    for hook in hooks:
        hook.end()
            
    totals['m'] = {x:totals['m'][x]/facts.shape[0] for x in totals['m']}
    totals['e1'] = {x:totals['e1'][x]/facts.shape[0] for x in totals['e1']}
    totals['e2'] = {x:totals['e2'][x]/facts.shape[0] for x in totals['e2']}

    # transfer to an output dict replacing tensors by their values
    answer = copy.deepcopy(totals)
    for (k0, v0) in answer.items():
        if type(v0) == dict:
            for (k1,v1) in v0.items():
                if type(v1) == Tensor:
                    v0[k1] = v1.item()
    # print("__ANSWER__", answer)
    return answer
