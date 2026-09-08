from __future__ import division
import time
import pandas as pd
import numpy as np

import logging
from os.path import join
import torch
from src.utils import nodes_to_graph,Ranking_all_batch



class Tester:
    def __init__(self, target_kg, supporter_kgs,model,device,data_dir):
        """
        :param target_kg: KnowledgeGraph object
        :param support_kgs: list[KnowledgeGraph]
        """
        self.target_kg = target_kg
        self.supporter_kgs = supporter_kgs
        self.device = device
        self.model = model
        self.data_dir = data_dir

       
    def get_hit_mrr(self,topk_indices_all,ground_truth):

        # ground_truth = ground_truth.repeat(1,kg.num_entity) #[n_test,n]
        zero_tensor = torch.tensor([0]).to(ground_truth.device)
        one_tensor = torch.tensor([1]).to(ground_truth.device)

        # Calculate Hit@1, Hit@10
        hits_1 = torch.where(ground_truth == topk_indices_all[:, [0]], one_tensor, zero_tensor).sum().item()
        hits_10 = torch.where(ground_truth == topk_indices_all[:, :10], one_tensor, zero_tensor).sum().item()


        # Calculate MRR
        gt_expanded = ground_truth.expand_as(topk_indices_all)
        hits = (gt_expanded == topk_indices_all).nonzero()
        ranks = hits[:, -1] + 1
        ranks = ranks.float()
        rranks = torch.reciprocal(ranks)
        mrr = torch.sum(rranks).data / ground_truth.size(0)

        return hits_1,hits_10,mrr

    @staticmethod
    def get_batch_hit_mrr(topk_indices, ground_truth):
        """Return exact metric counts for one exhaustive-ranking batch."""
        ground_truth = ground_truth.view(-1, 1)
        matches = topk_indices.eq(ground_truth)
        hits_1 = matches[:, :1].sum().item()
        hits_10 = matches[:, :10].sum().item()
        match_positions = matches.nonzero(as_tuple=False)
        if match_positions.shape[0] != ground_truth.shape[0]:
            raise RuntimeError(
                'Each unfiltered exhaustive ranking must contain its ground truth once'
            )
        reciprocal_rank_sum = torch.reciprocal(
            match_positions[:, 1].float() + 1.0
        ).sum().item()
        return hits_1, hits_10, reciprocal_rank_sum
    
    def test(self,args,is_val=True,is_lifted = False):
        """
        # for validation set!!

        Compute Hits@10 on first param.n_test test samples
        :param supporter_kg: needed when mode == KG1 or LINK_REDIRECT. None for KG0 or VOTING
        :param voting_function: used when mode==VOTING. Default: vote by count
        :return:
        """

        time0 = time.time()
        test_batch_size = getattr(args, 'test_batch_size', args.batch_size)
        precompute_batch_size = getattr(args, 'precompute_batch_size', args.batch_size)
        if is_val:
            samples = self.target_kg.h_val.shape[0]
            output_text = "Val:"
            kg_batch_generator = self.target_kg.generate_batch_data(self.target_kg.h_val, self.target_kg.r_val, self.target_kg.t_val, batch_size=test_batch_size, shuffle=False)
            
        else:
            samples = self.target_kg.h_test.shape[0]
            output_text = "Test:"
            kg_batch_generator = self.target_kg.generate_batch_data(self.target_kg.h_test, self.target_kg.r_test, self.target_kg.t_test, batch_size=test_batch_size, shuffle=False)
            
            
        total_entity_num = self.target_kg.num_entity
        self.pre_compute_all_embeddings(precompute_batch_size) # compute embeddings
                
        hits_1_compute = 0
        hits_10_compute = 0
        reciprocal_rank_sum = 0.0
        rows_seen = 0

        hr2t_train = None
        if is_lifted and not is_val:
            hr2t_train = hr2t_from_train_set(
                self.data_dir + 'kg', self.target_kg.lang
            )

        for kg_batch_each in kg_batch_generator:
            h_batch = kg_batch_each[:, 0].view(-1)
            r_batch = kg_batch_each[:, 1].to(self.device)  # global index
            h_embedding = self.target_kg.computed_entity_embedidng_KG[h_batch,:]
            model_predictions = self.model.predict(h_embedding, r_batch)
            model_predictions = torch.squeeze(model_predictions,dim=1)
            ranking_indices, ranking_scores = Ranking_all_batch(model_predictions, self.target_kg.computed_entity_embedidng_KG) 
            ground_truth_batch = kg_batch_each[:, 2].view(-1).to(self.device)

            if not (is_lifted and not is_val):
                batch_hits_1, batch_hits_10, batch_rr = self.get_batch_hit_mrr(
                    ranking_indices, ground_truth_batch
                )
                hits_1_compute += batch_hits_1
                hits_10_compute += batch_hits_10
                reciprocal_rank_sum += batch_rr
            else:
                # Preserve the repository's lifted filtering exactly, but
                # consume one row at a time instead of retaining every full
                # ranking on the GPU.
                for row_id in range(ranking_indices.shape[0]):
                    h_each = int(kg_batch_each[row_id, 0])
                    r_each = int(kg_batch_each[row_id, 1])
                    ranked = ranking_indices[row_id]
                    train_tails = hr2t_train.get((h_each, r_each), set())
                    if train_tails:
                        blocked = torch.tensor(
                            list(train_tails), dtype=ranked.dtype,
                            device=ranked.device,
                        )
                        ranked = ranked[~torch.isin(ranked, blocked)]
                    truth = ground_truth_batch[row_id]
                    positions = (ranked == truth).nonzero(as_tuple=False)
                    if positions.numel():
                        rank = int(positions[0, 0]) + 1
                        hits_1_compute += int(rank == 1)
                        hits_10_compute += int(rank <= 10)
                        reciprocal_rank_sum += 1.0 / rank

            rows_seen += ranking_indices.shape[0]
            del ranking_indices, ranking_scores, model_predictions

        assert rows_seen == samples
        mrr = reciprocal_rank_sum / samples
        
        hits_1_ratio = hits_1_compute/samples
        hits_10_ratio = hits_10_compute/samples
        
        # logging.info('===Validation %s===' % mode)
        
        if is_lifted and not is_val:
            logging.info('%s Hits@%d (%d triples,lifted): %f' % (output_text, 1, samples, hits_1_ratio))
            logging.info('%s Hits@%d (%d triples,lifted): %f' % (output_text, 10, samples, hits_10_ratio))
            logging.info('%s MRR (%d triples,lifted): %f' % (output_text,samples, mrr))
        else:
            logging.info('%s Hits@%d (%d triples): %f' % (output_text, 1, samples, hits_1_ratio))
            logging.info('%s Hits@%d (%d triples): %f' % (output_text, 10, samples, hits_10_ratio))
            logging.info('%s MRR (%d triples): %f' % (output_text,samples, mrr))
        print('time: %s' % (time.time() - time0))
        
        return [hits_1_ratio,hits_10_ratio,mrr]

    # def get_kg_embeddings_matrix(self,kg,batch_size,device):
    #     # All nodes in the dataset
    #     # model can be in cuda and outside of cuda
    #     node_index_tensor = torch.LongTensor([i for i in range(kg.num_entity)])
    #     graphs = nodes_to_graph(kg.subgraph_list_kg,node_index_tensor,batch_size)
    #
    #     embedding_list = []
    #     for graph_batch in graphs:
    #         assert graph_batch.edge_index.shape[1] == graph_batch.edge_attr.shape[0]
    #         graph_batch = graph_batch.to(device) # only used to retrive relations
    #         node_embeddings = self.model.forward_GNN_embedding(graph_batch,self.model.encoder_KG)
    #         embedding_list.append(node_embeddings)
    #
    #     embedding_table = torch.cat(embedding_list,dim=0).to(device) #[n,d]
    #
    #     return embedding_table
       
    def pre_compute_all_embeddings(self,batch_size):
        # no gradient compute!
        with torch.no_grad():
            self.target_kg.computed_entity_embedidng_KG = self.model.get_kg_embeddings_matrix(self.target_kg,batch_size,self.device,is_kg= True)
        
#             for supporter_KG in self.supporter_kgs:
#                 self.supporter_KG.computed_entity_embedidng_KG = self.get_kg_embeddings_matrix(self.target_kg,batch_size,self.device,is_aligned = False)


def extract_entities(id_score_tuples):
    return [ent_id for ent_id, score in id_score_tuples]


def filt_hits_at_n(results, lang, hr2t_train, n):
    """
    Filtered setting Hits@n when testing
    :param hr2t_train: {(h,r):set(t)}
    :param results: df, h,r,t, lang
    :return:
    """
    hits = 0
    for index, row in results.iterrows():
        t = row['t']
        predictions = row[lang]  # list[(entity,socre)]

        predictions = extract_entities(predictions)
        if (row['h'], row['r']) in hr2t_train:  # filter
            h, r = row['h'], row['r']
            predictions = [e for e in predictions if e not in hr2t_train[(h,r)]]
        predictions = predictions[:n]  # top n
        if t in predictions:
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@%d (%d triples)(filt): %.4f' % (n, results.shape[0], hits_ratio))
    return hits_ratio




def hr2t_from_train_set(data_dir, target_lang):
    train_df = pd.read_csv(join(data_dir, f'{target_lang}-train.tsv'), sep='\t')
    tripleset = set([tuple([h,r,t]) for h,r,t in (train_df.values)])

    hr2t = {}  # {(h,r):set(t)}
    for tp in tripleset:
        h,r,t=int(tp[0]),int(tp[1]),int(tp[2])
        if (h,r) not in hr2t:
            hr2t[(h,r)] = set()
        hr2t[(h,r)].add(t)
    return hr2t




