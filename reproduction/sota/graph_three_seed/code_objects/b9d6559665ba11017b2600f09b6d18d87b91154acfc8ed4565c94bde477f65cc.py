import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import nodes_to_k_graph, ranking_all_batch

import os
import time
import logging




def hr2t_from_splits(data_dir, target_lang, splits):
    """Collect known tails for the requested data splits."""
    hr2t = {}
    for split in splits:
        frame = pd.read_csv(
            os.path.join(data_dir, target_lang + '-' + split + '.tsv'),
            sep='\t', header=None,
        )
        tripleset = set(tuple(row) for row in frame.values)
        for h, r, t in tripleset:
            key = (int(h), int(r))
            hr2t.setdefault(key, set()).add(int(t))
    return hr2t


def hr2t_from_train_set(data_dir, target_lang):
    """Released evaluator compatibility: despite its name, use train+val."""
    return hr2t_from_splits(data_dir, target_lang, ('train', 'val'))

class Tester:
    def __init__(self, args, kg_objects_dict, model, device, data_dir):
        self.args = args
        self.kg_objects_dict = kg_objects_dict
        self.model = model
        self.device = device
        self.data_dir = data_dir
    
    
    def test(self, is_val=True, is_filtered=False, filter_splits=None,
             keep_ground_truth=False):
        """
        is_val: 验证集or测试集
        is_lifted: 
        """
        results = {}
        for kg_name in self.kg_objects_dict:
            results[kg_name] = self.test_kg(
                kg_name, is_val, is_filtered, filter_splits,
                keep_ground_truth,
            )
        
        return results
    
    def test_kg(self, kg_name, is_val=True, is_filtered=False,
                filter_splits=None, keep_ground_truth=False):
        time0 = time.time()
        kg = self.kg_objects_dict[kg_name]
        if is_val:
            h = kg.h_val
            r = kg.r_val
            t = kg.t_val
            output_text = '[' + kg_name + ']' + "Val:"
        else:
            h = kg.h_test
            r = kg.r_test
            t = kg.t_test
            output_text = '[' + kg_name + ']' + "Test:"
        num_samples = h.shape[0]
        ground_truth = t.view(-1, 1).to(self.device) # [num_samples, 1]
        kg_batch_generator = kg.generate_batch_data(h, r, t, batch_size=self.args.test_batch_size, shuffle=False)
        ground_truth_generator = DataLoader(ground_truth, batch_size=self.args.test_batch_size, shuffle=False)
    
        num_total_entities = kg.num_entities
        self.pre_compute_all_embeddings(kg_name)
        
        if is_filtered and not is_val:
            if filter_splits is None:
                hr2t_train = hr2t_from_train_set(
                    self.data_dir + '/kg', kg_name
                )
            else:
                hr2t_train = hr2t_from_splits(
                    self.data_dir + '/kg', kg_name, filter_splits
                )
        
        hits_1_compute, hits_10_compute, rranks_sum = 0, 0, 0
        for kg_batch, ground_truth_batch in zip(kg_batch_generator, ground_truth_generator):
            h_batch = kg_batch[:, 0] # [test_batch_size,]
            r_batch = kg_batch[:, 1].to(self.device) # [test_batch_size,]
            h_embedding = kg.computed_entity_embedding_kg[h_batch, :] # [test_batch_size, d]
            model_predictions = self.model.predict(h_embedding, r_batch) # h + r [test_batch_size, 1, d]
            model_predictions = torch.squeeze(model_predictions, dim=1) # [test_batch_size, d]
            ranking_indices, ranking_scores = ranking_all_batch(model_predictions, kg.computed_entity_embedding_kg)
            # [test_batch_size, num_total_entities], [test_batch_size, num_total_entities]

            if is_filtered and not is_val:
                # Mark only the (usually few) known tails for each query.  The
                # released implementation copied every full ranking to CPU and
                # ran a Python callback once per candidate entity.  Computing
                # the post-filter rank from this mask preserves its stable
                # partition exactly while keeping the full work batched.
                filter_mask = torch.zeros(
                    (h_batch.shape[0], num_total_entities),
                    dtype=torch.bool,
                    device=self.device,
                )
                filter_rows = []
                filter_entities = []
                for i in range(h_batch.shape[0]):
                    query = (h_batch[i].item(), r_batch[i].item())
                    known_tails = hr2t_train.get(query)
                    if known_tails:
                        filter_rows.extend([i] * len(known_tails))
                        filter_entities.extend(known_tails)
                if filter_rows:
                    filter_mask[
                        torch.tensor(filter_rows, device=self.device),
                        torch.tensor(filter_entities, device=self.device),
                    ] = True
                if keep_ground_truth:
                    # Standard filtered KGC removes every other known answer
                    # while retaining the answer currently being ranked.
                    filter_mask.scatter_(
                        1, ground_truth_batch.reshape(-1, 1), False
                    )
                batch_hits_1, batch_hits_10, batch_rranks_sum = self.get_filtered_hit_mrr(
                    ranking_indices, ground_truth_batch, filter_mask
                )
            else:
                batch_hits_1, batch_hits_10, batch_rranks_sum = self.get_hit_mrr(
                    ranking_indices, ground_truth_batch
                )
            hits_1_compute += batch_hits_1
            hits_10_compute += batch_hits_10
            rranks_sum += batch_rranks_sum
            
        hits_1_ratio = hits_1_compute / num_samples
        hits_10_ratio = hits_10_compute / num_samples
        mrr = rranks_sum / num_samples

        if is_filtered and not is_val:
            logging.info('{} filterd: {:.8f}, {:.8f}, {:.8f}'.format(output_text, hits_1_ratio, hits_10_ratio, mrr))
        else:
            logging.info('{} {:.8f}, {:.8f}, {:.8f}'.format(output_text, hits_1_ratio, hits_10_ratio, mrr))
        print('time: %s' % (time.time() - time0))

        return [hits_1_ratio, hits_10_ratio, mrr]

    
    def pre_compute_all_embeddings(self, kg_name):
        with torch.no_grad():
            kg = self.kg_objects_dict[kg_name]
            if kg.computed_entity_embedding_kg is not None:
                return
            kg_index = self.args.kgname2idx[kg_name]
            node_index_tensor = torch.arange(kg.num_entities)
            precompute_batch_size = getattr(self.args, 'precompute_batch_size', 0) \
                or self.args.test_batch_size
            dataloader = DataLoader(node_index_tensor, batch_size=precompute_batch_size, shuffle=False)
            embedding_list = []
            for data in dataloader:
                graph_batch_list = nodes_to_k_graph(kg.k_subgraph_list, data, self.args.device)
                node_embeddings = self.model.forward_GNN_embedding(graph_batch_list, kg_index)
                embedding_list.append(node_embeddings)
            embedding_table = torch.cat(embedding_list, dim=0) # [num_entities, d]
            self.kg_objects_dict[kg_name].computed_entity_embedding_kg = embedding_table


    def get_hit_mrr(self, topk_indices_all, ground_truth):
        
        zero_tensor = torch.tensor([0]).to(ground_truth.device)
        one_tensor = torch.tensor([1]).to(ground_truth.device)
        
        #
        hits_1 = torch.where(ground_truth == topk_indices_all[:, :1], one_tensor, zero_tensor).sum().item()
        hits_10 = torch.where(ground_truth == topk_indices_all[:, :10], one_tensor, zero_tensor).sum().item()
        
        gt_expanded = ground_truth.expand_as(topk_indices_all) # [N, k]
        hits = (gt_expanded == topk_indices_all).nonzero() # [N, 2]
        ranks = hits[:, -1] + 1 # [N,]
        # assert ranks.shape[0] == ground_truth.shape[0]
        ranks = ranks.float()
        rranks = torch.reciprocal(ranks)
        rranks_sum = torch.sum(rranks)
        
        return hits_1, hits_10, rranks_sum


    def get_filtered_hit_mrr(self, ranking_indices, ground_truth, filter_mask):
        """Metrics after the released evaluator's stable known-tail filter."""
        ground_truth = ground_truth.reshape(-1)
        ranked_is_filtered = filter_mask.gather(1, ranking_indices)
        matches = ranking_indices.eq(ground_truth.unsqueeze(1))
        original_position = matches.to(torch.long).argmax(dim=1)
        columns = torch.arange(ranking_indices.shape[1], device=ranking_indices.device)
        filtered_before = (
            ranked_is_filtered & (columns.unsqueeze(0) < original_position.unsqueeze(1))
        ).sum(dim=1)
        ranks = original_position + 1 - filtered_before
        target_is_filtered = filter_mask.gather(1, ground_truth.unsqueeze(1)).squeeze(1)
        valid = ~target_is_filtered
        hits_1 = ((ranks <= 1) & valid).sum().item()
        hits_10 = ((ranks <= 10) & valid).sum().item()
        # Select valid rows before reciprocal/sum to retain the released
        # evaluator's operation order (and therefore bitwise result).
        rranks_sum = torch.reciprocal(ranks[valid].to(torch.float32)).sum()
        return hits_1, hits_10, rranks_sum


