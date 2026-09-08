import json
import os
from collections import OrderedDict
from typing import List

import torch
import torch.utils.data
import tqdm

from config import args
from dict_hub import build_tokenizer
from doc_relation_v2_singal_with_head import collate, Example, Dataset
from logger_config import logger
from models_path_token_muti import build_model
from utils import AttrDict, move_to_cuda


class BertPredictor:

    def __init__(self, model=None, train_args=None):
        self.model = model
        self.train_args = train_args if train_args is not None else AttrDict()
        self.use_cuda = False

    def load(self, ckt_path, use_data_parallel=False):
        assert os.path.exists(ckt_path), ckt_path
        ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        build_tokenizer(self.train_args)
        self.model = build_model(self.train_args)

        # DataParallel will introduce 'module.' prefix
        state_dict = ckt_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'hr_queue' in k or 'hr_queue_ptr' in k or 'tail_queue' in k or 'tail_queue_ptr' in k:
                continue
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        #     , strict=True
        self.model.load_state_dict(new_state_dict, strict=False)

        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info('Use data parallel predictor')
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.use_cuda = True
        elif torch.cuda.is_available():
            self.model.cuda()
            self.use_cuda = True
        logger.info('Load model from {} successfully'.format(ckt_path))
        self.model.eval()

    def _setup_args(self):
        for k, v in args.__dict__.items():
            if k not in self.train_args.__dict__:
                logger.info('Set default attribute: {}={}'.format(k, v))
                self.train_args.__dict__[k] = v
        self.train_args.momentum = False
        logger.info('Args used in training: {}'.format(json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)))
        args.use_link_graph = self.train_args.use_link_graph
        args.is_test = True

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example], only_head_embedding=False):
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=args.workers,
            batch_size=max(args.batch_size, 512 * torch.cuda.device_count()),
            collate_fn=collate,
            shuffle=False)

        hr_tensor_list, tail_tensor_list = [], []
        hp_tensor_list = []
        if_path = []
        for batch_dict in tqdm.tqdm(data_loader):
            batch_dict['only_head_embedding'] = only_head_embedding
            if self.use_cuda and ('A100' not in torch.cuda.get_device_name(0) or torch.cuda.device_count() == 1 or
                                  args.distributed):
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            hr_tensor_list.append(outputs['hr_vector'].detach().cpu())
            if_path.extend(batch_dict['if_path'])
            hp_tensor_list.append(outputs['hr_ori_vector'].detach().cpu())
            if not only_head_embedding:
                tail_tensor_list.append(outputs['tail_vector'])

        if only_head_embedding:
            return torch.cat(hr_tensor_list, dim=0)
        else:
            return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0),torch.cat(hp_tensor_list, dim=0), if_path

    @torch.no_grad()
    def predict_by_entities(self, entity_exs) -> torch.tensor:
        examples = []
        for entity_ex in entity_exs:
            examples.append(Example(head_id='', relation='',
                                    tail_id=entity_ex.entity_id))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=args.workers,
            batch_size=max(args.batch_size, 512 * torch.cuda.device_count()),
            collate_fn=collate,
            shuffle=False)

        ent_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            batch_dict['only_ent_embedding'] = True
            if self.use_cuda and ('A100' not in torch.cuda.get_device_name(0) or torch.cuda.device_count() == 1 or
                                  args.distributed):
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])

        return torch.cat(ent_tensor_list, dim=0)
    @torch.no_grad()
    def rerank_by_path(self, 
                    batch_score: torch.tensor,
                    examples: List[Example],
                    hr_tensor: torch.tensor,
                    entity_dict,
                    rerank_num=0):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  
        n = rerank_num
        batch_size = batch_score.size(0)
        top_n_indices_list = []
        for idx in range(batch_size):
            _, sorted_indices = torch.sort(batch_score[idx], dim=-1, descending=True)
            top_n_indices = sorted_indices[:n]
            top_n_indices_list.append(top_n_indices)
            #batch_score[idx] = 1

        
        all_top_triplets = []
        for idx in range(batch_size):
            cur_ex = examples[idx]
            top_n_entities = [entity_dict.get_entity_by_idx(idx.item()) for idx in top_n_indices_list[idx]]
            top_triplets = create_triplets(cur_ex, top_n_entities)
            all_top_triplets.extend(top_triplets)

        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=all_top_triplets, task=args.task),
            num_workers=args.workers * torch.cuda.device_count(),  # 增加num_workers
            batch_size=max(args.batch_size, 512 * torch.cuda.device_count()),
            collate_fn=collate,
            shuffle=False,
            pin_memory=True
        )

        # 获取所有的hp向量
        all_hp_2 = []
        all_hp_3 = []
        all_weight_mask = []
        for batch_dict in tqdm.tqdm(data_loader):
            batch_dict['only_hp_embedding'] = True
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            #diagonal_elements = batch_dict['weight_mask'].diag()
            #all_weight_mask.append(diagonal_elements)
            all_hp_2.append(outputs['hp_2_vector'])
            all_hp_3.append(outputs['hp_3_vector'])

        # 合并所有hp向量
        all_hp_2 = torch.cat(all_hp_2, dim=0)  # [total_triplets, hidden_dim]
        all_hp_3 = torch.cat(all_hp_3, dim=0) 
        #all_weight_mask = torch.cat(all_weight_mask, dim=0)  # [total_triplets]
        # 将hr_tensor扩展为与all_hp对应的形式
        if self.use_cuda:
            hr_tensor = hr_tensor.to(all_hp_3.device)

        # 计算每个样本的delta并更新batch_score
        start_idx = 0
        for idx in range(batch_size):
            hr = hr_tensor[idx].unsqueeze(0)  # [1, hidden_dim]
            num_triplets = len(top_n_indices_list[idx])
            end_idx = start_idx + num_triplets

            hp_2_slice = all_hp_2[start_idx:end_idx]
            hp_3_slice = all_hp_3[start_idx:end_idx]
            weight = all_weight_mask[start_idx:end_idx]
            start_idx = end_idx

            logits_2 = hr.mm(hp_2_slice.t()).squeeze(0)
            logits_3 = hr.mm(hp_3_slice.t()).squeeze(0)
            logits = torch.max(logits_2, logits_3)
            #logits = logits_2+logits_3
            #logits = torch.nn.functional.softmax(logits, dim=-1)
            delta = logits.to(batch_score.device)
            indices = top_n_indices_list[idx].to(batch_score.device)

            batch_score[idx].index_add_(0, indices, delta)

        return batch_score
    # def rerank_by_path(self, 
    #                 batch_score: torch.tensor,
    #                 examples: List[Example],
    #                 hr_tensor: torch.tensor,
    #                 entity_dict):
    #     for idx in tqdm.tqdm(range(batch_score.size(0))):
    #         cur_ex = examples[idx]
    #         hr=hr_tensor[idx]
    #         sorted_score, sorted_indices = torch.sort(batch_score[idx], dim=-1, descending=True)
    #         n=50
    #         top_n = sorted_indices[:n]
    #         top_n_entities = [entity_dict.get_entity_by_idx(idx.item()) for idx in top_n]
    #         top_triplets= create_triplets(cur_ex, top_n_entities)
    #         data_loader = torch.utils.data.DataLoader(
    #             Dataset(path='', examples=top_triplets, task=args.task),
    #             num_workers=args.workers,
    #             batch_size=max(args.batch_size, 512 * torch.cuda.device_count()),
    #             collate_fn=collate,
    #             shuffle=False)
    #         for idx, batch_dict in enumerate(data_loader):
    #             batch_dict['only_hp_embedding'] = True
    #             if self.use_cuda and ('A100' not in torch.cuda.get_device_name(0) or torch.cuda.device_count() == 1 or
    #                                 args.distributed):
    #                 batch_dict = move_to_cuda(batch_dict)
    #             outputs = self.model(**batch_dict)
    #             hp=outputs['hr_vector']
    #             hr = hr.unsqueeze(0)
    #             logits=hr.mm(hp.t())
    #             scores = logits[0]
    #             delta = scores.to(batch_score.device)
    #             sorted_indices=torch.LongTensor(list(top_n)).to(batch_score.device)
    #             batch_score[idx].index_add_(0, sorted_indices, delta)
    #     return batch_score




#组建三元组，输入头实体和实体列表
def create_triplets(head_entity: str, entity_list: List[str]) -> List[Example]:
    top_triplets= []
    head_id=head_entity.head_id
    relation=head_entity.relation
    for entity in entity_list:
        entity_id=entity.entity_id
        top_triplets.append(Example(head_id=head_id, relation='',
                                    tail_id=entity_id))
    return top_triplets
