from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from prompter import Prompter
from triplet_mask import construct_mask


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits_sp: Union[torch.tensor, None]
    logits: Union[torch.tensor, None]
    logits_hr_hp_2: torch.tensor 
    logits_hr_hp_3: torch.tensor 
    logits_hp_2_t: torch.tensor
    logits_hp_3_t: torch.tensor
    logits_hr_ori: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    mask: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)

        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)
        self.num_insertions = 8
        self.hidden_size = self.config.hidden_size
        self.embed_dim=144
        self.learnable_embeddings_hr = nn.Embedding(474, self.embed_dim)#474,1644
        self.prompter = Prompter(self.embed_dim, self.num_insertions)
        # self.learnable_embeddings_hr = nn.Parameter(torch.randn(self.num_insertions, self.hidden_size), requires_grad=True)
        # self.learnable_embeddings_t = nn.Parameter(torch.randn(self.num_insertions, self.hidden_size), requires_grad=True)

    def _encode(self, encoder, token_ids, mask, token_type_ids,relation_id,insert=False):
        original_embeddings = encoder.embeddings(token_ids)
        batch_size, seq_length = token_ids.size()
        sep_token_id, cls_token_id = 102, 101
        modified_embeddings = []
        attention_mask_list = []
        token_type_ids_list = []
        sep_indices_all = (token_ids == sep_token_id).nonzero(as_tuple=False)
        mask_for_insertion = torch.ones(self.num_insertions, dtype=mask.dtype, device=mask.device)
        if insert:
            type_id_for_insertion = torch.ones(self.num_insertions, dtype=mask.dtype, device=mask.device)
            sep_indices=sep_indices_all[1::2, 1]

        else:
            type_id_for_insertion = torch.zeros(self.num_insertions, dtype=mask.dtype, device=mask.device)
            sep_indices=sep_indices_all[:, 1]
            insert_embeddings = self.learnable_embeddings_t
        for i in range(batch_size):
            index = torch.tensor([relation_id[i]],device='cuda')
            embedding_vector = self.learnable_embeddings_hr(index)
            insert_embeddings = self.prompter(embedding_vector)
            sep = sep_indices[i].item()
            embeddings = torch.cat([
                original_embeddings[i, :sep],
                insert_embeddings,
                original_embeddings[i, sep:]
                ], dim=0)

            attention_mask_modified = torch.cat([
                mask[i, :sep],
                mask_for_insertion,
                mask[i, sep:]
                ], dim=0)

            token_type_ids_modified = torch.cat([
                token_type_ids[i, :sep],
                type_id_for_insertion,
                token_type_ids[i, sep:]
                ], dim=0)

            token_type_ids_list.append(token_type_ids_modified)
            attention_mask_list.append(attention_mask_modified)
            modified_embeddings.append(embeddings)
        token_type_ids = torch.stack(token_type_ids_list)
        modified_embeddings = torch.stack(modified_embeddings)
        mask = torch.stack(attention_mask_list)

        outputs = encoder(
            inputs_embeds=modified_embeddings,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        cls_output = nn.functional.normalize(cls_output, dim=1)

        return cls_output
    def _encode_ori(self, encoder, token_ids, mask, token_type_ids):
        # device = next(encoder.parameters()).device

        # # 将所有输入张量移动到 encoder 的设备上
        # token_ids =  token_ids.to(device)
        # mask = mask.to(device)
        # token_type_ids = token_type_ids.to(device)
        outputs = encoder(input_ids=token_ids,
                        attention_mask=mask,
                        token_type_ids=token_type_ids,
                        return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        cls_output = nn.functional.normalize(cls_output, dim=1)
        return cls_output
    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                hrp_2_token_ids, hrp_2_mask, hrp_2_token_type_ids,
                hrp_3_token_ids, hrp_3_mask, hrp_3_token_type_ids,
                tail_token_ids=None, tail_mask=None, tail_token_type_ids=None,
                head_token_ids=None, head_mask=None, head_token_type_ids=None,
                relation_id=None,
                negative_token_ids=None, negative_mask=None, negative_token_type_ids=None,
                only_ent_embedding=False, only_head_embedding=False,only_hp_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)
        if only_head_embedding:
            return self.predict_head_embedding(hr_token_ids=hr_token_ids,
                                               hr_mask=hr_mask,
                                               hr_token_type_ids=hr_token_type_ids)
        if only_hp_embedding:
            return self.predict_hp_embedding(hrp_2_token_ids=hrp_2_token_ids,hrp_2_mask=hrp_2_mask,hrp_2_token_type_ids=hrp_2_token_type_ids,
                                             hrp_3_token_ids=hrp_3_token_ids,hrp_3_mask=hrp_3_mask,hrp_3_token_type_ids=hrp_3_token_type_ids)
        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids,
                                 relation_id=relation_id,
                                 insert=True
                                 )
        hr_ori_vector = self._encode_ori(self.hr_bert,
                                    token_ids=hr_token_ids,
                                    mask=hr_mask,
                                    token_type_ids=hr_token_type_ids
                                    )
        # hp_vector = self._encode(self.hr_bert,
        #                          token_ids=hrp_token_ids,
        #                          mask=hrp_mask,
        #                          token_type_ids=hrp_token_type_ids,
        #                          relation_id=relation_id,
        #                          insert=True)

        # tail_vector = self._encode(self.tail_bert,
        #                            token_ids=tail_token_ids,
        #                            mask=tail_mask,
        #                            token_type_ids=tail_token_type_ids,
        #                            insert=False)
        hp_2_vector = self._encode_ori(self.hr_bert,
                                 token_ids=hrp_2_token_ids,
                                 mask=hrp_2_mask,
                                 token_type_ids=hrp_2_token_type_ids)
        hp_3_vector = self._encode_ori(self.hr_bert,
                     token_ids=hrp_3_token_ids,
                     mask=hrp_3_mask,
                     token_type_ids=hrp_3_token_type_ids)

        tail_vector = self._encode_ori(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        if negative_token_ids is not None:
            negative_vector = self._encode(self.tail_bert,
                                           token_ids=negative_token_ids,
                                           mask=negative_mask,
                                           token_type_ids=negative_token_type_ids)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'hp_2_vector': hp_2_vector,
                'hp_3_vector': hp_3_vector,
                'hr_ori_vector': hr_ori_vector,
                # 'head_vector': head_vector.to('cuda:0') if head_token_ids is not None else None,
                'negative_vector': negative_vector if negative_token_ids is not None else None,}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        # 从 output_dict 中提取 'hr_vector' 和 'tail_vector'
        hr_vector,tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        
        # 获取批大小
        batch_size = hr_vector.size(0)
        hr_ori_vector = output_dict['hr_ori_vector']
        # 从 output_dict 中提取 'hp_vector'
        hp_2_vector = output_dict['hp_2_vector']
        hp_3_vector = output_dict['hp_3_vector']
        
        # 创建从 0 到 batch_size-1 的标签，并放到与 'hr_vector' 相同的设备上
        labels = torch.arange(batch_size).to(hr_vector.device)
        
        # 计算对比学习的 logits
        logits = hr_vector.mm(tail_vector.t())

        logits_hp_2_t = hp_2_vector.mm(tail_vector.t())
        logits_hp_3_t = hp_3_vector.mm(tail_vector.t())
        logits_hr_ori = hr_ori_vector.mm(tail_vector.t())
        if self.training:
            # 如果在训练模式下，从对角线元素中减去 margin
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
            logits_hp_2_t -= torch.zeros(logits_hp_2_t.size()).fill_diagonal_(self.add_margin).to(logits.device)
            logits_hp_3_t -= torch.zeros(logits_hp_3_t.size()).fill_diagonal_(self.add_margin).to(logits.device)
            logits_hr_ori -= torch.zeros(logits_hr_ori.size()).fill_diagonal_(self.add_margin).to(logits.device)
        
        # 用 log_inv_t 的指数值缩放 logits
        logits *= self.log_inv_t.exp()
        logits_hp_2_t *= self.log_inv_t.exp()
        logits_hp_3_t *= self.log_inv_t.exp()
        logits_hr_ori *= self.log_inv_t.exp()
        # 获取对比学习的三元组掩码
        triplet_mask_cl = batch_dict.get('triplet_mask_cl', None)
        
        if triplet_mask_cl is not None:
            # 在 triplet_mask_cl 为 False 的位置用一个大负值填充 logits
            logits.masked_fill_(~triplet_mask_cl, -1e4)
        
        # 计算跨度预测的 logits
        logits_sp = hr_vector.mm(tail_vector.t())
        
        # 计算 hr 和 hp 的 logits
        logits_hr_hp_2 = hr_vector.mm(hp_2_vector.t()) * self.log_inv_t.exp()
        logits_hr_hp_3 = hr_vector.mm(hp_3_vector.t()) * self.log_inv_t.exp()
        # 获取跨度预测的三元组掩码
        triplet_mask = batch_dict.get('triplet_mask', None)
        
        if self.training:
            # 如果在训练模式下，在 triplet_mask 为 False 的位置减去 margin
            logits_sp[~triplet_mask] -= self.add_margin
        
        # 用 log_inv_t 的指数值缩放 logits_sp
        logits_sp *= self.log_inv_t.exp()
        
        if output_dict['negative_vector'] is not None:
            # 如果存在负向量，则进行处理
            #print('negative_vector')
            negative_vector = output_dict['negative_vector']
            
            if negative_vector.dim() == 3:
                # 如果负向量有 3 个维度，则调整其形状
                num_negs = negative_vector.size(1)
                negative_vector = torch.reshape(negative_vector, [batch_size * num_negs, -1])
            
            # 计算跨度预测的负样本 logits
            negative_logits_sp = hr_vector.mm(negative_vector.t())
            negative_logits_sp *= self.log_inv_t.exp()
            
            # 计算对比学习的负样本 logits
            negative_logits_cl = hr_vector.mm(negative_vector.t())
            negative_logits_cl *= self.log_inv_t.exp()
            
            # 获取负样本的三元组掩码
            triplet_negative_mask = batch_dict.get('triplet_negative_mask', None)
            
            if triplet_negative_mask is not None:
                # 调整 negative_logits_sp 和 negative_logits_cl
                negative_logits_sp[~triplet_negative_mask] -= self.add_margin
                negative_logits_cl.masked_fill_(~triplet_negative_mask, -1e4)
            
            # 将负样本的 logits 拼接到原始 logits
            logits_sp = torch.cat([logits_sp, negative_logits_sp], dim=-1)
            logits = torch.cat([logits, negative_logits_cl], dim=-1)
            triplet_mask = torch.cat([triplet_mask, triplet_negative_mask], dim=-1)
        
        if self.pre_batch > 0 and self.training:
            # 如果适用，计算预批次的 logits
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)
        
        if self.args.use_self_negative and self.training:
            # 如果适用，处理自负样本
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            
            # 获取自负样本掩码
            self_negative_mask = batch_dict['self_negative_mask']
            
            # 在 self_negative_mask 为 False 的位置用一个大负值填充 self_neg_logits
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            
            # 将自负样本的 logits 拼接到原始 logits
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)
        
        # 返回包含计算出的 logits 和相关信息的字典
        return {
            'logits_sp': logits_sp,
            'logits': logits,
            'logits_hr_ori':logits_hr_ori,
            'logits_hp_2_t': logits_hp_2_t,
            'logits_hp_3_t': logits_hp_3_t,
            'labels': labels,
            'logits_hr_hp_2': logits_hr_hp_2,
            'logits_hr_hp_3': logits_hr_hp_3,
            'mask': ~triplet_mask if triplet_mask is not None else None,
            'inv_t': self.log_inv_t.detach().exp()
        }


    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode_ori(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids
                                   )
        return {'ent_vectors': ent_vectors.detach()}

    @torch.no_grad()
    def predict_head_embedding(self, hr_token_ids, hr_mask, hr_token_type_ids, **kwargs) -> dict:
        hr_vector = self._encode_ori(self.hr_bert,
                                    token_ids=hr_token_ids,
                                    mask=hr_mask,
                                    token_type_ids=hr_token_type_ids
                                    )
        return {'hr_vector': hr_vector}
    @torch.no_grad()
    def predict_hp_embedding(self, hrp_2_token_ids,hrp_2_mask,hrp_2_token_type_ids,
                            hrp_3_token_ids,hrp_3_mask,hrp_3_token_type_ids, **kwargs) -> dict:
        hp_2_vector = self._encode_ori(self.hr_bert,
                                    token_ids=hrp_2_token_ids,
                                    mask=hrp_2_mask,
                                    token_type_ids=hrp_2_token_type_ids
                                    )
        
        hp_3_vector = self._encode_ori(self.hr_bert,
                                    token_ids=hrp_3_token_ids,
                                    mask=hrp_3_mask,
                                    token_type_ids=hrp_3_token_type_ids
                                    )
        return {'hp_2_vector': hp_2_vector,
                'hp_3_vector': hp_3_vector}

def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    return output_vector
