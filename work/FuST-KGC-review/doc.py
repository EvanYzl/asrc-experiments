import os
import json
import torch
import torch.utils.data.dataset

from typing import Optional, List

from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from logger_config import logger

entity_dict = get_entity_dict()
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity



def get_second_order_neighbors(entity_id: str, first_order_neighbors: list) -> list:
    """
    获取一阶邻居的二阶邻居，并按度数排序。
    :param entity_id: 当前实体ID
    :param first_order_neighbors: 当前实体的一阶邻居ID列表
    :return: 按度数从高到低排序的二阶邻居列表
    """
    second_order_neighbors = []

    for neighbor_id in first_order_neighbors:
        # 获取邻居的二阶邻居
        neighbor_neighbors = get_link_graph().get_neighbor_ids(neighbor_id)

        # 获取邻居的度数
        degree = len(neighbor_neighbors)

        # 将邻居及其度数作为元组添加到列表中
        second_order_neighbors.append((neighbor_id, degree, neighbor_neighbors))

    # 按度数排序，从高到低
    second_order_neighbors = sorted(second_order_neighbors, key=lambda x: x[1], reverse=True)

    # 选择度数最多的两个二阶邻居
    selected_second_order_neighbors = []
    for neighbor_id, degree, neighbor_neighbors in second_order_neighbors:
        # 从二阶邻居中随机选择两个
        random.shuffle(neighbor_neighbors)
        selected_second_order_neighbors.extend(neighbor_neighbors[:3])  # 取每个邻居的前两个二阶邻居

    # 去重，确保没有重复的二阶邻居
    selected_second_order_neighbors = list(set(selected_second_order_neighbors))

    return selected_second_order_neighbors


import random


def get_neighbor_relations_and_entities(entity_id: str, tail_id: str = None) -> str:
    """
    对于给定的实体 entity_id：
    1) 随机采样 6 个一阶邻居；
    2) 从这 6 个一阶邻居里，挑选“度数”最高的 2 个；
    3) 对这 2 个一阶邻居，各采样 1 个它们的二阶邻居（总共 2 个）；
    4) 拼接成 r1 t1,r2 t2 的格式，用分号分隔不同的一阶邻居块；
    5) 将剩余的 4 个邻居（没有选择的）按原格式拼接。
    """
    # 1) 获取所有一阶邻居，并排除 tail_id（避免信息泄漏），然后随机采样 6 个
    all_neighbors = get_link_graph().get_neighbor_ids(entity_id)
    if tail_id and (not args.is_test):
        all_neighbors = [nid for nid in all_neighbors if nid != tail_id]

    if not all_neighbors:  # 没有邻居直接返回空
        return ""

    first_order_neighbors = random.sample(all_neighbors, min(6, len(all_neighbors)))

    # 2) 计算“度数”并从这 6 个里选出度数最高的 2 个
    #    “度数”就是该邻居本身的邻居数量
    neighbor_with_degs = []
    for nbr_id in first_order_neighbors:
        nbr_neighbors = get_link_graph().get_neighbor_ids(nbr_id)
        degree = len(nbr_neighbors)
        neighbor_with_degs.append((nbr_id, degree))

    # 按度数从大到小排序，取前 2 个
    neighbor_with_degs.sort(key=lambda x: x[1], reverse=True)
    top_first_order_neighbors = neighbor_with_degs[:3]  # 只要前 2 个

    # 3) 对这 2 个一阶邻居，各采样 1 个它们的二阶邻居
    used_second_order = set()  # 用于去重，防止二阶邻居重复
    results = []

    for nbr_id, _ in top_first_order_neighbors:
        # 先获取头实体到这个一阶邻居的关系；可能有多条关系，这里简单地取第一条
        relations_1 = get_link_graph().get_relations(entity_id, nbr_id)
        if not relations_1:
            continue
        r1 = relations_1[0]  # 取第一条或根据需要自行处理
        nbr_name = _parse_entity_name(entity_dict.get_entity_by_id(nbr_id).entity)

        # 在该一阶邻居的邻接里，再排除 entity_id、tail_id，防止回采到头实体或尾实体
        nbr_neighbors = get_link_graph().get_neighbor_ids(nbr_id)
        if entity_id in nbr_neighbors:
            nbr_neighbors.remove(entity_id)
        if tail_id and (tail_id in nbr_neighbors) and (not args.is_test):
            nbr_neighbors.remove(tail_id)

        # 打乱后采样 1 个作为二阶邻居
        random.shuffle(nbr_neighbors)
        second_order_id = None
        for candidate in nbr_neighbors:
            if candidate not in used_second_order:
                second_order_id = candidate
                used_second_order.add(candidate)
                break

        # 如果一个都没采到，说明没有可用的二阶邻居
        if not second_order_id:
            # 也可以根据需要跳过，或者只拼接一阶邻居
            results.append(f"{r1} {nbr_name}")
            continue

        # 获取一阶邻居到二阶邻居的关系
        relations_2 = get_link_graph().get_relations(nbr_id, second_order_id)
        if not relations_2:
            # 如果没有关系，就只记录一阶邻居
            results.append(f"{r1} {nbr_name}")
            continue
        r2 = relations_2[0]

        second_order_name = _parse_entity_name(entity_dict.get_entity_by_id(second_order_id).entity)

        # 拼接一阶邻居和二阶邻居
        e_name = _parse_entity_name(entity_dict.get_entity_by_id(entity_id).entity)
        combined_str = f"{e_name} {r1} {nbr_name},{nbr_name} {r2} {second_order_name}"
        results.append(combined_str)

    # 4) 拼接剩余的 4 个一阶邻居（没有被选择的）
    remaining_neighbors = [nbr_id for nbr_id, _ in neighbor_with_degs[3:]]
    for nbr_id in remaining_neighbors:
        relations_1 = get_link_graph().get_relations(entity_id, nbr_id)
        if not relations_1:
            continue
        r1 = relations_1[0]  # 取第一条或根据需要自行处理
        nbr_name = _parse_entity_name(entity_dict.get_entity_by_id(nbr_id).entity)
        e_name = _parse_entity_name(entity_dict.get_entity_by_id(entity_id).entity)
        results.append(f"{e_name} {r1} {nbr_name}")

    # 用分号分隔不同的一阶邻居块
    return "; ".join(results)



def _concat_head_and_desc_and_relations(head_word: str, head_desc: str, relations_and_tails: List[str]) -> str:
    """
    拼接头实体名称、描述和邻居实体的关系-尾实体对。
    """
    head_text = _concat_name_desc(head_word, head_desc)
    relations_tail_pairs = "; ".join(relations_and_tails)
    final_text = f"{head_text} {relations_tail_pairs}"
    return final_text


class Example:

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        """
        构造头实体、尾实体及其描述、关系的拼接，包括邻居关系和实体对。
        """
        head_desc, tail_desc = self.head_desc, self.tail_desc
        head_desc += ' ;' + get_neighbor_relations_and_entities(entity_id=self.head_id, tail_id=self.tail_id)
        tail_desc += '; ' + get_neighbor_relations_and_entities(entity_id=self.tail_id, tail_id=self.head_id)
        head_word = _parse_entity_name(self.head)
        relations_and_tails = [
            "{} {}".format(self.relation, _parse_entity_name(self.tail))
        ]

        # 拼接头实体、描述和邻居实体的关系-尾实体对
        head_text = _concat_name_desc(head_word, head_desc)

        hr_encoded_inputs = _custom_tokenize(text=head_text, text_pair=self.relation)
        # print(self.head,self.relation,self.tail)
        # print(self.head_desc)
        # print(head_text)
        # print("=============================")
        head_encoded_inputs = _custom_tokenize(text=head_text)
        tail_word = _parse_entity_name(self.tail)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

        return {
            'hr_token_ids': hr_encoded_inputs['input_ids'],
            'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
            'tail_token_ids': tail_encoded_inputs['input_ids'],
            'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
            'head_token_ids': head_encoded_inputs['input_ids'],
            'head_token_type_ids': head_encoded_inputs['token_type_ids'],
            'obj': self
        }


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)

    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])

    # Ensure the minimum length is 128
    mx_len = max(mx_len, 128)

    # Ensure the length doesn't exceed the maximum allowed length
    if args.max_num_tokens:
        mx_len = min(mx_len, args.max_num_tokens)

    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)

    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)

    for i, t in enumerate(batch_tensor):
        seq_len = len(t)

        # Handle the case if the sequence length is smaller than mx_len
        indices[i, :seq_len].copy_(t)
        if need_mask:
            mask[i, :seq_len].fill_(1)

        # Pad if sequence length is less than mx_len
        if seq_len < mx_len:
            indices[i, seq_len:mx_len].fill_(pad_token_id)
            if need_mask:
                mask[i, seq_len:mx_len].fill_(0)

    if need_mask:
        return indices, mask
    else:
        return indices

