import os
import json
from typing import List
from dataclasses import dataclass
from collections import deque
from logger_config import logger


@dataclass
class EntityExample:
    entity_id: str
    entity: str
    entity_desc: str = ''


class TripletDict:

    def __init__(self, path_list: List[str]):
        self.path_list = path_list
        logger.info('Triplets path: {}'.format(self.path_list))
        self.relations = set()
        self.hr2tails = {}
        self.triplet_cnt = 0

        for path in self.path_list:
            self._load(path)
        logger.info('Triplet statistics: {} relations, {} triplets'.format(len(self.relations), self.triplet_cnt))

    def _load(self, path: str):
        examples = json.load(open(path, 'r', encoding='utf-8'))
        examples += [reverse_triplet(obj) for obj in examples]  # 加载反向三元组
        for ex in examples:
            self.relations.add(ex['relation'])
            key = (ex['head_id'], ex['relation'])
            if key not in self.hr2tails:
                self.hr2tails[key] = set()
            self.hr2tails[key].add(ex['tail_id'])
        self.triplet_cnt = len(examples)

    def get_neighbors(self, h: str, r: str) -> set:
        return self.hr2tails.get((h, r), set())


class EntityDict:

    def __init__(self, entity_dict_dir: str, inductive_test_path: str = None):
        path = os.path.join(entity_dict_dir, 'entities.json')
        assert os.path.exists(path)
        self.entity_exs = [EntityExample(**obj) for obj in json.load(open(path, 'r', encoding='utf-8'))]

        if inductive_test_path:
            examples = json.load(open(inductive_test_path, 'r', encoding='utf-8'))
            valid_entity_ids = set()
            for ex in examples:
                valid_entity_ids.add(ex['head_id'])
                valid_entity_ids.add(ex['tail_id'])
            self.entity_exs = [ex for ex in self.entity_exs if ex.entity_id in valid_entity_ids]

        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}
        logger.info('Load {} entities from {}'.format(len(self.id2entity), path))

    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]

    def get_entity_by_id(self, entity_id: str) -> EntityExample:
        return self.id2entity[entity_id]

    def get_entity_by_idx(self, idx: int) -> EntityExample:
        return self.entity_exs[idx]

    def __len__(self):
        return len(self.entity_exs)


class LinkGraph:

    def __init__(self, train_path: str):
        logger.info('Start to build link graph from {}'.format(train_path))
        # id -> {neighbor_id: set(relationships)}
        self.graph = {}
        examples = json.load(open(train_path, 'r', encoding='utf-8'))

        # 假设 `ex['relation']` 包含头实体和尾实体之间的关系
        for ex in examples:
            head_id, tail_id, relation = ex['head_id'], ex['tail_id'], ex['relation']

            # 初始化head_id和tail_id在图中的数据结构
            if head_id not in self.graph:
                self.graph[head_id] = {}
            if tail_id not in self.graph:
                self.graph[tail_id] = {}

            # 更新head_id与tail_id之间的关系（有向边）
            if tail_id not in self.graph[head_id]:
                self.graph[head_id][tail_id] = set()
            self.graph[head_id][tail_id].add(relation)

            # **添加反向边：tail_id -> head_id**
            if head_id not in self.graph[tail_id]:
                self.graph[tail_id][head_id] = set()
            reverse_relation = self.reverse_relation(relation)  # 获取反向关系
            self.graph[tail_id][head_id].add(reverse_relation)

        logger.info('Done build link graph with {} nodes'.format(len(self.graph)))

    def reverse_relation(self, relation: str) -> str:
        """
        反向关系映射：根据关系的语义，为每个关系定义反向关系。
        例如，"_member_meronym" 的反向关系可以是 "has_member_meronym"。
        """
        reverse_relation_map = {
            'hypernym': 'has hypernym',
            'derivationally related form': 'has derivationally related form',
            'instance hypernym': 'has instance hypernym',
            'also see': 'also seen as',
            'member meronym': 'has member meronym',
            'synset domain topic of': 'has synset domain topic',
            'has part': 'is part of',
            'member of domain usage': 'has member of domain usage',
            'member of domain region': 'has member of domain_region',
            'verb group': 'is verb group of',
            'similar to': 'is similar to',
        }
        return reverse_relation_map.get(relation, f'inverse {relation}')  # 默认将 "inverse" 加到关系前

    def get_neighbor_ids(self, entity_id: str, max_to_keep=20) -> List[str]:
        """
        获取与指定实体的邻居实体（最多 `max_to_keep` 个）。
        注意：这里返回的是实体出发的邻居（即从实体指向的邻居）
        """
        neighbor_ids = self.graph.get(entity_id, {})
        return sorted(list(neighbor_ids.keys()))[:max_to_keep]

    def get_relations(self, entity_id: str, neighbor_id: str) -> List[str]:
        """
        获取指定实体与邻居实体之间的所有关系。
        """
        if entity_id in self.graph and neighbor_id in self.graph[entity_id]:
            return list(self.graph[entity_id][neighbor_id])
        return []

    def has_relation(self, head_id: str, tail_id: str) -> bool:
        """
        检查从head_id到tail_id是否存在关系。
        """
        return tail_id in self.graph.get(head_id, {})

    def get_n_hop_entity_indices(self, entity_id: str,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 max_nodes: int = 100000) -> set:
        """
        获取n跳内的所有实体的索引。
        """
        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, {}):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])


def reverse_triplet(obj):
    return {
        'head_id': obj['tail_id'],
        'head': obj['tail'],
        'relation': 'inverse {}'.format(obj['relation']),
        'tail_id': obj['head_id'],
        'tail': obj['head']
    }
