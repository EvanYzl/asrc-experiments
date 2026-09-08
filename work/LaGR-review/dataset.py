import os.path as osp
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from functools import partial
from lightning import LightningDataModule


def _read_mapping_table(file_path):
    obj2id = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            relation, rid = line.strip().split('\t')
            obj2id[relation] = int(rid)
    return obj2id


def _read_triplets(file_path, entity2id, relation2id):
    triplets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            rev_r = relation2id['-' + r]
            h, r, t = entity2id[h], relation2id[r], entity2id[t]
            triplets.append((h, r, t))
            triplets.append((t, rev_r, h))
    return triplets


def _build_hr2t(triplets):
    hr2t = defaultdict(set)
    for h, r, t in triplets:
        hr2t[(h, r)].add(t)
    return hr2t


def _build_sorted_adjacent_indices(triplets):
    triplet_tensor = torch.LongTensor(triplets)
    edge_index, edge_type = triplet_tensor[:, [0, 2]].transpose(0, 1), triplet_tensor[:, 1]
    node_in, node_out = edge_index
    key = node_in * (node_out.max() + 1) + node_out
    order = key.argsort()
    return edge_index[:, order], edge_type[order]


class KnowledgeGraph(LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers=4, root='data', coarse_root='coarse'):
        super().__init__()
        self.dataset, self.batch_size, self.num_workers = dataset, batch_size, num_workers
        if dataset[-3: -1] == '_v':
            data_path = osp.join(root, 'inductive', dataset)
            is_inductive = True
        else:
            data_path = osp.join(root, dataset)
            is_inductive = False

        entity2id = _read_mapping_table(osp.join(data_path, 'entities.txt'))
        relation2id = _read_mapping_table(osp.join(data_path, 'relations.txt'))

        self.num_entities = len(entity2id)
        self.num_relations = len(relation2id)
        self.num_nodes = self.num_entities

        train = _read_triplets(osp.join(data_path, 'train.txt'), entity2id, relation2id)
        valid = _read_triplets(osp.join(data_path, 'valid.txt'), entity2id, relation2id)

        self.triplets = {'train': train, 'valid': valid}
        self.hr2t = {'train': _build_hr2t(train)}
        self.edge_index, self.edge_type = _build_sorted_adjacent_indices(train)

        if is_inductive:
            self.num_nodes = None
            self.hr2t['valid'] = _build_hr2t(train + valid)
            data_path = data_path + '_ind'
            entity2id = _read_mapping_table(osp.join(data_path, 'entities.txt'))
            train = _read_triplets(osp.join(data_path, 'train.txt'), entity2id, relation2id)
            valid = _read_triplets(osp.join(data_path, 'valid.txt'), entity2id, relation2id)
            test = _read_triplets(osp.join(data_path, 'test.txt'), entity2id, relation2id)
            self.edge_index_test, self.edge_type_test = _build_sorted_adjacent_indices(train)
            self.num_entities_test = len(entity2id)
            self.hr2t['test'] = _build_hr2t(train + valid + test)
        else:
            self.edge_index_test, self.edge_type_test = self.edge_index, self.edge_type
            self.num_entities_test = self.num_entities
            test = _read_triplets(osp.join(data_path, 'test.txt'), entity2id, relation2id)
            self.hr2t['test'] = self.hr2t['valid'] = _build_hr2t(train + valid + test)
        self.triplets['test'] = test
        train = []
        for h, r, t in self.triplets['train']:
            if r % 2 == 0:
                train.append((h, r, t))
        self.triplets['train'] = train

        self.coarse_indices = None
        coarse_path = osp.join(coarse_root, f'{dataset}.pt')
        if osp.exists(coarse_path):
            coarse_scores = torch.load(coarse_path, map_location='cpu', weights_only=False)
            self.coarse_indices = torch.argsort(coarse_scores, descending=True)

    def _collate_fn(self, indices, split):
        triplets, hr2t, batch_size, coarse = self.triplets[split], self.hr2t[split], len(indices), None
        edge_index, edge_type, num_nodes = self.edge_index, self.edge_type, self.num_entities

        if split == 'test':
            edge_index, edge_type, num_nodes = self.edge_index_test, self.edge_type_test, self.num_entities_test
            coarse = self.coarse_indices[indices] if self.coarse_indices is not None else None

        flip_index = batch_size // 2 if split == 'train' else batch_size
        data, mask = [], torch.ones(batch_size, num_nodes)
        for i, j in enumerate(indices):
            h, r, t = triplets[j]
            if i >= flip_index:
                h, r, t = t, r + 1, h
            data.append((h, r, t))
            mask[i, list(hr2t[(h, r)])] = 0

        return {
            'data': torch.LongTensor(data),
            'mask': mask,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'num_nodes': num_nodes,
            'coarse': coarse
        }

    def _dataloader(self, split):
        return DataLoader([i for i in range(len(self.triplets[split]))],
                          shuffle=split == 'train',
                          collate_fn=partial(self._collate_fn, split=split),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def train_dataloader(self):
        return self._dataloader('train')

    def val_dataloader(self):
        return self._dataloader('valid')

    def test_dataloader(self):
        return self._dataloader('test')
