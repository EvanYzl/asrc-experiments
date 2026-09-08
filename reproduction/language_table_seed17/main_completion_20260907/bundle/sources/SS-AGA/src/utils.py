import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from os.path import join
import torch
from tqdm import tqdm
from src.graph_sampler import k_hop_subgraph
from torch_geometric.data import DataLoader as Loader
import os


class PackedSubgraphList:
    """Memory-compact, index-compatible storage for PyG ``Data`` graphs.

    SS-AGA keeps two graph views for every entity.  A Python ``Data`` object
    per view dominates Windows commit although the integer tensors themselves
    are small.  This container stores the exact fields in contiguous tensors
    and reconstructs a lightweight view only when a batch requests it.

    Self-learning may later replace individual graphs after adding an
    alignment edge.  Those sparse changes are retained in ``overrides`` so the
    public list-style read/write behavior remains intact.
    """

    def __init__(self, graphs):
        self.node_offsets = torch.empty(len(graphs) + 1, dtype=torch.long)
        self.edge_offsets = torch.empty(len(graphs) + 1, dtype=torch.long)
        self.node_offsets[0] = 0
        self.edge_offsets[0] = 0
        for index, graph in enumerate(graphs):
            self.node_offsets[index + 1] = (
                self.node_offsets[index] + graph.x.numel()
            )
            self.edge_offsets[index + 1] = (
                self.edge_offsets[index] + graph.edge_attr.numel()
            )

        total_nodes = int(self.node_offsets[-1])
        total_edges = int(self.edge_offsets[-1])
        self.x = torch.empty(total_nodes, dtype=torch.long)
        self.edge_index = torch.empty((2, total_edges), dtype=torch.long)
        self.edge_attr = torch.empty(total_edges, dtype=torch.long)
        self.y = torch.empty(len(graphs), dtype=torch.long)
        self.num_size = torch.empty(len(graphs), dtype=torch.long)
        for index, graph in enumerate(graphs):
            node_start = int(self.node_offsets[index])
            node_stop = int(self.node_offsets[index + 1])
            edge_start = int(self.edge_offsets[index])
            edge_stop = int(self.edge_offsets[index + 1])
            self.x[node_start:node_stop].copy_(graph.x.reshape(-1))
            self.edge_index[:, edge_start:edge_stop].copy_(graph.edge_index)
            self.edge_attr[edge_start:edge_stop].copy_(graph.edge_attr.reshape(-1))
            self.y[index] = graph.y.reshape(-1)[0]
            self.num_size[index] = graph.num_size.reshape(-1)[0]
        self.overrides = {}

    def __len__(self):
        return self.y.numel()

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index in self.overrides:
            return self.overrides[index]
        node_start = int(self.node_offsets[index])
        node_stop = int(self.node_offsets[index + 1])
        edge_start = int(self.edge_offsets[index])
        edge_stop = int(self.edge_offsets[index + 1])
        return Data(
            x=self.x[node_start:node_stop],
            edge_index=self.edge_index[:, edge_start:edge_stop],
            edge_attr=self.edge_attr[edge_start:edge_stop],
            y=self.y[index:index + 1],
            num_size=self.num_size[index:index + 1],
        )

    def __setitem__(self, index, graph):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        self.overrides[index] = graph


def pack_subgraph_list(graphs):
    if isinstance(graphs, PackedSubgraphList):
        return graphs
    packed = PackedSubgraphList(graphs)
    if len(graphs):
        sample_ids = np.unique(np.linspace(
            0, len(graphs) - 1, num=min(8, len(graphs)), dtype=np.int64
        ))
        for index in sample_ids:
            reference = graphs[int(index)]
            observed = packed[int(index)]
            for field in ("x", "edge_index", "edge_attr", "y", "num_size"):
                if not torch.equal(getattr(reference, field), getattr(observed, field)):
                    raise RuntimeError(
                        f"packed SS-AGA graph mismatch at {index}:{field}"
                    )
    return packed


def save_model(model, output_dir, filename, args):
    """
    Save the trained knowledge model under output_dir. Filename: 'language.h5'
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save the weights for the whole model
    ckpt_path = os.path.join(output_dir, filename)
    torch.save({
        'state_dict': model.state_dict(),
        'args': args,
    }, ckpt_path)

def get_negative_samples_alignment(batch_size_each, num_entity,num_negative=None):
    '''
    Generate one negative sample
    :param batch_size_each:
    :param num_entity:
    :return:
    '''
    if num_negative == None:
        rand_negs = torch.randint(high=num_entity, size=(batch_size_each,))  # [b,n]
    else:
        rand_negs = torch.randint(high=num_entity, size=(batch_size_each,num_negative))  # [b,n]

    return rand_negs


def load_model(ckpt_path, model, device):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    # Load checkpoint.
    checkpt = torch.load(ckpt_path)
    ckpt_args = checkpt['args']
    state_dict = checkpt['state_dict']
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # 3. load the new state dict
    model.load_state_dict(state_dict)
    model.to(device)


def get_negative_samples_graph(batch_size_each, num_entity):
    '''
    Generate one negative samaple
    :param batch_size_each:
    :param num_entity:
    :return:
    '''
    rand_negs = torch.randint(high=num_entity, size=(batch_size_each,))  # [b,1]

    return rand_negs



def Ranking_all_batch(predicted_t, embedding_matrix, k = None):
    '''
    Compute k-nearest neighbors in a batch
    If k== None, return ranked all candidatees
    otherwise, return top_k candidates
    :param predicted_t:
    :param embedding_matrix:
    :param k:
    :return:
    '''

    total_entity = embedding_matrix.shape[0]
    # ``torch.cdist`` computes the same exhaustive Euclidean distances without
    # materializing the repository's [batch, entity, dim] repeated tensor.
    distance = torch.cdist(predicted_t, embedding_matrix, p=2)  # [b,n]

    if k==None:
        k = total_entity

    top_k_scores, top_k_t = torch.topk(-distance, k=k)
    return top_k_t, top_k_scores




def get_language_list(data_dir):
    entity_dir = data_dir + "/entity"
    entity_files = list(os.listdir(entity_dir))
    entity_files = list(filter(lambda x: x[-3:] == "tsv", entity_files))
    entity_files = sorted(entity_files)
    print("Number of KGs is %d" % len(entity_files))

    kg_names = []
    for each_entity_file in entity_files:
        kg_name_each = each_entity_file[:2]
        kg_names.append(kg_name_each)

    return kg_names


def nodes_to_graph(sub_graph_list, node_index, batch_size=-1):
    one_batch = False
    if batch_size == -1:
        # get embeddings together without batch
        batch_size = node_index.shape[0]
        one_batch = True

    graphs = [sub_graph_list[i.item()] for i in node_index]

    graph_loader = Loader(graphs, batch_size=batch_size, shuffle=False)

    if one_batch:
        for one_batch in graph_loader:
            assert one_batch.edge_index.shape[1] == one_batch.edge_attr.shape[0]
            return one_batch
    else:
        return graph_loader


def nodes_to_graph_align(sub_graph_list, node_index, batch_size=-1):
    # node_index : [b,n_neg]
    one_batch = False
    if batch_size == -1:
        # get embeddings together without batch
        batch_size = node_index.shape[0]
        one_batch = True

    # Reshape into [b*n_neg,1]
    node_index = torch.reshape(node_index,(-1,))

    graphs = [sub_graph_list[i.item()] for i in node_index]

    graph_loader = Loader(graphs, batch_size=batch_size, shuffle=False)

    if one_batch:
        for one_batch in graph_loader:
            assert one_batch.edge_index.shape[1] == one_batch.edge_attr.shape[0]
            return one_batch
    else:
        return graph_loader



def create_subgraph_list(language, edge_index, edge_value, total_num_nodes, num_hops, k, node_base, relation_base):
    # Adding self-loop for nodes without edges
    # TODO: here k is for restricting the total number of edges in a subgrap. Wether to remove?
    # TODO padding self=edges for those do not have edges
    sub_graph_list = []
    num_edges = []

    source = edge_index[0].cpu().numpy()
    target = edge_index[1].cpu().numpy()
    relation = edge_value.cpu().numpy()
    incoming_order = np.argsort(target, kind='stable')
    incoming_counts = np.bincount(target, minlength=total_num_nodes)
    incoming_offsets = np.empty(total_num_nodes + 1, dtype=np.int64)
    incoming_offsets[0] = 0
    np.cumsum(incoming_counts, out=incoming_offsets[1:])

    def incoming_edges(nodes):
        chunks = [incoming_order[incoming_offsets[n]:incoming_offsets[n + 1]] for n in nodes]
        chunks = [chunk for chunk in chunks if chunk.size]
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    membership = np.zeros(total_num_nodes, dtype=np.bool_)
    for i in tqdm(range(total_num_nodes)):
        reached_parts = [np.asarray([i], dtype=np.int64)]
        frontier = reached_parts[0]
        for _ in range(num_hops):
            frontier_edges = incoming_edges(frontier)
            frontier = source[frontier_edges]
            reached_parts.append(frontier)

        subset = np.unique(np.concatenate(reached_parts))
        membership[subset] = True
        candidate_edges = incoming_edges(subset)
        candidate_edges = candidate_edges[membership[source[candidate_edges]]]
        candidate_edges = np.sort(candidate_edges)[:k]
        membership[subset] = False

        endpoints = edge_index[:, torch.from_numpy(candidate_edges)]
        used_nodes = np.unique(np.concatenate((
            np.asarray([i], dtype=np.int64), endpoints.cpu().numpy().reshape(-1),
        )))
        edge_index_each = torch.from_numpy(
            np.searchsorted(used_nodes, endpoints.cpu().numpy())
        ).long()
        x = torch.from_numpy(used_nodes).long() + node_base
        edge_attr = torch.from_numpy(relation[candidate_edges]).long() + relation_base
        node_position = int(np.searchsorted(used_nodes, i))

        assert edge_attr.shape[0] == edge_index_each.shape[1]

        node_position = torch.LongTensor([node_position])
        num_size = torch.LongTensor([len(used_nodes)])
        graph_each = Data(x=x, edge_index=edge_index_each, edge_attr=edge_attr, y=node_position, num_size=num_size)
        sub_graph_list.append(graph_each)
        num_edges.append(edge_index_each.shape[1])

    print(language + ":Average subgraph edges %.2f" % np.mean(num_edges))


    return sub_graph_list




def get_kg_edges_for_each(data_dir, language, is_target_KG=False):
    '''
    TODO: whether include directional edges (1. do not incorporate. 2. adding the numbe of relation embeddings)
    :param data_dir:
    :param language:
    :param is_target_KG:
    :return:
    '''
    train_df = pd.read_csv(join(data_dir, language + '-train.tsv'), sep='\t', header=None,
                           names=['v1', 'relation', 'v2'])

    val_df = pd.read_csv(join(data_dir, language + '-val.tsv'), sep='\t', header=None,
                         names=['v1', 'relation', 'v2'])

    # Training data graph construction
    sender_node_list = train_df['v1'].values.astype(np.int64).tolist()
    sender_node_list += train_df['v2'].values.astype(np.int64).tolist()

    receiver_node_list = train_df['v2'].values.astype(np.int64).tolist()
    receiver_node_list += train_df['v1'].values.astype(np.int64).tolist()

    edge_weight_list = train_df['relation'].values.astype(np.int64).tolist() + train_df['relation'].values.astype(
        np.int64).tolist()

    # unified: Adding validation edges from supporter KG as well
    if not is_target_KG:
        sender_node_list += val_df['v1'].values.astype(np.int64).tolist()
        sender_node_list += val_df['v2'].values.astype(np.int64).tolist()

        receiver_node_list += val_df['v2'].values.astype(np.int64).tolist()
        receiver_node_list += val_df['v1'].values.astype(np.int64).tolist()

        edge_weight_list += val_df['relation'].values.astype(np.int64).tolist()
        edge_weight_list += val_df['relation'].values.astype(np.int64).tolist()

    edge_index = torch.LongTensor(np.vstack((sender_node_list, receiver_node_list)))
    edge_weight = torch.LongTensor(np.asarray(edge_weight_list))
    return edge_index, edge_weight


def get_subgraph_list(data_dir, language, is_target_KG, num_entity, num_hop, k, node_base, relation_base):
    cache_dir = os.path.join(data_dir, 'ssaga_cache')
    os.makedirs(cache_dir, exist_ok=True)
    role = 'target' if is_target_KG else 'support'
    cache_path = os.path.join(cache_dir, f'{language}_{role}_h{num_hop}_k{k}.graph')
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    edge_index, edge_type = get_kg_edges_for_each(data_dir + "/kg", language, is_target_KG=is_target_KG)

    sub_graph_list = create_subgraph_list(language, edge_index, edge_type, num_entity, num_hop, k, node_base, relation_base)
    torch.save(sub_graph_list, cache_path)
    return sub_graph_list



def subgrarph_list_from_alignment(seed_pairs,kg0,kg1,is_kg_list = False):
    '''
    lang0, lang1 alignment pairs. np.int

    Append nodes from other kgs to x
    :param seed_pairs:
    :param lang0:
    :param lang1:
    :return:
    '''
    # TOod: check them!

    num_relation = kg0.num_relation  # Total number of relations in relation.txt + 1
    if is_kg_list:
        for (entity0, entity1) in seed_pairs:
            graph0 = kg0.subgraph_list_kg[entity0]
            graph1 = kg1.subgraph_list_kg[entity1]

            graph0.x = torch.cat([graph0.x, torch.LongTensor([entity1 + kg1.entity_id_base])])  # global index
            graph1.x = torch.cat([graph1.x, torch.LongTensor([entity0 + kg0.entity_id_base])])  # global index

            # undirected edges
            graph0.edge_index = torch.cat([graph0.edge_index,torch.LongTensor([[graph0.num_size,graph0.y], [graph0.y,graph0.num_size]])], dim=1)
            graph1.edge_index = torch.cat([graph1.edge_index,torch.LongTensor([[graph1.num_size,graph1.y], [graph1.y,graph1.num_size]])], dim=1)

            graph0.edge_attr = torch.cat([graph0.edge_attr,torch.LongTensor([num_relation + kg0.relation_id_base - 1,num_relation + kg0.relation_id_base - 1])])  # global index
            graph1.edge_attr = torch.cat([graph1.edge_attr,torch.LongTensor([num_relation + kg1.relation_id_base - 1,num_relation + kg1.relation_id_base - 1])])  # global index

            graph0.num_size = graph0.num_size + 1
            graph1.num_size = graph1.num_size + 1

            kg0.subgraph_list_kg[entity0] = graph0
            kg1.subgraph_list_kg[entity1] = graph1

    else:
        for (entity0, entity1) in seed_pairs:
            graph0 = kg0.subgraph_list_align[entity0]
            graph1 = kg1.subgraph_list_align[entity1]

            graph0.x = torch.cat([graph0.x, torch.LongTensor([entity1 + kg1.entity_id_base])])  # global index
            graph1.x = torch.cat([graph1.x, torch.LongTensor([entity0 + kg0.entity_id_base])])  # global index

            # undirected edges
            graph0.edge_index = torch.cat([graph0.edge_index, torch.LongTensor([[graph0.num_size, graph0.y], [graph0.y, graph0.num_size]])],dim=1)
            graph1.edge_index = torch.cat([graph1.edge_index, torch.LongTensor([[graph1.num_size, graph1.y], [graph1.y, graph1.num_size]])],dim=1)

            graph0.edge_attr = torch.cat([graph0.edge_attr, torch.LongTensor([num_relation + kg0.relation_id_base - 1, num_relation + kg0.relation_id_base - 1])])  # global index
            graph1.edge_attr = torch.cat([graph1.edge_attr, torch.LongTensor([num_relation + kg1.relation_id_base - 1, num_relation + kg1.relation_id_base - 1])])  # global index
            graph0.num_size = graph0.num_size + 1
            graph1.num_size = graph1.num_size + 1

            kg0.subgraph_list_align[entity0] = graph0
            kg1.subgraph_list_align[entity1] = graph1





