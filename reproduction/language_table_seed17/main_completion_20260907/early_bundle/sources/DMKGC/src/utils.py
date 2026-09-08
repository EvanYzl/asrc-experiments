"""Graph construction, model persistence, and evaluation utilities."""

import copy
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphLoader
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm


class MyData(Data):
    """PyG Data subclass: edge_kg_index is excluded from batch dimension inference."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_kg_index':
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def save_model(model, output_dir, filename, args):
    """Save model weights and training arguments."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, filename)
    torch.save({'state_dict': model.state_dict(), 'args': args}, ckpt_path)


def get_negative_samples_graph(batch_size_each, num_entity):
    return torch.randint(high=num_entity, size=(batch_size_each,))


def ranking_all_batch(predicted_t, embedding_matrix, k=None, define_score=None, d_r=None):
    """Rank candidate entities in batch and return top-k indices and scores."""
    if (predicted_t.ndim == 3 and predicted_t.size(1) == 1
            and embedding_matrix.ndim == 3 and embedding_matrix.size(0) == 1):
        # DMKGC's fixed decoder is Euclidean TransE.  cdist produces the same
        # [batch, candidates] distances without a broadcast 3-D difference.
        distance = torch.cdist(predicted_t[:, 0, :], embedding_matrix[0, :, :], p=2)
    elif define_score is not None:
        distance = define_score([predicted_t, embedding_matrix])
    else:
        distance = torch.norm(predicted_t - embedding_matrix, dim=2)

    if k is None:
        k = embedding_matrix.size(1)

    top_k_scores, top_k_indexes = torch.topk(-distance, k=k)
    return top_k_indexes, top_k_scores


def get_language_list(entity_dir):
    entity_files = sorted(os.listdir(entity_dir))
    return [e[:2] for e in entity_files]


def get_kg_edges_for_each(kg_dir, language):
    """Load train/val edges for one KG and build an undirected edge_index."""
    train_df = pd.read_csv(
        os.path.join(kg_dir, language + '-train.tsv'),
        sep='\t', header=None, names=['head', 'relation', 'tail'],
    )
    val_df = pd.read_csv(
        os.path.join(kg_dir, language + '-val.tsv'),
        sep='\t', header=None, names=['head', 'relation', 'tail'],
    )

    sender_node_list = train_df['head'].values.astype(int).tolist()
    sender_node_list += train_df['tail'].values.astype(int).tolist()
    receiver_node_list = train_df['tail'].values.astype(int).tolist()
    receiver_node_list += train_df['head'].values.astype(int).tolist()
    edge_relation_list = train_df['relation'].values.astype(int).tolist()
    edge_relation_list += train_df['relation'].values.astype(int).tolist()

    val_sender = val_df['head'].values.astype(int).tolist() + val_df['tail'].values.astype(int).tolist()
    val_receiver = val_df['tail'].values.astype(int).tolist() + val_df['head'].values.astype(int).tolist()
    val_relation = val_df['relation'].values.astype(int).tolist() + val_df['relation'].values.astype(int).tolist()

    sender_node_list += val_sender
    receiver_node_list += val_receiver
    edge_relation_list += val_relation

    edge_index = torch.LongTensor(np.vstack((sender_node_list, receiver_node_list)))
    edge_relation = torch.LongTensor(np.asarray(edge_relation_list))
    return edge_index, edge_relation


def get_all_edges(kg_dir, kg_objects_dict, all_entity_global_index):
    """Merge edges from all domain KGs and map entity IDs to the global space."""
    edge_index_list = []
    edge_relation_list = []

    def get_global(x, language):
        return all_entity_global_index[language][x]

    for language in kg_objects_dict:
        edge_index, edge_relation = get_kg_edges_for_each(kg_dir, language)
        edge_index_list.append(edge_index.apply_(lambda x: get_global(x, language)))
        edge_relation_list.append(edge_relation)

    return torch.cat(edge_index_list, dim=1), torch.cat(edge_relation_list, dim=0)


def create_subgraph_list(edge_index, edge_type, total_num_nodes, num_hops, k):
    """Build a k-hop subgraph per global entity with at most ``k`` edges.

    The upstream implementation calls :func:`k_hop_subgraph` once per entity.
    That function materializes several tensors the size of the complete edge
    set on every call.  On Windows/PyTorch 2.x those temporaries accumulate in
    the allocator and exhaust the commit limit.  The CSR traversal below is
    exactly equivalent for the default ``source_to_target, directed=False``
    semantics: it expands incoming neighbors for ``num_hops``, takes the
    induced edges in their original order, and uses sorted node relabeling.
    """
    subgraph_list = []
    num_edges = []

    source = edge_index[0].cpu().numpy()
    target = edge_index[1].cpu().numpy()
    relation = edge_type.cpu().numpy()
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
        # Nodes outside the retained ``k`` edges are isolated after truncation
        # and cannot affect the queried center in a message-passing GNN.  Drop
        # them so every cached graph has at most ``2 * k + 1`` nodes.
        used_nodes = np.unique(np.concatenate((
            np.asarray([i], dtype=np.int64), endpoints.cpu().numpy().reshape(-1),
        )))
        relabeled = np.searchsorted(used_nodes, endpoints.cpu().numpy())
        edge_index_each = torch.from_numpy(relabeled).long()
        edge_attr = torch.from_numpy(relation[candidate_edges]).long()
        subgraph_node_ids = torch.from_numpy(used_nodes).long()
        node_position = int(np.searchsorted(used_nodes, i))
        assert edge_attr.shape[0] == edge_index_each.shape[1]

        subgraph_each = Data(
            x=subgraph_node_ids,
            edge_index=edge_index_each,
            edge_attr=edge_attr,
            y=torch.LongTensor([node_position]),
            num_size=torch.LongTensor([len(subgraph_node_ids)]),
        )
        subgraph_list.append(subgraph_each)
        num_edges.append(edge_index_each.shape[1])

    print('Average subgraph edges %.2f' % np.mean(num_edges))
    return subgraph_list


def get_k_subgraph_list(subgraph_list, node_index, kg_index, num_kgs, entity2kgidx, total_num_nodes, data_dir):
    """Build and cache compact domain-view subgraphs for one KG.

    Upstream stores hundreds of thousands of tiny ``Data`` Python objects.
    Their object and pickle overhead exceeds the local Windows commit limit.
    ``packed_subgraphs_v1`` stores the same fields in padded integer tensors;
    :func:`nodes_to_k_graph` materializes only the current mini-batch.
    """
    k_subgraph_list_path = os.path.join(data_dir, f'kg{kg_index}_k_subgraph_list.graph')
    if os.path.exists(k_subgraph_list_path):
        return torch.load(k_subgraph_list_path)

    print('get_k_subgraph_list: kg_index:', kg_index)

    max_edges = max((graph.edge_index.shape[1] for graph in subgraph_list), default=0)
    max_nodes = 2 * max_edges + 1
    num_local_entities = len(node_index)
    packed = {
        'format': 'packed_subgraphs_v1',
        'x': torch.full((num_local_entities, num_kgs, max_nodes), -1, dtype=torch.int32),
        'edge_index': torch.full((num_local_entities, num_kgs, 2, max_edges), -1, dtype=torch.int16),
        'edge_kg_index': torch.full((num_local_entities, num_kgs, 2, max_edges), -1, dtype=torch.int8),
        'edge_attr': torch.full((num_local_entities, num_kgs, max_edges), -1, dtype=torch.int32),
        'num_nodes': torch.zeros((num_local_entities, num_kgs), dtype=torch.int16),
        'num_edges': torch.zeros((num_local_entities, num_kgs), dtype=torch.int16),
        'y': torch.zeros((num_local_entities, num_kgs), dtype=torch.int16),
    }

    for local_entity_index, i_tensor in enumerate(tqdm(node_index)):
        i = int(i_tensor)
        subgraph = subgraph_list[i]
        nodes = subgraph.x
        edge_index = subgraph.edge_index
        edge_attr = subgraph.edge_attr

        for j in range(num_kgs):
            eligible = torch.tensor([
                (node.item() == i) or (j in entity2kgidx[node.item()])
                for node in nodes
            ], dtype=torch.bool)
            edge_mask = eligible[edge_index[0]] & eligible[edge_index[1]]
            edge_global = nodes[edge_index[:, edge_mask]]
            subset_j = torch.unique(torch.cat((torch.tensor([i]), edge_global.reshape(-1))))
            inv = torch.searchsorted(subset_j, torch.tensor(i)).view(1)
            edge_index_j = torch.searchsorted(subset_j, edge_global)
            edge_attr_j = edge_attr[edge_mask]
            edge_kg_index = torch.where(
                edge_global == i,
                torch.full_like(edge_global, kg_index),
                torch.full_like(edge_global, j),
            )

            n_nodes = subset_j.numel()
            n_edges = edge_attr_j.numel()
            packed['x'][local_entity_index, j, :n_nodes] = subset_j.to(torch.int32)
            packed['edge_index'][local_entity_index, j, :, :n_edges] = edge_index_j.to(torch.int16)
            packed['edge_kg_index'][local_entity_index, j, :, :n_edges] = edge_kg_index.to(torch.int8)
            packed['edge_attr'][local_entity_index, j, :n_edges] = edge_attr_j.to(torch.int32)
            packed['num_nodes'][local_entity_index, j] = n_nodes
            packed['num_edges'][local_entity_index, j] = n_edges
            packed['y'][local_entity_index, j] = inv.item()

    torch.save(packed, k_subgraph_list_path)
    return packed


def nodes_to_k_graph(k_subgraph_list, node_index, device, shuffle=False):
    """Pack per-KG subgraphs for entities in a batch into PyG batches."""
    batch_size = node_index.shape[0]
    graph_batches = []

    if isinstance(k_subgraph_list, dict) and k_subgraph_list.get('format') == 'packed_subgraphs_v1':
        rows = node_index.detach().to(device='cpu', dtype=torch.long).reshape(-1)
        num_kgs = k_subgraph_list['x'].shape[1]
        for kg_view in range(num_kgs):
            num_nodes = k_subgraph_list['num_nodes'][rows, kg_view].long()
            num_edges = k_subgraph_list['num_edges'][rows, kg_view].long()
            node_mask = torch.arange(k_subgraph_list['x'].shape[2]).unsqueeze(0) < num_nodes.unsqueeze(1)
            x = k_subgraph_list['x'][rows, kg_view].long()[node_mask]
            edge_mask = torch.arange(k_subgraph_list['edge_attr'].shape[2]).unsqueeze(0) < num_edges.unsqueeze(1)
            flat_edge_mask = edge_mask.reshape(-1)
            node_offsets = torch.cat((
                torch.zeros(1, dtype=torch.long),
                torch.cumsum(num_nodes, dim=0)[:-1],
            ))
            edge_index_padded = k_subgraph_list['edge_index'][rows, kg_view].long() + node_offsets[:, None, None]
            edge_index = edge_index_padded.permute(1, 0, 2).reshape(2, -1)[:, flat_edge_mask]
            edge_kg_padded = k_subgraph_list['edge_kg_index'][rows, kg_view].long()
            edge_kg_index = edge_kg_padded.permute(1, 0, 2).reshape(2, -1)[:, flat_edge_mask]
            edge_attr = k_subgraph_list['edge_attr'][rows, kg_view].long()[edge_mask]
            y = k_subgraph_list['y'][rows, kg_view].long().reshape(-1)
            graph_batches.append(MyData(
                x=x,
                edge_index=edge_index,
                edge_kg_index=edge_kg_index,
                edge_attr=edge_attr,
                y=y,
                num_size=num_nodes,
            ).to(device))
        return graph_batches

    for i in range(len(k_subgraph_list[0])):
        graphs = [k_subgraph_list[j.item()][i] for j in node_index]
        graph_loader = GraphLoader(graphs, batch_size=batch_size, shuffle=shuffle)
        for batch in graph_loader:
            graph_batches.append(batch.to(device))

    return graph_batches


def get_ent_id(graph_input):
    """Extract global IDs of center entities from a batched subgraph."""
    y = graph_input.y.reshape(-1)
    sizes = graph_input.num_size.reshape(-1)
    node_base = torch.cat((
        torch.zeros(1, dtype=sizes.dtype, device=sizes.device),
        torch.cumsum(sizes, dim=0)[:-1],
    ))
    return graph_input.x[y + node_base].reshape(-1)
