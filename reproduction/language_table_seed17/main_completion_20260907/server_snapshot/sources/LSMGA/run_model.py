#!/usr/bin/env python
# coding: utf-8


import os
import sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0') # Respect scheduler GPU isolation.
from os.path import join
print('Current working dir', os.getcwd())

import numpy as np
import torch
from src.data_loader import MKGDataset
from src.validate import Tester
from src.utils import nodes_to_k_graph, get_k_subgraph_list, get_language_list, get_negative_samples_graph, save_model
from src.lsmga_model import LSMGA

import logging
import argparse
import random
from random import SystemRandom
from tqdm import tqdm



def set_logger(model_dir, args):
    '''
    Write logs to checkpoint and console
    '''
    experimentID = int(SystemRandom().random() * 100000)
    log_file = model_dir + "/" + '_'.join(args.langs) + "_train_" + str(experimentID) + ".log"
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return experimentID


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models'
    )

    # Data loader related
    parser.add_argument('--remove_language', type=str, default='', help="remove kg")
    parser.add_argument('--k', default=10, type=int, help="how many nominations to consider")
    parser.add_argument('--num_hops', default=2, type=int, help="hop sampling")
    parser.add_argument('--data_path', default="dataset", type=str, help="data path")
    parser.add_argument('--dataset', default="dbp5l", type=str, choices=['dbp5l', 'depkg'], help="dataset")

    #KG model related
    parser.add_argument('--margin', default=0.3, type=float)
    parser.add_argument('--dim', default=256, type=int, help = 'kg embedding dimension')

    # GNN related
    parser.add_argument('--n_layers_gnn', default=2, type=int,help="GNN layer")
    parser.add_argument('--encoder_hdim_gnn', default=256, type=int, help='dimension of GNN')
    parser.add_argument('--n_heads', default=1, type=int, help="heads of attention")

    # Training Related
    parser.add_argument('--epoch_each', default=3, type=int, help="epochs for each KG")
    parser.add_argument('--round', default=50, type=int,help="rounds to train")
    parser.add_argument('--lr', '--learning_rate', default=5e-3, type=float, help="learning ratel")
    parser.add_argument('--batch_size', default=200, type=int, help="batch size for training")
    parser.add_argument('--micro_batch_size', default=0, type=int,
                        help="training micro-batch; 0 keeps the full batch")
    parser.add_argument('--optimizer', type=str, default="Adam", help='Adam, AdamW')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Others
    parser.add_argument('--device', default='cuda:0', type=str, help="which device to use")
    parser.add_argument('--test_batch_size', default=200, type=int, help="batch size for testing")
    parser.add_argument('--precompute_batch_size', default=0, type=int,
                        help='entity-embedding batch size; 0 follows test_batch_size')
    parser.add_argument('--MAX_SAM', default=10000000000, type=int,
                        help="subset of data for smoke-test debugging; default keeps the full dataset")
    parser.add_argument('--seed', default=2020, type=int, help='random seed')
    parser.add_argument('--resume_checkpoint', default='', type=str,
                        help='atomic full-state checkpoint used to resume completed rounds')

    return parser.parse_args(args)


def train_kg_batch(args, kg, kg_index, optimizer, num_epoch, model):
    kg_dataloader = kg.generate_batch_data(kg.h_train, kg.r_train, kg.t_train, batch_size=args.batch_size, shuffle=True)
    for one_epoch in range(num_epoch):
        kg_loss_sum = torch.zeros((), device=args.device)
        kg_batch_count = 0
        for kg_each in tqdm(kg_dataloader, disable=not sys.stderr.isatty()):
            # kg_each: [batch_size, 3]
            batch_size = kg_each.shape[0]
            micro_batch_size = args.micro_batch_size if args.micro_batch_size > 0 else batch_size
            if micro_batch_size < batch_size:
                raise ValueError('The released broadcast loss requires a full logical batch; splitting it removes cross-example comparisons')
            t_neg_index = get_negative_samples_graph(batch_size, kg.num_entities)
            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=args.device)
            for start in range(0, batch_size, micro_batch_size):
                stop = min(start + micro_batch_size, batch_size)
                entity_indices = torch.cat((
                    kg_each[start:stop, 0],
                    kg_each[start:stop, 2],
                    t_neg_index[start:stop],
                ), dim=0)
                entity_graph_batch_list = nodes_to_k_graph(
                    kg.k_subgraph_list, entity_indices, args.device
                )
                kg_micro_batch = kg_each[start:stop].to(args.device)
                loss = model.forward_kg_combined(
                    entity_graph_batch_list, kg_micro_batch, kg_index
                )
                weight = float(stop - start) / float(batch_size)
                (loss * weight).backward()
                batch_loss += loss.detach() * weight
                del (loss, entity_indices, entity_graph_batch_list,
                     kg_micro_batch)
            optimizer.step()
            kg_loss_sum += batch_loss
            kg_batch_count += 1
            del t_neg_index

        logging.info('KG {:s} Epoch {:d} [Train KG Loss {:.6f}|'.format(
            kg.lang,
            one_epoch,
            (kg_loss_sum / kg_batch_count).item()))


def save_resume_checkpoint(path, model, optimizer, completed_round, all_langs):
    """Atomically persist every state needed for an exact round-boundary resume."""
    if not path:
        return
    checkpoint_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    temporary_path = checkpoint_path + '.tmp'
    state = {
        'format': 'lsmga_full_state_v1',
        'completed_round': completed_round,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'all_langs': list(all_langs),
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def load_resume_checkpoint(path, model, optimizer, all_langs, device):
    """Restore a checkpoint and return the first round still to execute."""
    if not path or not os.path.exists(path):
        return 0, all_langs
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('format') != 'lsmga_full_state_v1':
        raise ValueError(f'unsupported LSMGA resume checkpoint: {path}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    random.setstate(checkpoint['python_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
    if torch.cuda.is_available() and checkpoint.get('cuda_rng_state_all') is not None:
        torch.cuda.set_rng_state_all([
            state.cpu() for state in checkpoint['cuda_rng_state_all']
        ])
    restored_langs = list(checkpoint['all_langs'])
    if set(restored_langs) != set(all_langs):
        raise ValueError('resume checkpoint language set does not match current dataset')
    return int(checkpoint['completed_round']) + 1, restored_langs



def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device(args.device)


    args.entity_dim = args.dim
    args.relation_dim = args.entity_dim



    remove_lang = args.remove_language
    all_langs = get_language_list(args.data_path + args.dataset + '/entity')

    if args.dataset == 'dbp5l':
        all_langs = ['el', 'en', 'es', 'fr', 'ja'] # dbp5l
    elif args.dataset == 'depkg':
        all_langs = ['de', 'es', 'fr', 'it', 'jp', 'uk'] # depkg
    if remove_lang:
        all_langs.remove(remove_lang)
    print(f"Number of KGs is {len(all_langs)}")
    args.langs = all_langs
    kgname2idx = {}
    for i in range(len(all_langs)):
        kgname2idx[all_langs[i]] = i

    
    model_dir = join('./' + args.dataset + "/trained_model", '_'.join(all_langs))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    experimentID = set_logger(model_dir, args)  # set logger
    logging.info('logger setting finished')
    
    # load data
    dataset = MKGDataset(args)
    kg_objects_dict, subgraph_list, all_entity2kgs, all_entity_global_index = dataset.load_data()
    logging.info('subgraph_list loaded')

    all_entity2kgidx = {}
    for global_id, langs in all_entity2kgs.items():
        langidx = set([kgname2idx[l] for l in langs])
        all_entity2kgidx[global_id] = langidx


    graph_dir = '_'.join(args.langs)
    for lang in kg_objects_dict.keys():
        kg_lang = kg_objects_dict[lang]
        kg_index = kgname2idx[lang]
        node_index = kg_lang.entity_global_index
        k_subgraph_list = get_k_subgraph_list(subgraph_list, node_index, kg_index, dataset.num_kgs, all_entity2kgidx, dataset.num_entities, os.path.join(dataset.data_dir, graph_dir))
        logging.info('kg' + str(kg_index) + '_k_subgraph_list loaded')
        kg_lang.k_subgraph_list = k_subgraph_list
    args.num_entities = dataset.num_entities
    args.num_relations = dataset.num_relations
    args.num_kgs = dataset.num_kgs

    args.kgname2idx = kgname2idx

    del subgraph_list

    if args.dataset == 'dbp5l':
        args.lr = 0.005
        args.margin = 0.5
    elif args.dataset == 'depkg':
        args.lr = 0.001
        args.margin = 0.3

    args.entity_dim = args.dim
    args.relation_dim = args.entity_dim

    # logging
    logging.info('remove language: %s' % (remove_lang))
    logging.info(f'languages: {args.langs}')
    logging.info(f'device: {args.device}')
    logging.info(f'batch_size: {args.batch_size}')
    logging.info(f'k: {args.k}')
    logging.info(f'num_hops: {args.num_hops}')
    logging.info(f'lr: {args.lr}')
    logging.info(f'margin: {args.margin}')
    logging.info(f'dim: {args.dim}')
    logging.info(f'experimentID: {experimentID}')


    # Build Model
    model = LSMGA(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_round, all_langs = load_resume_checkpoint(
        args.resume_checkpoint, model, optimizer, all_langs, args.device
    )
    

    logging.info('model initialization done')

    validator = Tester(args, kg_objects_dict, model, args.device, args.data_path + args.dataset)
    

    if start_round:
        logging.info(f'resumed from completed round {start_round - 1}')
    for i in range(start_round, args.round):
        logging.info(f'Round: {i}')

        model.train()


        # 
        random.shuffle(all_langs)
        for kg_name in all_langs:
            kg = kg_objects_dict[kg_name]
            kg_index = kgname2idx[kg_name]
            train_kg_batch(args, kg, kg_index, optimizer, args.epoch_each, model)

            logging.info(f'=== experimentID {experimentID} round {i}')

        model.eval()
        with torch.no_grad():
            # Parameters stay fixed across validation, raw test and filtered
            # test.  Compute each language's entity table once for this trio.
            for kg in kg_objects_dict.values():
                kg.computed_entity_embedding_kg = None
            metrics_val = validator.test(is_val=True)  # validation set
            metrics_test1 = validator.test(is_val=False, is_filtered=False)  # Test set
            metrics_test2 = validator.test(is_val=False, is_filtered=True)  # Test set

            for kg in kg_objects_dict.values():
                kg.computed_entity_embedding_kg = None
            torch.cuda.empty_cache()

            filename = "experiment_" + str(experimentID) + "_epoch_" + str(i) + '.ckpt'
            save_model(model, model_dir, filename, args)
            save_resume_checkpoint(
                args.resume_checkpoint, model, optimizer, i, all_langs
            )

if __name__ == "__main__":
    main(parse_args())

