import numpy as np
import csv
import os
import argparse
import numpy as np
import re
from sklearn.model_selection import train_test_split


def transform_pair(pair, dbp15k_dir, transform_dir, seed):
    """Function to transform a language pair dataset from dbp-15k

    Args:
        pair (str): 'zh_en' / 'fr_en' / 'ja_en'
        dbp15k_dir (str): path to the root directory of dbp-15k
        transform_dir (str): path of the output directory
        seed (int): random_state
    """

    print(f'Transforming {pair}')
    lang1, lang2 = pair.split('_')

    # Create Entity Lists
    ents_path = os.path.join(transform_dir, pair, 'entity_lists')
    if not os.path.exists(ents_path):
        os.makedirs(ents_path)

    ents1 = dict()
    id2id_map_1 = dict()
    lines = open(f'{dbp15k_dir}/{pair}/ent_ids_1',
                 encoding='utf8').read().strip().split('\n')
    for line in lines:
        i, ent = line.split('\t')
        i = int(i)
        assert i not in id2id_map_1
        id2id_map_1[i] = len(id2id_map_1)
        ents1[id2id_map_1[i]] = ent

    ents2 = dict()
    id2id_map_2 = dict()
    lines = open(f'{dbp15k_dir}/{pair}/ent_ids_2',
                 encoding='utf8').read().strip().split('\n')
    for line in lines:
        i, ent = line.split('\t')
        i = int(i)
        assert i not in id2id_map_2
        id2id_map_2[i] = len(id2id_map_2)
        ents2[id2id_map_2[i]] = ent

    open(os.path.join(ents_path, lang1 + '.tsv'), 'w',
         encoding='utf8').write('\n'.join(list(ents1.values())))
    open(os.path.join(ents_path, lang2 + '.tsv'), 'w',
         encoding='utf8').write('\n'.join(list(ents2.values())))

    # Transform triples
    kgs_path = os.path.join(transform_dir, pair, 'kgs')
    if not os.path.exists(kgs_path):
        os.makedirs(kgs_path)

    trips1 = []
    rels1 = set()
    lines = open(f'{dbp15k_dir}/{pair}/triples_1').read().strip().split('\n')
    for line in lines:
        sub, rel, obj = map(int, line.split('\t'))
        rels1.add(rel)
        trips1.append((sub, rel, obj))

    trips2 = []
    rels2 = set()
    lines = open(f'{dbp15k_dir}/{pair}/triples_2').read().strip().split('\n')
    for line in lines:
        sub, rel, obj = map(int, line.split('\t'))
        rels2.add(rel)
        trips2.append((sub, rel, obj))

    rels2relid_map = dict()
    i = 0  # relid counter
    rels_ILL = dict()
    lines = open(f'{dbp15k_dir}/{pair}/ref_r_ids').read().strip().split('\n')

    for line in lines:
        rel1, rel2 = map(int, line.split('\t')[:2])
        rels_ILL[rel2] = rel1

    for rel in rels1:
        assert rel not in rels_ILL
        rels2relid_map[rel] = i
        i += 1

    for rel in rels2:
        if rel in rels_ILL:
            rels2relid_map[rel] = rels2relid_map[rels_ILL[rel]]
        else:
            rels2relid_map[rel] = i
            i += 1

    triples1 = []
    for trip in trips1:
        s, r, o = trip
        triples1.append(
            '\t'.join(map(str, [id2id_map_1[s], rels2relid_map[r], id2id_map_1[o]])))

    triples2 = []
    for trip in trips2:
        s, r, o = trip
        triples2.append(
            '\t'.join(map(str, [id2id_map_2[s], rels2relid_map[r], id2id_map_2[o]])))

    def split(data):  # splits data in 60:30:10 ratio
        train, temp = train_test_split(data, test_size=0.4, random_state=seed)
        val, test = train_test_split(temp, test_size=0.25, random_state=seed)
        train, val, test = map("\n".join, [train, val, test])
        return train, val, test

    train1, val1, test1 = split(triples1)
    open(os.path.join(kgs_path, lang1 + '-train.tsv'),
         'w', encoding='utf8').write(train1)
    open(os.path.join(kgs_path, lang1 + '-val.tsv'),
         'w', encoding='utf8').write(val1)
    open(os.path.join(kgs_path, lang1 + '-test.tsv'),
         'w', encoding='utf8').write(test1)

    train2, val2, test2 = split(triples2)
    open(os.path.join(kgs_path, lang2 + '-train.tsv'),
         'w', encoding='utf8').write(train2)
    open(os.path.join(kgs_path, lang2 + '-val.tsv'),
         'w', encoding='utf8').write(val2)
    open(os.path.join(kgs_path, lang2 + '-test.tsv'),
         'w', encoding='utf8').write(test2)

    # Write Global Relations
    open(os.path.join(transform_dir, pair, 'relations.txt'), 'w',
         encoding='utf8').write('\n'.join(map(str, set(rels2relid_map.values()))))

    # Write Entity Alignments
    entILL_path = os.path.join(transform_dir, pair, 'seed_alignment')
    if not os.path.exists(entILL_path):
        os.makedirs(entILL_path)

    ents_ILL = []
    lines = open(f'{dbp15k_dir}/{pair}/ref_ent_ids').read().strip().split('\n')
    for line in lines:
        ent1, ent2 = map(int, line.split('\t'))
        ents_ILL.append(
            '\t'.join(map(str, [id2id_map_1[ent1], id2id_map_2[ent2]])))

    open(os.path.join(entILL_path, pair.replace('_', '-') + '.tsv'), 'w',
         encoding='utf8').write('\n'.join(ents_ILL))


def transform(args):
    dbp15k_dir = args.dbp15k
    transform_dir = args.outdir
    seed = args.seed

    if not os.path.exists(transform_dir):
        print('Creating output directory:', os.path.abspath(transform_dir))
        os.makedirs(transform_dir)

    pairs = [pair for pair in os.listdir(
        dbp15k_dir) if re.match(f"\w\w_\w\w", pair)]
    for pair in pairs:
        transform_pair(pair, dbp15k_dir, transform_dir, seed)

# download DBP15K from https://drive.google.com/file/d/1Ya1S-SXdCZkCjAlibf4NIr-1t0n8uB0-/view?usp=sharing
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbp15k", required=True, help='/path/to/dbp15k/dir')
    ap.add_argument("--outdir", required=True, help='/path/to/output/dir')
    ap.add_argument("--seed", required=False, type=int, default=123, help='random_state')

    av = ap.parse_args()
    transform(av)
    print(f'Transformation Complete! Data saved at {os.path.abspath(av.outdir)}')
