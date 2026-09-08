import os
import errno
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_PATH = "../data"

def prepare_dataset(path, name):
    files = ['train', 'valid', 'test']

    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            items = line.split(' .')[0].split()
            if len(items) != 3:
                continue
            lhs, rel, rhs = items
        to_read.close()

    entities_to_id = dict()
    with open(f'{path}/entity2id.txt') as f:
        for line in f.readlines():
            entitie,id = line.strip().split()
            entities_to_id[entitie] = int(id)

    relations_to_id = dict()
    with open(f'{path}/relation2id.txt') as f:
        for line in f.readlines():
            relation,id = line.strip().split()
            relations_to_id[relation] = int(id)

    print("{} entities and {} relations".format(len(entities_to_id), len(relations_to_id)))
    n_relations = len(relations_to_id)
    n_entities = len(entities_to_id)
    # os.makedirs(os.path.join(DATA_PATH, name))
    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()
    for f in files:
        file_path = os.path.join(path, f)
        print("Processing file {}".format(file_path))
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            items = line.split(' .')[0].split()
            if len(items) != 3:
                continue
            lhs, rel, rhs = items
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs in examples:
            to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()


if __name__ == "__main__":
#    datasets = ['WN18RR', 'FB237', 'YAGO3-10', 'DB15K','MKG-W','MKG-Y', 'conceptnet-100k']
    datasets = ['DB15K','MKG-W','MKG-Y']#,'FB15K237']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset(
                os.path.join(
                    '../datasets', d
                ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise