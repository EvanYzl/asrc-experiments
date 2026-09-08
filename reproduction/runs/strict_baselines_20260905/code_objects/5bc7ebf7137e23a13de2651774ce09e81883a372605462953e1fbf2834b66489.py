import argparse
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import sys
sys.path.append("..")
from importers.ea_ra_kgc import EaRaKgcData

def main(av):

    data_info = EaRaKgcData(av.dbp5l)
    combo_base = os.path.join(data_info.dir, "combined")
    combined_path = os.path.join(combo_base, "Combined_" + str(av.ea_percent)
                                 + "_" + str(av.ra_percent) + "/")
    map_path = os.path.join(combined_path, "mapping.txt")
    align_path_partial = os.path.join(data_info.dir, "seed_alignment_" +
                                      str(av.ea_percent) + "/")
    alignment_path = os.path.join(data_info.dir, "seed_alignment/")

    print("loading", av.model_path)
    model = torch.load(av.model_path)
    print("loaded", len(model.keys()), "keys from", av.model_path)

    entity_path = os.path.join(data_info.dir, "entity_lists/")
    entity_list = {lang: open(entity_path + lang + ".tsv").readlines()
                 for lang in data_info.langs}

    rel_path = os.path.join(data_info.dir, "relations.txt")
    rel = open(rel_path).readlines()

    train = {lang: open(os.path.join(data_info.dir, f"kgs/{lang}-train.tsv"))
             for lang in data_info.langs}

    given_alignment = {}
    used_alignment = {}

    mapping = {lang: {} for lang in data_info.langs}

    with open(map_path) as mapp:
        lines = mapp.readlines()
    for line in lines:
        a = line.split()
        mapping[a[2]][int(a[1])] = int(a[0])

    alignments = {"-".join(pair): open(alignment_path + "-".join(pair) + ".tsv")
                 for pair in data_info.lang_pairs}
    for pair in data_info.lang_pairs:
        lines = alignments["-".join(pair)].readlines()
        for line in lines:
            a = line.split()
            given_alignment.setdefault((pair[0], int(float(a[0]))), []).append((pair[1], int(float(a[1]))))
            given_alignment.setdefault((pair[1], int(float(a[1]))), []).append((pair[0], int(float(a[0]))))

    alignments_partial = {"-".join(pair): open(align_path_partial + "-".join(pair) + ".tsv")
                 for pair in data_info.lang_pairs}
    for pair in data_info.lang_pairs:
        lines = alignments_partial["-".join(pair)].readlines()
        for line in lines:
            a = line.split()
            used_alignment.setdefault((pair[0], int(float(a[0]))), []).append((pair[1], int(float(a[1]))))
            used_alignment.setdefault((pair[1], int(float(a[1]))), []).append((pair[0], int(float(a[0]))))

    ent_name = {lang: {} for lang in data_info.langs}

    with open(map_path) as mapp:
        lines = mapp.readlines()
    for line in lines:
        a = line.split()
        ent_name[a[2]][int(a[0])] = entity_list[a[2]][int(a[1])]

    entity = dict()
    lang_rels = dict()
    for lang in data_info.langs:
        lines = train[lang].readlines()
        lang_rels[lang] = set()
        for line in lines:
            a = line.split()
            lang_rels[lang].add(int(a[1]))
            entity.setdefault(int(a[1]), []).append((entity_list[lang][int(a[0])], entity_list[lang][int(a[2])]))
        train[lang].close()

    count = {}
    for i in range(len(rel)):
        count[rel[i]] = 0
        for lang in data_info.langs:
            if i in lang_rels[lang]:
                count[rel[i]] += 1

    rel_map = model['relation_map']
    rel_embedd_real = model['model_weights']['R_re.weight']
    rel_embedd_im = model['model_weights']['R_im.weight']
    rel_embedd = torch.cat((rel_embedd_real, rel_embedd_im), 1)

    ent_map = model['entity_map']
    ent_embedd_real = model['model_weights']['E_re.weight']
    ent_embedd_im = model['model_weights']['E_im.weight']
    ent_embedd = torch.cat((ent_embedd_real, ent_embedd_im), 1)
    ent_embedd = ent_embedd.detach().cpu().numpy()

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    rel_embedd = rel_embedd.detach().cpu().numpy()
    mat = cosine_similarity(rel_embedd, rel_embedd)


    def f(map_, idx):
        for i in map_:
            if i == '<OOV>':
                return len(map_)
            if map_[i] == idx:
                return int(i)


    e_map = {ent_map[i]: i for i in ent_map}

    def get_label(num):
        if num == 1:
            return 0
        if num <= 10:
            return 1
        if num <= 50:
            return 2
        if num <= 100:
            return 3
        if num <= 500:
            return 4
        if num <= 1000:
            return 5
        return 6


    print("RA evaluation:")
    hits_1 = [0, 0, 0, 0, 0, 0, 0]
    hits_3 = [0, 0, 0, 0, 0, 0, 0]
    c = [0, 0, 0, 0, 0, 0, 0]
    for i in tqdm(range(len(rel_embedd))):
        idx = f(rel_map, i)
        if int(str(idx)[0]) == '6':
            continue
        rel_sc = [(mat[i][j], f(rel_map, j)) for j in range(len(mat[i]))]
        rel_sc.sort(reverse=True)
        if int(str(idx)[1:]) in entity:
            x = len(entity[int(str(idx)[1:])])
        else:
            x = 0
            continue
        num_lang = count[rel[int(str(idx)[1:])]]
        label_id = get_label(x)
        c[label_id] = c[label_id] + (num_lang - 1)
        a = 0
        for score, j in rel_sc:
            if int(str(j)[1:]) == int(str(idx)[1:]):
                if idx != j:
                    if a == 0:
                        hits_1[label_id] = hits_1[label_id] + 1
                    if a < 3:
                        hits_3[label_id] = hits_3[label_id] + 1
            else:
                a = a + 1
                if a == 3:  # MAGIC
                    break
    print("Hits@1", hits_1)
    print("Hits@3", hits_3)
    print("Total", c)
    print((hits_1[1] + hits_1[2] + hits_1[3] + hits_1[4]) / (
                c[1] + c[2] + c[3] + c[4]),
          (hits_3[1] + hits_3[2] + hits_3[3] + hits_3[4]) / (
                      c[1] + c[2] + c[3] + c[4]))
    print((hits_1[5] + hits_1[6]) / (c[5] + c[6]),
          (hits_3[5] + hits_3[6]) / (c[5] + c[6]))

    mat = cosine_similarity(ent_embedd, ent_embedd)

    max_id = len(mat) - 1
    print(max_id)
    hits1_ = {}
    hits10_ = {}
    c_ = {}

    print("\nEA evaluation:")
    n = 0
    for lang1, id1 in tqdm(given_alignment):
        n = n + 1
        for lang2, id2 in given_alignment[(lang1, id1)]:
            if (lang1, id1) in used_alignment and \
                    (lang2, id2) in used_alignment[(lang1, id1)]:
                continue
            if id1 in mapping[lang1] and id2 in mapping[lang2]:
                idx1 = mapping[lang1][id1]
                idx2 = mapping[lang2][id2]
                if idx1 < max_id and idx2 < max_id:
                    d1 = ent_map[str(idx1)]
                    d2 = ent_map[str(idx2)]
                    if idx1 == idx2:
                        continue
                    if (lang1, lang2) not in c_:
                        c_[(lang1, lang2)] = 0
                    c_[(lang1, lang2)] = c_[(lang1, lang2)] + 1
                    ent_sc = [(mat[d1][i], e_map[i]) for i in range(len(mat[d1]))]
                    ent_sc.sort(reverse=True)
                    j = 0
                    p = []
                    u = 0
                    for sc, i in ent_sc:
                        if i == '<OOV>' or int(i) not in ent_name[lang2]:
                            continue
                        p.append((sc, ent_name[lang2][int(i)]))
                        if j <= 1 and i != '<OOV>' and int(i) == idx2:
                            if (lang1, lang2) not in hits1_:
                                hits1_[(lang1, lang2)] = 0
                            hits1_[(lang1, lang2)] = hits1_[(lang1, lang2)] + 1
                        if i != '<OOV>' and int(i) == idx2:
                            u = u + 1
                            if (lang1, lang2) not in hits10_:
                                hits10_[(lang1, lang2)] = 0
                            hits10_[(lang1, lang2)] = hits10_[(lang1, lang2)] + 1
                        j = j + 1
                        if j > 10:
                            break
    print(hits1_, hits10_, c_)

    hits1 = 0
    hits10 = 0
    c = 0
    for l1, l2 in c_:
        hits1 = hits1 + hits1_[(l1, l2)]
        c = c + c_[(l1, l2)]
    for l1, l2 in c_:
        hits10 = hits10 + hits10_[(l1, l2)]
        print((l1, l2), hits1_[(l1, l2)] / c_[(l1, l2)],
              hits10_[(l1, l2)] / c_[(l1, l2)])
    print(hits1, hits10, c, hits1 / c, hits10 / c)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbp5l", required=True, help="/path/to/DBP-5L/")
    ap.add_argument("--ea_percent", required=True, type=int)
    ap.add_argument("--ra_percent", required=True, type=int)
    ap.add_argument("--model_path", required=True, help="/path/to/model.pt")
    av = ap.parse_args()
    print(av)
    main(av)