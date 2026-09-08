import argparse
import csv
import os
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

from importers.ea_ra_kgc import EaRaKgcData


class Graph:
    def __init__(self):
        self.V = []
        self.adj = {}

    def num_nodes(self):
        return len(self.V)

    def num_edges(self):
        return sum([len(alst) for alst in self.adj.values()])

    def DFSUtil(self, temp, v, visited):
        visited[v] = True
        temp.append(v)
        for i in self.adj[v]:
            if i not in visited or visited[i] == False:
                temp = self.DFSUtil(temp, i, visited)
        return temp

    def addNode(self, v):
        if v not in self.V:
            self.V.append(v)
        if v not in self.adj:
            self.adj[v]=[]

    def addEdge(self, v, w):
        self.addNode(v)
        self.addNode(w)
        self.adj[v].append(w)
        self.adj[w].append(v)

    def connectedComponents(self):
        visited = {}
        cc = []
        for i in self.V:
            visited[i]=False
        for v in self.V:
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc


def collect_train_kg(meta, lang, kgraph, ents):
    """
    Reads train kg in lang, augments kgraph with lang-prefixed node
    ID,  collects all lang-specific ent IDs, and all relation IDs.
    """
    kg_path = os.path.join(meta.dir, "kgs")
    with open(os.path.join(kg_path, lang+"-train.tsv")) as lkgtripf:
        num_triples = 0
        for trip in tqdm(csv.reader(lkgtripf, delimiter='\t'),
                         desc=lang):
            num_triples += 1
            sub, rel, obj = int(trip[0]), int(trip[1]), int(trip[2])
            if sub not in ents:
                ents.append(sub)
            if obj not in ents:
                ents.append(obj)
            kgraph.addNode((lang, sub))
            kgraph.addNode((lang, obj))
        print("read", num_triples, "triples from", lang,
              "|V|=", kgraph.num_nodes(),
              "|E|=", kgraph.num_edges(),
              "|ents|=", len(ents))


def collect_ea(meta: EaRaKgcData, ea_percent, langA, langB, kgraph):
    """Scan entity alignment file of langA, langB, and add
    equivalence edges to kgraph."""
    alignment_path = meta.ea_path(langA, langB, ea_percent)
    with open(alignment_path) as align_file:
        for equiv in tqdm(csv.reader(align_file, delimiter='\t'),
                         desc=langA+"-"+langB):
            entA, entB = int(float(equiv[0])), int(float(equiv[1]))
            kgraph.addEdge((langA, entA), (langB, entB))


def get_geid(lang_ent: Tuple[str,int],
             en_to_id: Dict[Tuple[str,int], int]):
    """ Removed unused val arg """
    if lang_ent in en_to_id:
        return en_to_id[lang_ent]
    else:
        en_to_id[lang_ent] = len(en_to_id)
        return en_to_id[lang_ent]


def xfer_triples_global_ids(meta : EaRaKgcData, lang: str,
                            en_to_id : Dict[Tuple[str, int], int],
                            aligned_rels : List[int], inf, outf):
    """
    Read (isub, irel, iobj) triples in lang from inf.
    Map isub and iobj to global ent IDs.
    Consult aligned_rels to prefix irel with the proper lang code
    to form orel.  Write out triples to outf.
    """
    lid_prefix = meta.lang_to_lid(lang)  # lang-specific prefix
    uid_prefix = meta.maxlid_plus_one()  # shared prefix for this rel
    assert uid_prefix < 10, "cannot support >8 langs"
    for (isid, irid, ioid) in tqdm(meta.read_tsv_int_file(inf), desc=lang):
        osid = get_geid((lang, isid), en_to_id)
        ooid = get_geid((lang, ioid), en_to_id)
        if irid in aligned_rels:
            orid = meta.lang_rel_do_prefix(uid_prefix, irid)
        else:
            orid = meta.lang_rel_do_prefix(lid_prefix, irid)
        outf.write("{}\t{}\t{}\n".format(osid,orid,ooid))


def main(av):
    if not os.path.isdir(av.combined):
        print("creating", av.combined)
        os.mkdir(av.combined, mode=0o755)

    meta =  EaRaKgcData(av.dbp5l)

    alignment_rela: List[int] = list()
    """global relids of rels that will be exposed as aligned"""
    with open(meta.ra_path(av.ra_percent)) as ra_file:
        for rid_str in ra_file:  # without any prefix
            alignment_rela.append(int(rid_str))
    print("|alignment_rela|=", len(alignment_rela))

    dbp5l_kgpath = os.path.join(av.dbp5l, "kgs")

    comb_path = os.path.join(av.combined, "Combined_" +
                             str(av.ea_percent) + "_" +
                             str(av.ra_percent))
    if not os.path.isdir(comb_path):
        print("creating", comb_path)
        os.mkdir(comb_path, mode=0o755)

    entity_graph=Graph()
    lang_to_ents = dict()
    for lang in meta.langs:
        lang_to_ents[lang] = list()
        collect_train_kg(meta, lang, entity_graph, lang_to_ents[lang])
        lang_to_ents[lang].sort()        

    for (langA, langB) in meta.lang_pairs:
        collect_ea(meta, av.ea_percent,
                          langA, langB, entity_graph)

    print("|V|=", entity_graph.num_nodes(),
          "|E|=", entity_graph.num_edges())

    # Find equivalence classes of entities across all languages
    # given the pairwise equivalences, and assign new IDs to
    # the coalesced entity group.

    cc=entity_graph.connectedComponents()
    
    en_to_id: Dict[Tuple[str, int], int] =dict()
    """ key = (lang, lang_ent_id), val = global_id """
    idx: int = 0
    for entgrp in cc:
        for lang_ent in entgrp:
            en_to_id[lang_ent] = idx
        idx += 1  # next connected component
    print("|en_to_id|=", len(en_to_id), "idx=", idx)

    # Write out per-lang filter entity files.
    for lang in meta.langs:
        lang_ents = lang_to_ents[lang]
        with open(os.path.join(comb_path, "filters_" + lang + ".txt"),
                  "w") as ra_lang_filter:
            for ent in lang_ents:
                ra_lang_filter.write(str(get_geid((lang,ent),en_to_id))+'\n')

    # Write out combined KGC train triple file.
    # It uses global ent and rel ID spaces.
    ra_train=open(comb_path+"/train.txt",'w')
    for lang in meta.langs:
        inpath = dbp5l_kgpath + "/" + lang + "-train.tsv"
        print("xfer_triples_global_ids", inpath)
        with open(inpath) as lang_train_f:
            xfer_triples_global_ids(meta, lang, en_to_id,
                                    alignment_rela,
                                    lang_train_f, ra_train)
    ra_train.close()

    # Write out combined KGC dev=valid triple file with global IDs.
    ra_valid=open(comb_path+"/valid.txt",'w')
    for lang in meta.langs:
        inpath = dbp5l_kgpath + "/" + lang + "-val.tsv"
        print("xfer_triples_global_ids", inpath)
        with open(inpath) as lang_valid_f:
            xfer_triples_global_ids(meta, lang, en_to_id,
                                    alignment_rela,
                                    lang_valid_f, ra_valid)
    ra_valid.close()

    # Write out per-lang KGC test triple files but with global IDs.
    for lang in meta.langs:
        inp = os.path.join(dbp5l_kgpath, lang+"-test.tsv")
        outp = os.path.join(comb_path, "test_" + lang + ".txt")
        with open(inp) as inf, open(outp, "w") as outf:
            xfer_triples_global_ids(meta, lang, en_to_id,
                                    alignment_rela,
                                    inf, outf)

    # Write out mapping between combined and per-lang ent IDs.
    ra_mapping = open(os.path.join(comb_path, "mapping.txt"), 'w')
    for lang,ent in en_to_id:    
        ra_mapping.write(str(en_to_id[(lang,ent)])+"\t"+
                         str(ent)+"\t"+lang+"\n")
    ra_mapping.close()

    combo_train_triples: Set[Tuple[int]]=set()  # Earlier called "triple_map"
    """ Uses global ent IDs; removes lang prefix from rel, 
    so rel is global too. """
    combined_train_path = os.path.join(comb_path, "train.txt")
    for (isid, irid, ioid) in tqdm(meta.read_tsv_int_path(combined_train_path),
                                   desc="combined_train"):
        orid = meta.lang_rel_un_prefix(irid)
        combo_train_triples.add((isid, orid, ioid))
    print("|combo_train_triples|=", len(combo_train_triples))

    for lang in meta.langs:
        combined_LL_test = open(os.path.join(comb_path,
                                             "test_" + lang + ".txt"))
        """Read per-lang test triples with global IDs. """
        ra_filtered_LL_test = open(os.path.join(comb_path, lang +
                                                "_f_test.txt"), 'w')
        ra_overlap_LL_test = open(os.path.join(comb_path, lang +
                                               "_o_test.txt"),'w')
        m, n = 0, 0
        for (isid, irid, ioid) in tqdm(meta.read_tsv_int_file(combined_LL_test),
                                       desc="combined_" + lang + "_test"):
            m += 1
            orid = meta.lang_rel_un_prefix(irid)
            if (isid, orid, ioid) in combo_train_triples:
                ra_overlap_LL_test.write("{}\t{}\t{}\n".format(isid,irid,ioid))
                n += 1
            else:
                ra_filtered_LL_test.write("{}\t{}\t{}\n".format(isid,irid,ioid))
        print(lang, "{:.2f}".format(100.0*n/m), "percent overlap")
        ra_overlap_LL_test.close()
        ra_filtered_LL_test.close()
        combined_LL_test.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbp5l", required=True,
                    help="/path/to/dbp5l/")
    ap.add_argument("--combined", required=False,
                    help="[deprecated] /path/to/combined/")
    ap.add_argument("--ea_percent", type=int, required=False,
                    help="EA percent; if not provided, will sweep a range")
    ap.add_argument("--ra_percent", type=int, required=False,
                    help="RA percent; if not provided, will sweep a range")
    av = ap.parse_args()
    if av.ea_percent:
        ea_percents = [av.ea_percent]
    else:
        ea_percents = [20, 40, 50, 60, 80]
    if av.ra_percent:
        ra_percents = [av.ra_percent]
    else:
        ra_percents = [0, 20, 40, 60, 80]
    if not av.combined:
        av.combined = os.path.join(av.dbp5l, "combined/")

    for ea_percent in ea_percents:
        for ra_percent in ra_percents:
            av.ea_percent = int(ea_percent)
            av.ra_percent = int(ra_percent)
            main(av)
