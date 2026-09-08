"""
Samples seed alignments and multilingual relations
from original DBP5L files.
"""

import argparse
import glob
import numpy as np
import os
import re
from importers.ea_ra_kgc import EaRaKgcData


def sample_ent_equiv(ea_in_path, ea_percent, ea_out_path):
    with open(ea_in_path) as infile, open(ea_out_path, "w") as outfile:
        inlines = infile.readlines()
        print("read", len(inlines), "lines from", ea_in_path)
        nout = round(ea_percent * len(inlines) / 100.0)
        outlines = np.random.choice(inlines, size=nout, replace=False)
        outfile.write("\n".join([x.strip() for x in outlines]))
        print("wrote", len(outlines), "lines to", ea_out_path)


def sample_multiling_rels(av):
    meta = EaRaKgcData(av.inpath)

    relpath = os.path.join(av.inpath, "relations.txt")
    with open(relpath) as relfile:
        rels = relfile.readlines()
    relid_to_langs = dict()
    lx = 0
    for _ in rels:
        relid_to_langs[lx] = list()
        lx += 1
    print("initialized", lx, "relations from", relpath)

    for inpath in glob.glob(os.path.join(av.inpath, "kgs/*-train.tsv")):
        inname = os.path.basename(inpath)
        inlang_match = re.match("(.*)-train.tsv", inname)
        if not inlang_match:
            continue
        inlang = inlang_match.group(1)
        with open(inpath) as infile:
            triples = infile.readlines()
            for triple in triples:
                sro = triple.strip().split()
                relid = int(sro[1])
                if inlang not in relid_to_langs[relid]:
                    relid_to_langs[relid].append(inlang)
        print("accumulated", inpath)

    # collect relids with at least two languages
    multiling_relids = list()
    for relid in relid_to_langs:
        if len(relid_to_langs[relid]) > 1:
            multiling_relids.append(relid)
    print(len(multiling_relids), "of", len(relid_to_langs),
          "relations are multilingual")
    # sample down
    nrels = round(av.rel_percent * len(multiling_relids) / 100.0)
    outrelids = np.random.choice(multiling_relids, size=nrels, replace=False)
    outrelpath = meta.ra_path(av.rel_percent)
    with open(outrelpath, "w") as relfile:
        relfile.write("\n".join([str(x) for x in outrelids]))
    print("sampled", len(outrelids), "rels to", outrelpath)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inpath", required=True, help="/path/to/original/dbp5l/")
    ap.add_argument("--seed", type=int, default=41)
    av = ap.parse_args()
    np.random.seed(av.seed)

    for ea_percent in [20, 40, 50, 60, 80]:
        av.ea_percent = ea_percent
        outdir = os.path.join(av.inpath, "seed_alignment_"+str(av.ea_percent))
        if not os.path.isdir(outdir):
            print("creating", outdir)
            os.mkdir(outdir, mode=0o755)
        for ea_in_path in glob.glob(os.path.join(av.inpath, "seed_alignment/*.tsv")):
            ea_in_name = os.path.basename(ea_in_path)
            ea_out_path = os.path.join(outdir, ea_in_name)
            sample_ent_equiv(ea_in_path, av.ea_percent, ea_out_path)

    for rel_percent in [0, 20, 40, 60, 80]:
        av.rel_percent = rel_percent
        sample_multiling_rels(av)
