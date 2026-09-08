"""Lossless file-format port for graph baselines; separate source relation IDs."""
from pathlib import Path
import json
import shutil
import numpy as np
from frozen_data import ROOT,sha256,atomic_json

def prepare_wk3l(profile):
    dest=ROOT/'reproduction/strict_baselines/graph_data'/profile/'datasetwk3l'
    for sub in ['entity','kg','seed_alignlinks']:(dest/sub).mkdir(parents=True,exist_ok=True)
    files={};nfr=None;relation_names=[]
    for kg,folder_name in [('fr','WK3l-15k_FR'),('en','WK3l-15k_EN_F')]:
        folder=ROOT/'data/raw/atransn'/folder_name
        ent=folder/'entity_dict.txt';rel=folder/'relation_dict.txt'
        lines=rel.read_text(encoding='utf-8').splitlines()
        if kg=='fr':nfr=len(lines)
        offset=0 if kg=='fr' else nfr
        relation_names.extend(f'{kg}:{x}' for x in lines)
        shutil.copy2(ent,dest/'entity'/f'{kg}.tsv')
        for p in [ent,rel]:files[p.relative_to(ROOT).as_posix()]=sha256(p)
        for original,new in [('train','train'),('valid','val'),('test','test')]:
            p=folder/f'{original}_triple_id.txt';files[p.relative_to(ROOT).as_posix()]=sha256(p)
            a=np.loadtxt(p,dtype=np.int64,ndmin=2);mapped=a.copy();mapped[:,1]+=offset
            target=dest/'kg'/f'{kg}-{new}.tsv';np.savetxt(target,mapped,fmt='%d',delimiter='\t')
            restored=np.loadtxt(target,dtype=np.int64,ndmin=2);restored[:,1]-=offset
            np.testing.assert_array_equal(restored,a)
    (dest/'relations.txt').write_text('\n'.join(relation_names)+'\n',encoding='utf-8')
    align=ROOT/'data/raw/atransn/SHARED/wk3l-15k_en_f_fr_aligned_entity_id.txt'
    a=np.loadtxt(align,dtype=np.int64,ndmin=2);np.savetxt(dest/'seed_alignlinks'/'en-fr.tsv',a,fmt='%d',delimiter='\t')
    np.testing.assert_array_equal(a,np.loadtxt(dest/'seed_alignlinks'/'en-fr.tsv',dtype=np.int64,ndmin=2))
    files[align.relative_to(ROOT).as_posix()]=sha256(align)
    atomic_json(dest/'strict_graph_manifest.json',{'dataset':'wk3l','input_facts':'train-only','k':10,'num_hops':2,
        'source_files':files,'relation_mapping':{'fr_offset':0,'en_offset':nfr},
        'adaptation':'Lossless IDs except invertible disjoint source relation offset; exact arrays checked; all supplied alignment pairs retained. FR target evaluation and selection only.'})
    return dest
