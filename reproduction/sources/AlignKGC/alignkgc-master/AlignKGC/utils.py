import csv
import shutil
import os
import sys
import subprocess
import typing
from collections import Counter
from typing import Any, Dict, List, Tuple
import torch
from torch.functional import Tensor
import tqdm

from importers.ea_ra_kgc import EaRaKgcData


color_map = {"green":32, "black":0, "blue":34, "red":31, 'yellow':33}


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=None, fill='*', color="blue"):
    """
    Simple utility to display an updatable progress bar on the terminal.\n
    :param iteration: The numer of iteration completed (% task done)
    :param total: The total number of iterations to be done
    :param prefix: The string to be displayed before the progress bar
    :param suffix: The string to be displayed after the progress bar
    :param decimals: The number of decimal places in the percent complete indicator
    :param length: The length of the bar. If None, the length is calculated as to fill up the screen
    :param fill: The character to fill the bar with
    :param color: Color of the bar
    :return: None
    """
    if(length is None):
        r, _ = shutil.get_terminal_size((120, 80))
        length = max(r-len(prefix)-len(suffix)-11, 10)
    percent = ("{0:5."+str(decimals)+"f}").format(100.0*iteration/total)
    filled_length = int(length*iteration//total)
    bar = fill*filled_length + '-'*(length-filled_length)
    print('\r%s |\033[1;%dm%s\033[0;0m| %s%% %s'%(prefix, color_map[color], bar, percent, suffix), end='\r')
    if iteration==total:
        print()


def colored_print(color, message):
    """
    Simple utility to print in color
    :param color: The name of color from color_map
    :param message: The message to print in color
    :return: None
    """
    print('\033[1;%dm%s\033[0;0m' % (color_map[color], message))


class tee(object):
    """Used to duplicate stdout and/or stderr to a named log file."""
    def __init__(self, out_err_path_, append=False) -> None:
        super().__init__()
        self.out_err_path = out_err_path_
        self.out_err_file = open(out_err_path_, "a" if append else "w", buffering=1)
        self.stdout = sys.stdout
        print("opening", self.out_err_path)

    def __del__(self):
        print("closing", self.out_err_path)
        self.out_err_file.close()

    def write(self, data):
        self.out_err_file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.out_err_file.flush()
        self.stdout.flush()


def removeElements(lst, k): 
    """
    helper for removing elements in list occuring less than k times
    """
    counted = Counter(lst) 
    return [el for el in lst if counted[el] >= k] 


def get_entity_alignments(filename, emap=None):
    """ deprecated """
    mappings ={}
    f=open(filename)
    lines = f.readlines()
    for line in lines:
        a=line.split()
        x=a[0]
        y=a[1]
        if x in emap and y in emap:
            p1=emap[x]
            p2=emap[y]
            mappings[p1]=p2
            mappings[p2]=p1
    return mappings


def get_filter(filename, em=None,rm=None,add_unknowns=True, nonoov_entity_count=None):
    filt=[]
    file=open(filename)
    lines = file.readlines()
    for line in lines:
        a=line.split()
        x=a[0]
        if x in em:
            filt.append(em[x])
    print(len(filt))
    return filt


def intersect(a, b):
    return list(set(a) & set(b))


def union(a, b):
    return list(set(a) | set(b))


def log_eval_scores(writer, valid_score, test_score, num_iter):
    for metric in ['mrr','hits10','hits1']:
        writer.add_scalar('{}/valid_m'.format(metric), valid_score['m'][metric] , num_iter)
        writer.add_scalar('{}/valid_e1'.format(metric), valid_score['e1'][metric] , num_iter)
        writer.add_scalar('{}/valid_e2'.format(metric), valid_score['e2'][metric] , num_iter)
        writer.add_scalar('{}/test_m'.format(metric), test_score['m'][metric] , num_iter)
        writer.add_scalar('{}/test_e1'.format(metric), test_score['e1'][metric] , num_iter)
        writer.add_scalar('{}/test_e2'.format(metric), test_score['e2'][metric] , num_iter)


def get_rel_align_dict(meta, kgc_train_path : str,
                       emap : Dict[Any,int]=None,
                       rmap : Dict[Any,int]=None):
    """
    Computes SO-pair sets of relations.
    :param meta: not needed, but may need later
    :param kgc_train_path: combined (s,r,o) train fold.
    entIDs are global. relID is lang-prefixed global.
    :param emap: entity map
    :param rmap: relation map
    :return: dict(relID, set(SO-pair)); dict(relID, lang)
    """
    rel_ent_pairs : Dict[int, List[Tuple[int, int]]] = dict()
    lang : Dict[int, str] = dict()
    train=open(kgc_train_path)
    lines = train.readlines()
    for line in lines:
        a=line.split()
        assert type(rmap[a[1]]) == int
        if rmap[a[1]] not in rel_ent_pairs:
            rel_ent_pairs[rmap[a[1]]] = []
            lang[rmap[a[1]]] = a[1][0]
        rel_ent_pairs[rmap[a[1]]].append((emap[a[0]], emap[a[2]]))
    train.close()
    return rel_ent_pairs, lang


def _relation_vectors(pairs, S_re, S_im):
    """Build all normalized subject/object vectors for one relation at once."""
    device = S_re.weight.device
    pair_tensor = torch.as_tensor(pairs, dtype=torch.long, device=device)
    subjects, objects = pair_tensor[:, 0], pair_tensor[:, 1]
    vectors = torch.cat((S_re(subjects), S_im(subjects),
                         S_re(objects), S_im(objects)), dim=-1)
    return vectors / torch.norm(vectors, dim=1, keepdim=True)


def _relation_pair_score(left, right, a, b):
    """The released soft-intersection score for one ordered relation pair."""
    mat = left @ right.t()
    val = torch.max(mat, 1).values
    val1 = torch.max(mat, 0).values
    sig = torch.sigmoid(a * val - b)
    sig1 = torch.sigmoid(a * val1 - b)
    # This is the same membership test and subtraction order as the release,
    # without materialising ``sig1.repeat(...).t()``.
    not_intersect = sig[(sig1[:, None] != sig[None, :]).t().prod(1) == 1]
    return (torch.sum(sig) - torch.sum(not_intersect)) / len(mat)


def _scores_against_language(left, target_vectors, target_segments,
                             target_relation_count, a, b,
                             column_chunk_size=8192):
    """Score one relation against every relation of a target language.

    The original implementation launches one small matrix multiplication for
    every relation pair.  Here the target relation matrices are concatenated,
    while segmented maxima reproduce each individual pair's row/column maxima.
    Chunking bounds temporary memory without changing the reductions.
    """
    row_count = left.shape[0]
    device = left.device
    dtype = left.dtype
    row_max = torch.full((row_count, target_relation_count), -torch.inf,
                         dtype=dtype, device=device)
    col_max_parts = []
    for start in range(0, target_vectors.shape[0], column_chunk_size):
        end = min(start + column_chunk_size, target_vectors.shape[0])
        segment = target_segments[start:end]
        mat = left @ target_vectors[start:end].t()
        row_max.scatter_reduce_(
            1, segment.unsqueeze(0).expand(row_count, -1), mat,
            reduce="amax", include_self=True)
        col_max_parts.append(mat.max(dim=0).values)

    sig = torch.sigmoid(a * row_max - b)
    sig1 = torch.sigmoid(a * torch.cat(col_max_parts) - b)
    membership_count = torch.zeros(
        (row_count, target_relation_count), dtype=torch.int32, device=device)
    for start in range(0, sig1.shape[0], column_chunk_size):
        end = min(start + column_chunk_size, sig1.shape[0])
        segment = target_segments[start:end]
        matches = sig.index_select(1, segment).eq(sig1[start:end].unsqueeze(0))
        membership_count.scatter_add_(
            1, segment.unsqueeze(0).expand(row_count, -1),
            matches.to(torch.int32))
    has_match = membership_count.ne(0)
    not_intersect = torch.where(has_match, torch.zeros_like(sig), sig)
    return (sig.sum(dim=0) - not_intersect.sum(dim=0)) / row_count


def update_imp_sc(meta: EaRaKgcData,
                  rel_ent_pairs : Dict[int, List[Tuple[int, int]]],
                  lang : Dict[int, str], S_re, S_im, a, b,
                  track_grad=False):
    """
    With current entity embeddings, recomputes soft-asymmetric set similarities
    (as approximate max matching) between relations, which are represented
    as SO-vector sets.
    :param rel_ent_pairs: dict(relID, set(SO-pair)) ... map : rel -> SO pairs
    :param lang: dict(relID, lidStr) ... relation to language
    :param S_re: entity embeddings, real part
    :param S_im: entity embeddings, imaginary part
    :param a: sigmoid slope
    :param b: sigmoid offset
    """
    lid_strings = [str(lid) for lid in meta.lids()]
    shared_lid = meta.maxlid_plus_one()

    # Candidate selection has never contributed gradients in the released
    # trainer.  Build one compact vector matrix per relation under no_grad;
    # selected mutual pairs are recomputed below when gradients are required.
    with torch.no_grad():
        grid_to_sovecs : Dict[int, Any] = {}
        for rel0, pairs in tqdm.tqdm(
                rel_ent_pairs.items(), desc="update_imp_sc, relation vectors"):
            assert type(rel0) == int
            if len(pairs) < 2 or int(lang[rel0]) == shared_lid:
                continue
            grid_to_sovecs[rel0] = _relation_vectors(pairs, S_re, S_im)

        relations_by_language = {
            lid: sorted((rel for rel in grid_to_sovecs if lang[rel] == lid),
                        reverse=True)
            for lid in lid_strings
        }
        directed : Dict[str, Dict[int, Tuple[Tensor, int]]] = {
            lid: {} for lid in lid_strings
        }

        for source_lid in lid_strings:
            source_relations = relations_by_language[source_lid]
            for target_lid in lid_strings:
                if target_lid == source_lid:
                    continue
                target_relations = relations_by_language[target_lid]
                if not source_relations or not target_relations:
                    continue
                target_sizes = [grid_to_sovecs[rel].shape[0]
                                for rel in target_relations]
                target_vectors = torch.cat(
                    [grid_to_sovecs[rel] for rel in target_relations], dim=0)
                target_segments = torch.repeat_interleave(
                    torch.arange(len(target_relations),
                                 device=target_vectors.device),
                    torch.as_tensor(target_sizes, device=target_vectors.device))
                score_rows = []
                for rel1 in tqdm.tqdm(
                        source_relations,
                        desc="update_imp_sc {}->{}".format(source_lid,
                                                           target_lid),
                        leave=False):
                    score_rows.append(_scores_against_language(
                        grid_to_sovecs[rel1], target_vectors, target_segments,
                        len(target_relations), a, b))
                score_matrix = torch.stack(score_rows, dim=0)
                best_columns = score_matrix.argmax(dim=1)
                best_scores = score_matrix.gather(
                    1, best_columns.unsqueeze(1)).squeeze(1)
                best_columns_cpu = best_columns.cpu().tolist()
                for row, rel1 in enumerate(source_relations):
                    directed[target_lid][rel1] = (
                        best_scores[row], target_relations[best_columns_cpu[row]])

        mutual = []
        for rel1 in grid_to_sovecs:
            source_lid = lang[rel1]
            for target_lid in lid_strings:
                if target_lid == source_lid or rel1 not in directed[target_lid]:
                    continue
                _, rel2 = directed[target_lid][rel1]
                if rel2 in directed[source_lid] and \
                        directed[source_lid][rel2][1] == rel1:
                    mutual.append((target_lid, rel1, rel2))

    # Recompute only the selected mutual pairs with the released pairwise
    # expression.  This both matches its final score and keeps autograd graphs
    # small for the late-stage update of parameter ``b``.
    vector_cache = {}
    score_cache = {}

    def vectors(rel):
        if rel not in vector_cache:
            vector_cache[rel] = _relation_vectors(
                rel_ent_pairs[rel], S_re, S_im)
        return vector_cache[rel]

    def score(rel1, rel2):
        key = (rel1, rel2)
        if key not in score_cache:
            score_cache[key] = _relation_pair_score(
                vectors(rel1), vectors(rel2), a, b)
        return score_cache[key]

    context = torch.enable_grad() if track_grad else torch.no_grad()
    equiv_rel = {lid: {} for lid in lid_strings}
    with context:
        for target_lid, rel1, rel2 in tqdm.tqdm(
                mutual, desc="update_imp_sc, mutual pairs"):
            sc = score(rel1, rel2)
            sc2 = score(rel2, rel1)
            equiv_rel[target_lid][rel1] = (min(sc, sc2), rel2)
    return equiv_rel


class SaveEval(object):
    """To be passed as a hook into evaluate.evaluate for saving reciprocal rank
    of each test instance to a file for significance tests."""
    def __init__(self, save_eval_file: typing.IO) -> None:
        super().__init__()
        self.eval_csv = csv.writer(save_eval_file)

    def save(self, lang: str, sub: Tensor, rel: Tensor, obj: Tensor,
             sub_ranks: Tensor, obj_ranks: Tensor):
        sub_, rel_, obj_ = sub.cpu().numpy(), rel.cpu().numpy(), obj.cpu().numpy()
        sub_ranks_, obj_ranks_ = sub_ranks.cpu().numpy(), obj_ranks.cpu().numpy()
        nrows = sub_.shape[0]
        assert nrows == rel_.shape[0] and nrows == obj_.shape[0]
        assert nrows == sub_ranks_.shape[0] and nrows == obj_ranks_.shape[0]
        for rx in range(nrows):
            self.eval_csv.writerow([lang] + [sub_[rx,0]] + [rel_[rx,0]] +
                                   [obj_[rx,0]] + [sub_ranks_[rx]] +
                                   [obj_ranks_[rx]])
