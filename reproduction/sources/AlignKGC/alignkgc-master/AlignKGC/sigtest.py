import argparse
import csv
import re
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


def csv_to_e2rr(av: argparse.Namespace, which_algo: str):
    eval_path = "{}/combined/Combined_{}_{}/{}/save_eval.csv".format(av.dbp5l,
                av.ea_percent, av.ra_percent, which_algo)
    ans = dict()
    sum_e2rr, num_e2rr = 0, 0
    with open(eval_path) as csvfile:
        for (name, sub, rel, obj, sub_rank, obj_rank) in csv.reader(csvfile):
            # only filtered
            if not re.match(".*_f_test.txt", name):
                continue
            ans[(name, sub, rel, obj)] = 1. / float(obj_rank)
            sum_e2rr += 1. / float(obj_rank)
            num_e2rr += 1.
    print(which_algo, "\te2mrr", sum_e2rr / num_e2rr)
    return ans


def scatter(perf_a, perf_b):
    assert sorted(perf_a.keys()) == sorted(perf_b.keys()), \
        "{} != {}".format(len(perf_a), len(perf_b))
    perf_arr_a, perf_arr_b = list(), list()
    for k in sorted(perf_a.keys()):
        perf_arr_a.append(perf_a[k])
        perf_arr_b.append(perf_b[k])
    tstat, pval = ttest_rel(perf_arr_a, perf_arr_b)
    print(tstat, pval)


def main(av):
    perf_a = csv_to_e2rr(av, av.algo_a)
    perf_b = csv_to_e2rr(av, av.algo_b)
    scatter(perf_a, perf_b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbp5l", required=True, help="/path/to/DBP-5L/")
    ap.add_argument("--ea_percent", type=int, required=True)
    ap.add_argument("--ra_percent", type=int, required=True)
    ap.add_argument("--algo_a", required=True, help="model_AlgoA/tmpAAA")
    ap.add_argument("--algo_b", required=True, help="model_AlgoB/tmpBBB")
    av = ap.parse_args()
    main(av)
