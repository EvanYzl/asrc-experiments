"""Audit KEnS batched alignment kNN against its original Keras implementation."""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

KENS_ROOT = Path(__file__).resolve().parents[1] / 'sources' / 'KEnS'
sys.path.insert(0, str(KENS_ROOT))

import src.param as param
from src.model import (
    assert_batched_knn_matches_reference,
    exact_batched_l2_knn,
)


def load_entity_embeddings(path):
    with h5py.File(path, 'r') as handle:
        datasets = []
        handle['model_weights'].visititems(
            lambda name, obj: datasets.append((name, obj[()]))
            if isinstance(obj, h5py.Dataset)
            and name.endswith('embeddings:0')
            else None
        )
    if not datasets:
        raise RuntimeError(f'No embedding dataset found in {path}')
    # Entity embeddings are the largest first-axis embedding table.
    return max((value for _, value in datasets), key=lambda value: value.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-model', type=Path)
    parser.add_argument('--candidate-model', type=Path)
    parser.add_argument('--synthetic-shape', type=int, nargs=3,
                        metavar=('QUERIES', 'CANDIDATES', 'DIM'))
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()

    if args.synthetic_shape is not None:
        n_query, n_candidate, dim = args.synthetic_shape
        rng = np.random.default_rng(2020)
        queries = rng.normal(size=(n_query, dim)).astype(np.float32)
        candidates = rng.normal(size=(n_candidate, dim)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
    elif args.query_model is not None and args.candidate_model is not None:
        queries = load_entity_embeddings(args.query_model).astype(np.float32)
        candidates = load_entity_embeddings(args.candidate_model).astype(np.float32)
    else:
        parser.error('provide both model paths or --synthetic-shape')
    if queries.shape[1] != candidates.shape[1]:
        raise RuntimeError('Embedding dimensions differ')
    param.dim = queries.shape[1]
    param.k = args.k

    sample_ids = np.unique(np.linspace(
        0, queries.shape[0] - 1,
        num=min(args.samples, queries.shape[0]), dtype=np.int64
    ))
    elapsed = 0.0
    for _ in range(args.repeat):
        start = time.perf_counter()
        batched = exact_batched_l2_knn(
            queries, candidates, args.k, batch_size=args.batch_size
        )
        elapsed += time.perf_counter() - start
        assert_batched_knn_matches_reference(
            queries[sample_ids],
            candidates,
            batched[sample_ids],
            args.k,
            sample_count=len(sample_ids),
        )
    print(
        f'PASS queries={queries.shape[0]} candidates={candidates.shape[0]} '
        f'dim={queries.shape[1]} k={args.k} checked={len(sample_ids)}x{args.repeat} '
        f'batched_seconds_total={elapsed:.3f}'
    )


if __name__ == '__main__':
    main()
