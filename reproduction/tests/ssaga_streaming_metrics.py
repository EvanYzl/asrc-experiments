"""Prove SS-AGA streaming metrics equal its original concatenated metrics."""

import sys
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[1] / "sources" / "SS-AGA"
sys.path.insert(0, str(SOURCE))

from src.validate import Tester


def main():
    generator = torch.Generator().manual_seed(2020)
    rows = torch.stack([torch.randperm(257, generator=generator) for _ in range(37)])
    truth = torch.tensor([int(rows[i, (i * 17) % 257]) for i in range(37)]).view(-1, 1)

    tester = object.__new__(Tester)
    old_hits_1, old_hits_10, old_mrr = tester.get_hit_mrr(rows, truth)
    new_hits_1 = 0
    new_hits_10 = 0
    new_rr = 0.0
    for start in range(0, rows.shape[0], 8):
        h1, h10, rr = tester.get_batch_hit_mrr(
            rows[start:start + 8], truth[start:start + 8]
        )
        new_hits_1 += h1
        new_hits_10 += h10
        new_rr += rr

    assert old_hits_1 == new_hits_1
    assert old_hits_10 == new_hits_10
    assert abs(float(old_mrr) - new_rr / rows.shape[0]) < 1e-8
    print(
        f"ssaga_streaming_metrics_equivalent=True "
        f"hits1={new_hits_1} hits10={new_hits_10} "
        f"mrr={new_rr / rows.shape[0]:.9f}"
    )


if __name__ == "__main__":
    main()
