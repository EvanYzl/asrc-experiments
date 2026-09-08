"""Compare vectorized SS-AGA mutual-CSLS selection to the released loop."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / "sources" / "SS-AGA"
sys.path.insert(0, str(SOURCE))

from src.ssaga_model import select_csls_mutual_links


def reference(n01, n10, c01, c10, m0, m1, a0, a1):
    links = []
    a0_set = set(int(x) for x in a0)
    a1_set = set(int(x) for x in a1)
    reverse_cache = {}
    for e0 in range(n01.shape[0]):
        if e0 in a0_set:
            continue
        best_e1 = -1
        best_score = -np.inf
        for pos, e1_tensor in enumerate(n01[e0]):
            e1 = int(e1_tensor)
            if e1 in a1_set:
                score = float(2 * c01[e0, pos] - m1[e1])
                if score > best_score:
                    best_e1 = e1
                    best_score = score
        if best_e1 == -1:
            continue
        if best_e1 not in reverse_cache:
            best_e0 = -1
            best_reverse_score = -np.inf
            for pos, candidate_tensor in enumerate(n10[best_e1]):
                candidate = int(candidate_tensor)
                if candidate not in a0_set:
                    score = float(2 * c10[best_e1, pos] - m0[candidate])
                    if score > best_reverse_score:
                        best_e0 = candidate
                        best_reverse_score = score
            reverse_cache[best_e1] = best_e0
        if reverse_cache[best_e1] == e0:
            links.append([e0, best_e1])
    return torch.tensor(links, dtype=torch.long).reshape(-1, 2)


def main():
    generator = torch.Generator().manual_seed(2020)
    for trial in range(200):
        n0 = 5 + trial % 13
        n1 = 6 + trial % 11
        k = min(3, n0, n1)
        n01 = torch.stack([
            torch.randperm(n1, generator=generator)[:k] for _ in range(n0)
        ])
        n10 = torch.stack([
            torch.randperm(n0, generator=generator)[:k] for _ in range(n1)
        ])
        # Integer-valued scores deliberately create ties and exercise the
        # released loop's strict-'>', first-candidate tie rule.
        c01 = torch.randint(-2, 3, (n0, k), generator=generator).float()
        c10 = torch.randint(-2, 3, (n1, k), generator=generator).float()
        m0 = torch.randint(-2, 3, (n0,), generator=generator).float()
        m1 = torch.randint(-2, 3, (n1,), generator=generator).float()
        a0 = torch.randperm(n0, generator=generator)[:trial % (n0 + 1)]
        a1 = torch.randperm(n1, generator=generator)[:(trial * 3) % (n1 + 1)]

        expected = reference(n01, n10, c01, c10, m0, m1, a0, a1)
        observed = select_csls_mutual_links(
            n01, n10, c01, c10, m0, m1, a0, a1, n0, n1
        )
        assert torch.equal(expected, observed), (trial, expected, observed)
    print("ssaga_csls_vectorization_equivalent=True trials=200")


if __name__ == "__main__":
    main()
