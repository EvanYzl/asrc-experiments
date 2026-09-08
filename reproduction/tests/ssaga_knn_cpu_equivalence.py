"""Check SS-AGA's chunked CPU kNN against the released exhaustive formula."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

SOURCE = Path(__file__).resolve().parents[1] / "sources" / "SS-AGA"
sys.path.insert(0, str(SOURCE))

from src.ssaga_model import get_KNN_batch


def main() -> None:
    generator = torch.Generator().manual_seed(2020)
    for trial in range(100):
        n_query = 7 + trial % 19
        n_candidate = 11 + trial % 23
        dim = 5 + trial % 17
        k = min(3, n_candidate)
        query = torch.randn(n_query, dim, generator=generator)
        candidate = torch.randn(n_candidate, dim, generator=generator)
        unused_query_bert = torch.randn(n_query, dim, generator=generator)
        unused_candidate_bert = torch.randn(n_candidate, dim, generator=generator)

        expected = torch.topk(
            -torch.norm(
                query.unsqueeze(1) - candidate.unsqueeze(0), dim=2
            ),
            k=k,
        ).indices
        observed = get_KNN_batch(
            (query, unused_query_bert),
            (candidate, unused_candidate_bert),
            k=k,
            device=torch.device("cpu"),
            batch_size=1 + trial % 9,
        )
        assert torch.equal(expected, observed), (trial, expected, observed)

    print("ssaga_cpu_knn_equivalent=True trials=100")


if __name__ == "__main__":
    main()
