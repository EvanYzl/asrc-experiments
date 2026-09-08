"""Regression checks for semantics-preserving AlignKGC acceleration."""

import sys
from pathlib import Path

import torch


SOURCE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SOURCE))

from AlignKGC.AlignKGC_trainer import AlignKGC  # noqa: E402
from AlignKGC.models import complex as Complex  # noqa: E402
from AlignKGC.utils import (  # noqa: E402
    _relation_pair_score,
    _relation_vectors,
    update_imp_sc,
)


class Meta:
    def lids(self):
        return [1, 2, 3]

    def maxlid_plus_one(self):
        return 4


def reference_update(meta, relation_pairs, language, e_re, e_im, a, b):
    vectors = {
        relation: _relation_vectors(pairs, e_re, e_im)
        for relation, pairs in relation_pairs.items()
        if len(pairs) >= 2 and int(language[relation]) != meta.maxlid_plus_one()
    }
    directed = {str(lid): {} for lid in meta.lids()}
    for relation1 in vectors:
        for target_lid in directed:
            if target_lid == language[relation1]:
                continue
            candidates = []
            for relation2 in vectors:
                if language[relation2] != target_lid:
                    continue
                score = _relation_pair_score(
                    vectors[relation1], vectors[relation2], a, b)
                candidates.append((float(score.detach()), relation2, score))
            _, relation2, score = max(candidates, key=lambda row: (row[0], row[1]))
            directed[target_lid][relation1] = (score, relation2)

    answer = {str(lid): {} for lid in meta.lids()}
    for relation1 in vectors:
        source_lid = language[relation1]
        for target_lid in directed:
            if target_lid == source_lid:
                continue
            score, relation2 = directed[target_lid][relation1]
            score2, relation3 = directed[source_lid][relation2]
            if relation3 == relation1:
                answer[target_lid][relation1] = (min(score, score2), relation2)
    return answer


def compare_updates(actual, expected):
    assert actual.keys() == expected.keys()
    for lid in actual:
        assert actual[lid].keys() == expected[lid].keys()
        for relation in actual[lid]:
            actual_score, actual_relation = actual[lid][relation]
            expected_score, expected_relation = expected[lid][relation]
            assert actual_relation == expected_relation
            torch.testing.assert_close(actual_score, expected_score, rtol=2e-6, atol=1e-8)


def test_soft_relation_update():
    torch.manual_seed(17)
    e_re = torch.nn.Embedding(18, 7)
    e_im = torch.nn.Embedding(18, 7)
    relation_pairs = {
        101: [(0, 1), (2, 3), (4, 5)],
        102: [(1, 3), (3, 5)],
        201: [(6, 7), (8, 9), (10, 11)],
        202: [(7, 9), (9, 11)],
        301: [(12, 13), (14, 15), (16, 17)],
        302: [(13, 15), (12, 17)],
        401: [(0, 6), (1, 7)],
    }
    language = {
        101: "1", 102: "1", 201: "2", 202: "2",
        301: "3", 302: "3", 401: "4",
    }
    meta = Meta()
    a = 2.3
    b = torch.tensor(1.1, requires_grad=True)
    with torch.no_grad():
        expected = reference_update(
            meta, relation_pairs, language, e_re, e_im, a, b)
    actual = update_imp_sc(
        meta, relation_pairs, language, e_re, e_im, a, b,
        track_grad=False)
    compare_updates(actual, expected)

    differentiable = update_imp_sc(
        meta, relation_pairs, language, e_re, e_im, a, b,
        track_grad=True)
    compare_updates(differentiable, expected)
    loss = sum(score for matches in differentiable.values()
               for score, _ in matches.values())
    loss.backward(retain_graph=True)
    assert b.grad is not None and torch.isfinite(b.grad)


def reference_relation_loss(trainer, loss_type, entity_backtrack):
    loss = torch.zeros((), dtype=trainer.scoring_function.R_re.weight.dtype)
    cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for lid in trainer.meta.lids():
        for relation1, (score, relation2) in trainer.equiv_rel[str(lid)].items():
            if not entity_backtrack:
                score = score.detach()
            p1 = torch.tensor([relation1])
            p2 = torch.tensor([relation2])
            if loss_type == "L1":
                loss = loss + (
                    (score * trainer.scoring_function.R_re(p1) -
                     score * trainer.scoring_function.R_re(p2)).abs().mean() +
                    (score * trainer.scoring_function.R_im(p1) -
                     score * trainer.scoring_function.R_im(p2)).abs().mean())
            else:
                loss = loss + (
                    score * (1 - cosine(trainer.scoring_function.R_re(p1),
                                        trainer.scoring_function.R_re(p2))) +
                    score * (1 - cosine(trainer.scoring_function.R_im(p1),
                                        trainer.scoring_function.R_im(p2))))
    return loss


def test_relation_loss():
    torch.manual_seed(23)
    holder = type("Holder", (), {})()
    holder.R_re = torch.nn.Embedding(16, 9)
    holder.R_im = torch.nn.Embedding(16, 9)
    trainer = object.__new__(AlignKGC)
    trainer.scoring_function = holder
    trainer.meta = Meta()
    scores = [torch.tensor(0.2, requires_grad=True),
              torch.tensor(0.7, requires_grad=True),
              torch.tensor(0.4, requires_grad=True)]
    trainer.equiv_rel = {
        "1": {1: (scores[0], 8)},
        "2": {5: (scores[1], 11)},
        "3": {9: (scores[2], 2)},
    }
    for loss_type in ("L1", "cos"):
        for backtrack in (0, 1):
            expected = reference_relation_loss(trainer, loss_type, backtrack)
            actual = trainer.rel_alignment_loss(loss_type, backtrack)
            torch.testing.assert_close(actual, expected.reshape(()),
                                       rtol=2e-6, atol=1e-8)


def test_subsampled_entity_scores():
    torch.manual_seed(29)
    reference = Complex(entity_count=37, relation_count=11, embedding_dim=13,
                        batch_norm=False, unit_reg=False)
    optimized = Complex(entity_count=37, relation_count=11, embedding_dim=13,
                        batch_norm=False, unit_reg=False)
    optimized.load_state_dict(reference.state_dict())
    s = torch.randint(0, 37, (6, 1))
    r = torch.randint(0, 11, (6, 1))
    o = torch.randint(0, 37, (6, 1))
    head_candidates = torch.randperm(37)[:17]
    tail_candidates = torch.randperm(37)[:17]

    head_full = reference(None, r, o)
    tail_full = reference(s, r, None)
    expected_head_positive = head_full.gather(1, s).squeeze(1)
    expected_head_negative = head_full.index_select(1, head_candidates)
    expected_tail_positive = tail_full.gather(1, o).squeeze(1)
    expected_tail_negative = tail_full.index_select(1, tail_candidates)
    actual = optimized.score_subsampled_entities(
        s, r, o, head_candidates, tail_candidates)
    expected = (expected_head_positive, expected_head_negative,
                expected_tail_positive, expected_tail_negative)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor,
                                   rtol=2e-5, atol=2e-7)

    labels = torch.zeros(s.shape[0], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()
    reference_loss = (
        criterion(torch.cat((expected[0][:, None], expected[1]), dim=1), labels)
        + criterion(torch.cat((expected[2][:, None], expected[3]), dim=1), labels))
    optimized_loss = (
        criterion(torch.cat((actual[0][:, None], actual[1]), dim=1), labels)
        + criterion(torch.cat((actual[2][:, None], actual[3]), dim=1), labels))
    reference_loss.backward()
    optimized_loss.backward()
    torch.testing.assert_close(optimized_loss, reference_loss,
                               rtol=2e-6, atol=1e-8)
    for actual_parameter, expected_parameter in zip(
            optimized.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad,
                                   rtol=3e-5, atol=3e-7)


if __name__ == "__main__":
    test_soft_relation_update()
    test_relation_loss()
    test_subsampled_entity_scores()
    print("alignkgc_grouped_relation_update_equivalent=True")
    print("alignkgc_vectorized_relation_loss_equivalent=True")
    print("alignkgc_direct_subsample_scores_and_gradients_equivalent=True")
