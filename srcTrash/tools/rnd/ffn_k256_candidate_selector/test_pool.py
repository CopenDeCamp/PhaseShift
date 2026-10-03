import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "ffn_k_tile_oracle"))

from prefilter import (act_energy, make_pool, pool_bool, pool_scores,  # noqa: E402
                       restricted_pair, restricted_residual, score_rank,
                       unrestricted_pair, weight_norm2)
from subset_select import (apply_skip_indices, objective, partials_to_rows,  # noqa: E402
                           rows_to_skip_mask, small_l2)


def test_act_energy_counts():
    torch.manual_seed(0)
    T, ng, kg = 5, 7, 8
    h = torch.randn(T, ng * kg)
    a = act_energy(h, kg)
    assert a.shape == (T, ng)
    manual = h.reshape(T, ng, kg).pow(2).sum(-1)
    assert torch.allclose(a, manual, atol=1e-5)


def test_act_energy_deterministic():
    h = torch.randn(4, 68 * 4)
    assert torch.equal(act_energy(h, 4), act_energy(h, 4))


def test_weight_norm2_counts():
    torch.manual_seed(1)
    N, ng, kg, tw = 128, 7, 8, 32
    w = torch.randn(N, ng * kg)
    wn = weight_norm2(w, kg, tw)
    assert wn.shape == (N // tw, ng)
    manual = w.reshape(N // tw, tw, ng, kg).pow(2).sum(-1).sum(1)
    assert torch.allclose(wn, manual, atol=1e-4)


def test_weight_norm2_deterministic():
    w = torch.randn(64, 68 * 4)
    assert torch.equal(weight_norm2(w, 4, 64), weight_norm2(w, 4, 64))


def test_pool_size():
    scores = torch.randn(10, 68)
    for C in (32, 40, 48, 56, 68):
        pool = make_pool(scores, C)
        assert pool.shape == (10, C)
        assert int(pool.max()) < 68
        assert torch.equal(pool, pool)
        assert len(set(pool[0].tolist())) == C


def test_pool_requires_C_ge_k():
    scores = torch.randn(3, 68)
    for C, k in ((32, 16), (40, 24), (48, 27), (56, 28), (68, 28)):
        assert C >= k
        pool = make_pool(scores, C)
        idx = restricted_residual(torch.randn(3, 68, 64), pool, k)
        assert idx.shape == (3, k)


def test_pool_outside_group_always_kept():
    torch.manual_seed(2)
    rows = torch.randn(4, 68, 64)
    scores = torch.randn(4, 68)
    pool = make_pool(scores, 32)
    for sel in (restricted_residual, lambda r, p, k: restricted_pair(r, p, k, 4, 0)):
        idx = sel(rows, pool, 16)
        pm = pool_bool(pool, 68)
        hit = torch.gather(pm, 1, idx)
        assert bool(hit.all())


def test_c68_matches_unrestricted():
    torch.manual_seed(3)
    rows = torch.randn(4, 68, 64)
    scores = torch.randn(4, 68)
    pool = make_pool(scores, 68)
    for k in (16, 24, 28):
        a = restricted_pair(rows, pool, k, 16, 8)
        b = unrestricted_pair(rows, k, 16, 8)
        assert torch.equal(torch.sort(a, -1).values, torch.sort(b, -1).values)
        assert torch.allclose(objective(rows, a), objective(rows, b), atol=1e-4)


def test_recall_calculation():
    torch.manual_seed(4)
    scores = torch.randn(6, 68)
    pool = make_pool(scores, 32)
    Sstar = small_l2(torch.randn(6, 68, 64), 16)
    pm = pool_bool(pool, 68)
    hit = torch.gather(pm, 1, Sstar)
    rec = hit.float().mean(-1)
    manual = torch.tensor([len(set(pool[r].tolist()) & set(Sstar[r].tolist())) / 16.0
                           for r in range(6)])
    assert torch.allclose(rec.cpu(), manual, atol=1e-6)


def test_candidate_only_isolation():
    torch.manual_seed(5)
    rows = torch.randn(4, 68, 64)
    pool = make_pool(torch.randn(4, 68), 40)
    pm = pool_bool(pool, 68)
    idx = restricted_residual(rows, pool, 16)
    rows2 = rows.clone()
    rows2[~pm] = torch.randn(rows2[~pm].shape) * 100.0
    idx2 = restricted_residual(rows2, pool, 16)
    assert torch.equal(idx, idx2)


def test_score_rank():
    scores = torch.tensor([[3.0, 1.0, 2.0]])
    rank = score_rank(scores)
    assert rank.tolist() == [[2, 0, 1]]


def test_pool_scores_shapes():
    torch.manual_seed(6)
    T, ng, tw, nt = 3, 68, 64, 2
    rows = torch.randn(T * nt, ng, tw)
    a = torch.rand(T, ng)
    wn = torch.rand(nt, ng)
    for kind in ("exact_l2", "act_energy", "act_wnorm"):
        q = pool_scores(kind, rows, a, wn)
        assert q.shape == (T * nt, ng)
    qe = pool_scores("act_energy", rows, a, wn)
    assert torch.allclose(qe[0], a[0])
    assert torch.allclose(qe[1], a[0])
    assert torch.allclose(qe[2], a[1])
    qw = pool_scores("act_wnorm", rows, a, wn)
    assert torch.allclose(qw[0], a[0] * wn[0])
    assert torch.allclose(qw[1], a[0] * wn[1])


def test_mask_removes_specified_groups():
    torch.manual_seed(7)
    k = 16
    c = torch.randn(68, 1, 64)
    idx = restricted_residual(partials_to_rows(c, 64), make_pool(torch.randn(1, 68), 32), k)
    keep = rows_to_skip_mask(idx, 1, 1, 68)
    assert int((~keep).sum()) == k
    y = apply_skip_indices(c, idx, 64)
    expected = c.sum(0) - c[idx.reshape(-1)].sum(0)
    assert torch.allclose(y, expected, atol=1e-5)


def test_tile_independence():
    torch.manual_seed(8)
    rows = torch.randn(2, 68, 64)
    scores = torch.randn(2, 68)
    pool = make_pool(scores, 40)
    batch = restricted_pair(rows, pool, 16, 4, 0)
    solo0 = restricted_pair(rows[0:1], pool[0:1], 16, 4, 0)
    solo1 = restricted_pair(rows[1:2], pool[1:2], 16, 4, 0)
    assert torch.equal(batch[0:1], solo0)
    assert torch.equal(batch[1:2], solo1)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"PASS {len(tests)} tests")


if __name__ == "__main__":
    main()
