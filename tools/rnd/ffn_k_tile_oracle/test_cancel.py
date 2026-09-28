import torch

from oracle_down import group_partials, oracle_down_chunk
from subset_select import (apply_skip_indices, exact_subset, objective, pair_greedy_swap,
                           pair_seeded_greedy, partials_to_rows, residual_greedy, small_l2)


def _rand(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32)


def test_cancellation_pair():
    C = torch.zeros(1, 8, 4)
    C[0, 0, 0] = 10.0
    C[0, 1, 0] = -10.0
    C[0, 2, 0] = 0.1
    C[0, 3, 0] = 0.2
    C[0, 4, 0] = 0.05
    C[0, 5, 0] = 0.07
    C[0, 6, 0] = 0.03
    C[0, 7, 0] = 0.04
    s_small = small_l2(C, 2)
    s_pair = pair_greedy_swap(C, 2)
    assert set(s_pair[0].tolist()) == {0, 1}, s_pair
    assert objective(C, s_pair).item() < 1e-12
    assert objective(C, s_small).item() > objective(C, s_pair).item() + 1e-6
    print(f"  small_l2={s_small[0].tolist()} J={objective(C, s_small).item():.6g}")
    print(f"  pair_swap={s_pair[0].tolist()} J={objective(C, s_pair).item():.6g}")


def test_token_dependent():
    C = torch.zeros(2, 8, 2)
    C[0, 0, 0] = 10.0
    C[0, 1, 0] = -10.0
    C[1, 2, 1] = 10.0
    C[1, 3, 1] = -10.0
    others = {0: {2, 3, 4, 5, 6, 7}, 1: {0, 1, 4, 5, 6, 7}}
    for r, gs in others.items():
        for g in gs:
            C[r, g, 0] = 0.1 * g + 0.05
            C[r, g, 1] = 0.1 * g + 0.07
    idx = pair_greedy_swap(C, 2)
    assert set(idx[0].tolist()) == {0, 1}, idx[0]
    assert set(idx[1].tolist()) == {2, 3}, idx[1]


def test_tile_independent():
    A = torch.zeros(1, 8, 2)
    A[0, 0, 0] = 5.0
    A[0, 1, 0] = -5.0
    B = torch.zeros(1, 8, 2)
    B[0, 2, 1] = 5.0
    B[0, 3, 1] = -5.0
    for r, target in ((A, {0, 1}), (B, {2, 3})):
        for g in range(8):
            if g not in target:
                r[0, g, 0] = 0.1 * g + 0.05
                r[0, g, 1] = 0.1 * g + 0.07
    C = torch.cat([A, B], 0)
    idx = pair_greedy_swap(C, 2)
    assert set(idx[0].tolist()) == {0, 1}, idx[0]
    assert set(idx[1].tolist()) == {2, 3}, idx[1]


def test_deterministic():
    C = _rand((16, 12, 8), 3)
    a = pair_greedy_swap(C, 4)
    b = pair_greedy_swap(C, 4)
    assert torch.equal(a, b)
    c = residual_greedy(C, 4)
    d = residual_greedy(C, 4)
    assert torch.equal(c, d)


def test_exhaustive_methods():
    results = {}
    for seed in range(4):
        ng, k, D, B = 11, 4, 8, 5
        C = _rand((B, ng, D), 100 + seed)
        ex = exact_subset(C, k)
        j_ex = objective(C, ex)
        methods = {
            "SMALL_L2": small_l2(C, k),
            "RESIDUAL_GREEDY": residual_greedy(C, k),
            "PAIR_SEEDED_GREEDY": pair_seeded_greedy(C, k, seed_count=16),
            "PAIR_GREEDY_SWAP": pair_greedy_swap(C, k, seed_count=16),
        }
        for name, idx in methods.items():
            r = (objective(C, idx) / j_ex.clamp_min(1e-30)).mean().item()
            results.setdefault(name, []).append(r)
    for name, rs in results.items():
        avg = sum(rs) / len(rs)
        print(f"  {name:20s} objective/exact mean={avg:.4f} max={max(rs):.4f}")
    assert max(results["PAIR_GREEDY_SWAP"]) < 2.0


def test_apply_skip_exact():
    ng, T, N, tw = 8, 5, 32, 16
    h = _rand((T, ng * 256), 7)
    w = _rand((N, ng * 256), 8)
    c = group_partials(h, w, 256)
    nt = N // tw
    rows = partials_to_rows(c, tw)
    R = T * nt
    skip = torch.stack([torch.randperm(ng)[:3] for _ in range(R)])
    y = apply_skip_indices(c, skip, tw)
    ref = torch.zeros(T, N)
    cr = c
    for t in range(T):
        for tile in range(nt):
            r = t * nt + tile
            kept = [g for g in range(ng) if g not in set(skip[r].tolist())]
            for g in kept:
                ref[t, tile * tw:(tile + 1) * tw] += cr[g, t, tile * tw:(tile + 1) * tw]
    assert torch.allclose(y, ref, atol=1e-6)


def test_skip0_full():
    ng, T, N, tw = 8, 4, 32, 16
    h = _rand((T, ng * 256), 11)
    w = _rand((N, ng * 256), 12)
    c = group_partials(h, w, 256)
    R = T * (N // tw)
    skip = torch.zeros(R, 0, dtype=torch.long)
    y = apply_skip_indices(c, skip, tw)
    assert torch.allclose(y, c.sum(0), atol=1e-6)


def test_small_l2_matches_oracle():
    ng, T, N, tw, k = 8, 4, 32, 16, 2
    h = _rand((T, ng * 256), 21)
    w = _rand((N, ng * 256), 22)
    y_ref, _, _ = oracle_down_chunk(h, w, 256, tw, ng - k)
    c = group_partials(h, w, 256)
    rows = partials_to_rows(c, tw)
    skip = small_l2(rows, k)
    y = apply_skip_indices(c, skip, tw)
    assert torch.allclose(y, y_ref, atol=1e-5)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"PASS {len(tests)} tests")


if __name__ == "__main__":
    main()
