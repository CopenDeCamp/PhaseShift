import torch

from projection import apply_R, hadamard64, make_R, make_R_all
from sketch_select import EXACT_METHODS, SKETCH_METHODS, choose
from subset_select import apply_skip_indices, partials_to_rows, rows_to_skip_mask


def test_hadamard_orthogonal():
    H = hadamard64()
    assert H.shape == (64, 64)
    eye = (H @ H.t()) / 64.0
    assert torch.allclose(eye, torch.eye(64), atol=1e-6)


def test_projection_orthonormal_rows():
    R = make_R(3, 7, dmax=16, seed=0)
    g = R @ R.t()
    assert torch.allclose(g, torch.eye(16), atol=1e-6)


def test_projection_deterministic():
    a = make_R(3, 7, 16, 0)
    b = make_R(3, 7, 16, 0)
    assert torch.equal(a, b)
    c = make_R(3, 7, 16, 1)
    assert not torch.equal(a, c)


def test_projection_nested():
    r16 = make_R(5, 2, 16, 0)
    r8 = make_R(5, 2, 8, 0)
    r4 = make_R(5, 2, 4, 0)
    assert torch.equal(r16[:8], r8)
    assert torch.equal(r8[:4], r4)


def test_projection_tile_dependent():
    r0 = make_R(1, 0, 8, 0)
    r1 = make_R(1, 1, 8, 0)
    assert not torch.equal(r0, r1)


def test_apply_matches_matmul():
    C = torch.randn(2, 3, 68, 64)
    Rall = make_R_all(0, 3, 16, 0)
    z = apply_R(C, Rall, 4)
    assert z.shape == (2, 3, 68, 4)
    manual = torch.einsum("ptnk,tdk->ptnd", C, Rall[:, :4, :])
    assert torch.allclose(z, manual, atol=1e-5)


def test_choose_sketch_ignores_exact():
    torch.manual_seed(0)
    c = torch.randn(4, 68, 64)
    z = apply_R(c.reshape(1, 4, 68, 64), make_R_all(0, 4, 16, 0), 8).reshape(4, 68, 8)
    for m in SKETCH_METHODS:
        a = choose(m, None, z, 16)
        b = choose(m, c, z, 16)
        assert torch.equal(a, b), m


def test_choose_deterministic():
    torch.manual_seed(1)
    c = torch.randn(2, 68, 64)
    z = torch.randn(2, 68, 8)
    for m in EXACT_METHODS:
        assert torch.equal(choose(m, c, None, 16), choose(m, c, None, 16))
    for m in SKETCH_METHODS:
        assert torch.equal(choose(m, None, z, 16), choose(m, None, z, 16))


def test_mask_removes_specified_groups():
    torch.manual_seed(2)
    k = 16
    c = torch.randn(68, 1, 64)
    idx = choose("small_l2", partials_to_rows(c, 64), None, k)
    keep = rows_to_skip_mask(idx, 1, 1, 68)
    assert int((~keep).sum()) == k
    skipped = torch.nonzero(~keep.reshape(-1)).reshape(-1)
    assert torch.equal(torch.sort(skipped).values, torch.sort(idx.reshape(-1)).values)
    y = apply_skip_indices(c, idx, 64)
    expected = c.sum(0) - c[idx.reshape(-1)].sum(0)
    assert torch.allclose(y, expected, atol=1e-5)


def test_tile_independence():
    torch.manual_seed(3)
    rows = torch.randn(2, 68, 64)
    batch = choose("sk_pair4", None, rows, 16)
    solo0 = choose("sk_pair4", None, rows[0:1], 16)
    solo1 = choose("sk_pair4", None, rows[1:2], 16)
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
