import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "ffn_k_tile_oracle"))

from prefilter import make_pool, pool_bool  # noqa: E402
from pool_sketch import pool_sketch_select, pool_sketch_z  # noqa: E402
from projection import apply_R, make_R, make_R_all  # noqa: E402
from subset_select import residual_greedy  # noqa: E402


def test_projection_deterministic():
    assert torch.equal(make_R(3, 7, 16, 0), make_R(3, 7, 16, 0))
    assert not torch.equal(make_R(3, 7, 16, 0), make_R(3, 8, 16, 0))


def test_projection_nested_and_orthonormal():
    r16 = make_R(1, 2, 16, 0)
    r8 = make_R(1, 2, 8, 0)
    assert torch.equal(r16[:8], r8)
    assert torch.allclose(r16 @ r16.t(), torch.eye(16), atol=1e-6)


def test_pool_sketch_z_shape():
    torch.manual_seed(0)
    rows = torch.randn(4, 68, 64)
    pool = make_pool(torch.randn(4, 68), 40)
    Rall = make_R_all(0, 1, 32, 0)
    z = pool_sketch_z(rows, pool, Rall, 16)
    assert z.shape == (4, 40, 16)
    zfull = apply_R(rows.reshape(4, 1, 68, 64), Rall, 16).reshape(4, 68, 16)
    manual = torch.gather(zfull, 1, pool.unsqueeze(-1).expand(-1, -1, 16))
    assert torch.allclose(z, manual, atol=1e-5)


def test_pool_sketch_matches_selector_on_z():
    torch.manual_seed(1)
    rows = torch.randn(3, 68, 64)
    pool = make_pool(torch.randn(3, 68), 40)
    Rall = make_R_all(0, 3, 32, 0)
    k = 16
    idx = pool_sketch_select(rows, pool, Rall, 16, "residual", k)
    z = pool_sketch_z(rows, pool, Rall, 16)
    idx_manual = torch.gather(pool, 1, residual_greedy(z, k))
    assert torch.equal(idx, idx_manual)


def test_pool_sketch_ignores_outside_candidates():
    torch.manual_seed(2)
    rows = torch.randn(4, 68, 64)
    pool = make_pool(torch.randn(4, 68), 40)
    pm = pool_bool(pool, 68)
    Rall = make_R_all(0, 4, 32, 0)
    a = pool_sketch_select(rows, pool, Rall, 16, "residual", 16)
    rows2 = rows.clone()
    rows2[~pm] = torch.randn(rows2[~pm].shape) * 100.0
    b = pool_sketch_select(rows2, pool, Rall, 16, "residual", 16)
    assert torch.equal(a, b)


def test_pool_sketch_outputs_inside_pool():
    torch.manual_seed(3)
    rows = torch.randn(4, 68, 64)
    pool = make_pool(torch.randn(4, 68), 48)
    Rall = make_R_all(0, 4, 48, 0)
    pm = pool_bool(pool, 68)
    for m in ("residual", "pair4"):
        idx = pool_sketch_select(rows, pool, Rall, 24, m, 24, 4)
        assert bool(torch.gather(pm, 1, idx).all())


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"PASS {len(tests)} tests")


if __name__ == "__main__":
    main()
