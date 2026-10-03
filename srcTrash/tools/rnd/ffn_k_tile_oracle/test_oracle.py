import torch

from oracle_down import (apply_mask, group_count, group_partials, group_scores,
                         oracle_down, oracle_down_chunk, reference_down,
                         reference_masked_down, select_keep_mask, tile_count)


def _rand(shape, seed, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype)


def test_skip0_parity():
    h = _rand((17, 512), 1, torch.bfloat16)
    w = _rand((128, 512), 2, torch.bfloat16)
    y, _, _ = oracle_down_chunk(h, w, 256, 64, 2)
    ref = reference_down(h, w)
    assert torch.allclose(y, ref, atol=1e-3), (y - ref).abs().max().item()
    full = oracle_down(h, w, 256, 64, 2, chunk=7)
    assert torch.allclose(full.float(), ref.to(torch.bfloat16).float(), atol=0.0)


def test_chunk_invariance():
    h = _rand((23, 512), 3)
    w = _rand((128, 512), 4)
    a = oracle_down(h, w, 256, 64, 1, chunk=5)
    b = oracle_down(h, w, 256, 64, 1, chunk=23)
    assert torch.allclose(a, b, atol=1e-5)


def test_group_removed_exactly():
    h = _rand((9, 512), 5)
    w = _rand((128, 512), 6)
    skip = 1
    y = reference_masked_down(h, w, 256, 64, skip)
    c = group_partials(h, w, 256)
    score = group_scores(c, 64)
    ng = group_count(512, 256)
    nt = tile_count(128, 64)
    mask = select_keep_mask(score, ng - skip)
    T, N = 9, 128
    manual = torch.zeros(T, nt, 64)
    for g in range(ng):
        keep = mask[:, :, g].reshape(T, nt, 1)
        manual += c[g].reshape(T, nt, 64) * keep
    manual = manual.reshape(T, N)
    assert torch.allclose(y, manual, atol=1e-5)
    assert (mask.sum(-1) == ng - skip).all()


def test_token_masks_differ():
    h = _rand((64, 512), 7)
    w = _rand((128, 512), 8)
    c = group_partials(h, w, 256)
    score = group_scores(c, 64)
    mask = select_keep_mask(score, 1)
    rows = mask.reshape(mask.shape[0], -1).cpu().tolist()
    assert len({tuple(r) for r in rows}) > 1


def test_tile_masks_differ():
    h = _rand((8, 512), 9)
    w = _rand((128, 512), 10)
    c = group_partials(h, w, 256)
    score = group_scores(c, 64)
    mask = select_keep_mask(score, 1)
    for t in range(8):
        a = mask[t, 0].cpu().tolist()
        b = mask[t, 1].cpu().tolist()
        if a != b:
            return
    raise AssertionError("all tiles shared identical mask")


def test_tile_locality():
    K, N = 512, 128
    h = _rand((4, K), 11)
    w = torch.ones((N, K), dtype=torch.float32) * 1e-3
    w[:64, :256] = 1.0
    w[64:, 256:] = 1.0
    c = group_partials(h, w, 256)
    score = group_scores(c, 64)
    mask = select_keep_mask(score, 1)
    assert bool(mask[0, 0, 0]) and bool(mask[0, 1, 1])
    assert not bool(mask[0, 0, 1]) and not bool(mask[0, 1, 0])
    y = apply_mask(c, mask, 64)
    y_ref = torch.zeros_like(y)
    for t in range(4):
        for tile in range(2):
            kept = mask[t, tile].nonzero().flatten()
            for g in kept.tolist():
                y_ref[t, tile * 64:(tile + 1) * 64] += c[g, t, tile * 64:(tile + 1) * 64]
    assert torch.allclose(y, y_ref, atol=1e-5)
    assert float(y[:, 64:].abs().max()) > 0
    assert float(y[:, :64].abs().max()) > 0


def test_not_global_mask():
    h = _rand((16, 1024), 12)
    w = _rand((256, 1024), 13)
    c = group_partials(h, w, 256)
    score = group_scores(c, 64)
    assert score.shape == (16, tile_count(256, 64), group_count(1024, 256))
    assert score.shape[1] > 1
    mask = select_keep_mask(score, 1)
    per = mask.any(-1)
    assert per.shape == (16, 4)


def test_oracle_down_chunk_matches_oracle_down():
    h = _rand((40, 512), 14)
    w = _rand((128, 512), 15)
    full = oracle_down(h, w, 256, 64, 1, chunk=128)
    y, score, mask = oracle_down_chunk(h, w, 256, 64, 1)
    assert torch.allclose(full, y.to(full.dtype), atol=1e-5)
    assert score.shape == (40, 2, 2)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"PASS {len(tests)} tests")


if __name__ == "__main__":
    main()
