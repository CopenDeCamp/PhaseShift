import struct
from pathlib import Path

import torch


def read_corpus(path):
    data = Path(path).read_bytes()
    if data[:8] != b"PSKLDTOK":
        raise ValueError("bad corpus magic")
    ver, = struct.unpack_from("<I", data, 8)
    if ver != 1:
        raise ValueError("bad corpus version")
    count, = struct.unpack_from("<Q", data, 12)
    return list(struct.unpack_from("<%dI" % count, data, 20))


def group_count(k, k_group):
    if k % k_group != 0:
        raise ValueError(f"K={k} not divisible by k_group={k_group}")
    return k // k_group


def tile_count(n, tile_width):
    if tile_width > n or n % tile_width != 0:
        raise ValueError(f"N={n} not compatible with tile_width={tile_width}")
    return n // tile_width


def skip_count_for_fraction(ng, frac):
    return int(round(frac * ng))


def keep_count_for_fraction(ng, frac):
    return ng - skip_count_for_fraction(ng, frac)


def actual_skip_fraction(ng, skip_count):
    return skip_count / ng


def group_partials(h, w, k_group):
    if h.shape[-1] != w.shape[-1]:
        raise ValueError("K mismatch between activation and weight")
    T, K = h.shape
    N = w.shape[0]
    ng = group_count(K, k_group)
    hf = h.float()
    c = torch.empty((ng, T, N), dtype=torch.float32, device=h.device)
    for g in range(ng):
        s = g * k_group
        e = s + k_group
        c[g] = torch.mm(hf[:, s:e], w[:, s:e].float().t())
    return c


def group_scores(c, tile_width):
    ng, T, N = c.shape
    nt = tile_count(N, tile_width)
    cr = c.reshape(ng, T, nt, tile_width)
    return cr.pow(2).sum(-1).permute(1, 2, 0).contiguous()


def select_keep_mask(score, keep_count):
    ng = score.shape[-1]
    if keep_count >= ng:
        return torch.ones_like(score, dtype=torch.bool)
    if keep_count <= 0:
        return torch.zeros_like(score, dtype=torch.bool)
    idx = torch.topk(score, keep_count, dim=-1).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def apply_mask(c, mask, tile_width):
    ng, T, N = c.shape
    nt = tile_count(N, tile_width)
    cr = c.reshape(ng, T, nt, tile_width)
    m = mask.permute(2, 0, 1).unsqueeze(-1)
    return (cr * m).sum(0).reshape(T, N)


def oracle_down_chunk(h, w, k_group, tile_width, keep_count):
    c = group_partials(h, w, k_group)
    score = group_scores(c, tile_width)
    mask = select_keep_mask(score, keep_count)
    y = apply_mask(c, mask, tile_width)
    return y, score, mask


def oracle_down(h, w, k_group, tile_width, keep_count, chunk=128):
    T = h.shape[0]
    N = w.shape[0]
    parts = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        y, _, _ = oracle_down_chunk(h[start:end], w, k_group, tile_width, keep_count)
        parts.append(y)
    y = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
    return y.to(h.dtype)


def reference_down(h, w):
    return torch.mm(h.float(), w.float().t())


def reference_masked_down(h, w, k_group, tile_width, skip_count):
    ng = group_count(h.shape[-1], k_group)
    keep = ng - skip_count
    c = group_partials(h, w, k_group)
    score = group_scores(c, tile_width)
    mask = select_keep_mask(score, keep)
    return apply_mask(c, mask, tile_width)
