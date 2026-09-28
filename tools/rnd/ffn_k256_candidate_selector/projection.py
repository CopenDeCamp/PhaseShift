import torch

M64 = (1 << 64) - 1


def hadamard64():
    H = torch.ones(1, 1)
    for _ in range(6):
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def _base(layer, tile, seed):
    return ((layer * 1000003) ^ (tile * 9176) ^ (seed * 104729)) & M64


def make_R(layer, tile, dmax=16, seed=0):
    base = _base(layer, tile, seed)
    keys = [splitmix64(base + i) for i in range(64)]
    perm = sorted(range(64), key=lambda i: keys[i])
    signs = torch.tensor(
        [1.0 if (splitmix64(base ^ ((i + 1) * 2654435761)) & 1) == 0 else -1.0
         for i in range(dmax)], dtype=torch.float32)
    H = hadamard64()
    return (H[torch.tensor(perm[:dmax], dtype=torch.long)] / 8.0) * signs[:, None]


def make_R_all(layer, n_tiles, dmax=16, seed=0):
    return torch.stack([make_R(layer, t, dmax, seed) for t in range(n_tiles)], 0)


def apply_R(c4, R_all, d):
    return torch.einsum("ptnk,tdk->ptnd", c4, R_all[:, :d, :].to(c4.device))
