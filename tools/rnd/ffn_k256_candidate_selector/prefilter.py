import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "ffn_k_tile_oracle"))

from subset_select import (norms2, pair_greedy_swap, residual_greedy,  # noqa: E402
                           small_l2)

PREFILTERS = ("exact_l2", "act_energy", "act_wnorm")


def act_energy(h, k_group):
    T, K = h.shape
    ng = K // k_group
    return h.reshape(T, ng, k_group).float().pow(2).sum(-1)


def weight_norm2(w, k_group, tile_width):
    N, K = w.shape
    ng = K // k_group
    nt = N // tile_width
    out = torch.empty(nt, ng, dtype=torch.float32, device=w.device)
    for g in range(ng):
        blk = w[:, g * k_group:(g + 1) * k_group].float().pow(2).sum(-1)
        out[:, g] = blk.reshape(nt, tile_width).sum(-1)
    return out


def pool_scores(kind, rows, a, wn):
    if kind == "exact_l2":
        return norms2(rows)
    R, ng, _ = rows.shape
    if kind == "act_energy":
        T = a.shape[0]
        nt = R // T
        return a[:, None, :].expand(T, nt, ng).reshape(R, ng).contiguous()
    if kind == "act_wnorm":
        nt = wn.shape[0]
        T = R // nt
        return (a[:, None, :] * wn[None, :, :]).reshape(R, ng).contiguous()
    raise ValueError(kind)


def make_pool(scores, C):
    return torch.topk(scores, C, dim=-1, largest=False).indices


def pool_bool(pool, ng):
    m = torch.zeros(pool.shape[0], ng, dtype=torch.bool, device=pool.device)
    m.scatter_(1, pool, True)
    return m


def restricted_pair(rows, pool, k, seed_count=16, swap_iters=8):
    pg = torch.gather(rows, 1, pool.unsqueeze(-1).expand(-1, -1, rows.shape[-1]))
    idx = pair_greedy_swap(pg, k, seed_count, swap_iters, extra_seed=small_l2(pg, 2))
    return torch.gather(pool, 1, idx)


def restricted_residual(rows, pool, k):
    pg = torch.gather(rows, 1, pool.unsqueeze(-1).expand(-1, -1, rows.shape[-1]))
    return torch.gather(pool, 1, residual_greedy(pg, k))


def unrestricted_pair(rows, k, seed_count=16, swap_iters=8):
    return pair_greedy_swap(rows, k, seed_count, swap_iters, extra_seed=small_l2(rows, 2))


def score_rank(scores):
    return torch.argsort(torch.argsort(scores, dim=-1), dim=-1)
