import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "ffn_k_tile_oracle"))

from projection import apply_R  # noqa: E402
from subset_select import (pair_greedy_swap, pair_seeded_greedy,  # noqa: E402
                           residual_greedy, small_l2)

SKETCH_METHODS = ("residual", "pair4", "pair4_s1")


def pool_sketch_z(rows, pool, Rall, d):
    R, ng, tw = rows.shape
    nt = Rall.shape[0]
    T = R // nt
    c4 = rows.reshape(T, nt, ng, tw)
    z = apply_R(c4, Rall, d).reshape(R, ng, d)
    return torch.gather(z, 1, pool.unsqueeze(-1).expand(-1, -1, d))


def pool_sketch_select(rows, pool, Rall, d, method, k, seed_count=4):
    zg = pool_sketch_z(rows, pool, Rall, d)
    if method == "residual":
        idx = residual_greedy(zg, k)
    elif method == "pair4":
        idx = pair_seeded_greedy(zg, k, seed_count, extra_seed=small_l2(zg, 2))
    elif method == "pair4_s1":
        idx = pair_greedy_swap(zg, k, seed_count, 1, extra_seed=small_l2(zg, 2))
    else:
        raise ValueError(method)
    return torch.gather(pool, 1, idx)
