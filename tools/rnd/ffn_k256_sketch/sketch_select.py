import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "ffn_k_tile_oracle"))

from subset_select import (pair_greedy_swap, pair_seeded_greedy,  # noqa: E402
                           residual_greedy, small_l2)

EXACT_METHODS = ("small_l2", "exact_residual", "exact_pair")
SKETCH_METHODS = ("sk_residual", "sk_pair4", "sk_pair4_s1", "sk_pair8")


def choose(method, c, z, k, seed_count=4):
    if method == "small_l2":
        return small_l2(c, k)
    if method == "exact_residual":
        return residual_greedy(c, k)
    if method == "exact_pair":
        return pair_greedy_swap(c, k, 16, 8, extra_seed=small_l2(c, 2))
    if method == "sk_residual":
        return residual_greedy(z, k)
    if method == "sk_pair4":
        return pair_seeded_greedy(z, k, seed_count, extra_seed=small_l2(z, 2))
    if method == "sk_pair4_s1":
        return pair_greedy_swap(z, k, seed_count, 1, extra_seed=small_l2(z, 2))
    if method == "sk_pair8":
        return pair_seeded_greedy(z, k, 8, extra_seed=small_l2(z, 2))
    raise ValueError(method)
