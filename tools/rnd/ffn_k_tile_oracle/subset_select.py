import torch


def norms2(C):
    return (C * C).sum(-1)


def _gather_groups(C, idx):
    B, ng, D = C.shape
    m = idx.shape[1]
    return torch.gather(C, 1, idx.unsqueeze(-1).expand(-1, -1, D))


def objective(C, idx):
    s = _gather_groups(C, idx).sum(1)
    return (s * s).sum(-1)


def small_l2(C, k):
    if k <= 0:
        return torch.zeros(C.shape[0], 0, dtype=torch.long, device=C.device)
    return torch.topk(norms2(C), k, dim=-1, largest=False).indices


def residual_greedy(C, k, init_idx=None):
    B, ng, D = C.shape
    dev = C.device
    n2 = norms2(C)
    e = torch.zeros(B, D, device=dev, dtype=C.dtype)
    sel = torch.zeros(B, ng, dtype=torch.bool, device=dev)
    idx = torch.zeros(B, k, dtype=torch.long, device=dev)
    start = 0
    if init_idx is not None and init_idx.shape[1] > 0:
        start = init_idx.shape[1]
        idx[:, :start] = init_idx
        sel.scatter_(1, init_idx, True)
        e = _gather_groups(C, init_idx).sum(1)
    for step in range(start, k):
        dots = torch.bmm(C, e.unsqueeze(-1)).squeeze(-1)
        cost = n2 + 2.0 * dots
        cost = cost.masked_fill(sel, float("inf"))
        g = cost.argmin(-1)
        idx[:, step] = g
        sel.scatter_(1, g.unsqueeze(1), True)
        e = e + torch.gather(C, 1, g[:, None, None].expand(-1, 1, D)).squeeze(1)
    return idx


def pair_cost_matrix(C):
    B, ng, D = C.shape
    G = torch.bmm(C, C.transpose(1, 2))
    n2 = norms2(C)
    pc = n2[:, :, None] + n2[:, None, :] + 2.0 * G
    tri = torch.triu(torch.ones(ng, ng, dtype=torch.bool, device=C.device), diagonal=0)
    return pc.masked_fill(tri.unsqueeze(0), float("inf"))


def pair_seeds(C, seed_count):
    B, ng, D = C.shape
    pc = pair_cost_matrix(C).reshape(B, ng * ng)
    _, idxs = torch.topk(pc, seed_count, dim=-1, largest=False)
    return idxs // ng, idxs % ng


def pair_seeded_greedy(C, k, seed_count=16, extra_seed=None):
    if k <= 1:
        return small_l2(C, k)
    B, ng, D = C.shape
    dev = C.device
    n2 = norms2(C)
    si, sj = pair_seeds(C, seed_count)
    if extra_seed is not None and extra_seed.shape[1] >= 2:
        si = torch.cat([si, extra_seed[:, 0:1]], 1)
        sj = torch.cat([sj, extra_seed[:, 1:2]], 1)
    S = si.shape[1]
    e = _gather_groups(C, si) + _gather_groups(C, sj)
    sel = torch.zeros(B, S, ng, dtype=torch.bool, device=dev)
    sel.scatter_(2, si.unsqueeze(-1), True)
    sel.scatter_(2, sj.unsqueeze(-1), True)
    idx = torch.zeros(B, S, k, dtype=torch.long, device=dev)
    idx[:, :, 0] = si
    idx[:, :, 1] = sj
    for step in range(2, k):
        dots = torch.bmm(e, C.transpose(1, 2))
        cost = n2[:, None, :] + 2.0 * dots
        cost = cost.masked_fill(sel, float("inf"))
        g = cost.argmin(-1)
        idx[:, :, step] = g
        sel.scatter_(2, g.unsqueeze(-1), True)
        e = e + torch.gather(C, 1, g.unsqueeze(-1).expand(-1, -1, D))
    best = (e * e).sum(-1).argmin(-1)
    return idx.gather(1, best[:, None, None].expand(-1, 1, k)).squeeze(1)


def pair_greedy_swap(C, k, seed_count=16, max_swap_iters=8, extra_seed=None):
    if k <= 1:
        return small_l2(C, k)
    idx = pair_seeded_greedy(C, k, seed_count, extra_seed)
    B, ng, D = C.shape
    n2 = norms2(C)
    for _ in range(max_swap_iters):
        sel = torch.zeros(B, ng, dtype=torch.bool, device=C.device)
        sel.scatter_(1, idx, True)
        groups = _gather_groups(C, idx)
        e = groups.sum(1)
        cur = (e * e).sum(-1)
        E = e.unsqueeze(1) - groups
        dots = torch.bmm(E, C.transpose(1, 2))
        tn2 = (E * E).sum(-1, keepdim=True)
        cost = tn2 + n2[:, None, :] + 2.0 * dots
        cost = cost.masked_fill(sel[:, None, :], float("inf"))
        minval, arg = cost.reshape(B, k * ng).min(-1)
        improve = minval < cur
        if not bool(improve.any()):
            break
        a = arg // ng
        g = arg % ng
        new_idx = idx.scatter(1, a[:, None], g[:, None])
        idx = torch.where(improve[:, None], new_idx, idx)
    return idx


def exact_subset(C, k):
    from itertools import combinations
    B, ng, D = C.shape
    combos = list(combinations(range(ng), k))
    ci = torch.tensor(combos, dtype=torch.long, device=C.device)
    s = C[:, ci].sum(2)
    cost = (s * s).sum(-1)
    return ci[cost.argmin(-1)]


def apply_skip_indices(c, skip_idx, tile_width):
    ng, T, N = c.shape
    nt = N // tile_width
    R = T * nt
    keep = torch.ones(R, ng, dtype=torch.bool, device=c.device)
    keep.scatter_(1, skip_idx, False)
    keep = keep.reshape(T, nt, ng)
    cr = c.reshape(ng, T, nt, tile_width)
    return (cr * keep.permute(2, 0, 1).unsqueeze(-1)).sum(0).reshape(T, N)


def partials_to_rows(c, tile_width):
    ng, T, N = c.shape
    nt = N // tile_width
    cr = c.reshape(ng, T, nt, tile_width)
    return cr.permute(1, 2, 0, 3).reshape(T * nt, ng, tile_width)


def rows_to_skip_mask(skip_idx, T, nt, ng):
    R = T * nt
    keep = torch.ones(R, ng, dtype=torch.bool, device=skip_idx.device)
    keep.scatter_(1, skip_idx, False)
    return keep.reshape(T, nt, ng)
