import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ffn_k_tile_oracle"))

from oracle_down import group_partials, read_corpus  # noqa: E402
from run_oracle import load_model, mlp_list  # noqa: E402
from subset_select import norms2, objective, partials_to_rows, small_l2  # noqa: E402
from prefilter import (PREFILTERS, act_energy, make_pool, pool_bool, pool_scores,  # noqa: E402
                       restricted_pair, score_rank, unrestricted_pair, weight_norm2)


def capture(model, tokens, k_group, tile_width, layers, positions, device):
    mlps = dict(mlp_list(model))
    out = {}
    orig = {}
    for layer in layers:
        mlp = mlps[layer]
        orig[layer] = mlp.forward

        def make(L, m):
            def fwd(x):
                gate = m.gate_proj(x)
                up = m.up_proj(x)
                h = m.act_fn(gate) * up
                hf = h.reshape(-1, h.shape[-1])
                pos = torch.tensor(positions, dtype=torch.long, device=hf.device)
                hsel = hf.index_select(0, pos).float()
                c = group_partials(hsel, m.down_proj.weight, k_group)
                rows = partials_to_rows(c, tile_width).contiguous().cpu()
                out[L] = (rows, act_energy(hsel, k_group).cpu(),
                          weight_norm2(m.down_proj.weight, k_group, tile_width).cpu())
                return m.down_proj(h)
            return fwd

        mlp.forward = make(layer, mlp)

    ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        for layer, fwd in orig.items():
            mlps[layer].forward = fwd
    torch.cuda.empty_cache()
    return out


def j_stats(J):
    q = torch.quantile(J, torch.tensor([0.5, 0.9, 0.99], device=J.device))
    return {"mean": float(J.mean()), "median": float(q[0]), "p90": float(q[1]),
            "p99": float(q[2])}


def gather_recall(selected, pool, ng):
    m = pool_bool(pool, ng)
    hit = torch.gather(m, 1, selected)
    return hit.float().mean(-1)


def gather_rank(rank, selected):
    return torch.gather(rank, 1, selected)


def mask_overlap(a, b):
    ng = max(int(a.max()), int(b.max())) + 1
    ma = torch.zeros(a.shape[0], ng, dtype=torch.bool, device=a.device).scatter_(1, a, True)
    mb = torch.zeros(b.shape[0], ng, dtype=torch.bool, device=b.device).scatter_(1, b, True)
    return (ma & mb).sum(-1).float() / a.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B")
    ap.add_argument("--corpus", default="artifacts/ffn_prune/corpus.psktok")
    ap.add_argument("--layers", default="0,16,28,32,48,63")
    ap.add_argument("--n-tokens", type=int, default=128)
    ap.add_argument("--k-group", type=int, default=256)
    ap.add_argument("--tile-width", type=int, default=64)
    ap.add_argument("--pools", default="32,40,48,56,68")
    ap.add_argument("--skips", default="16,20,24,27,28")
    ap.add_argument("--prefilters", default="exact_l2,act_energy,act_wnorm")
    ap.add_argument("--seed-count", type=int, default=16)
    ap.add_argument("--swap-iters", type=int, default=8)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--mem-gib", type=int, default=27)
    ap.add_argument("--out", default="artifacts/ffn_k256_candidate_selector/pool_diag.json")
    args = ap.parse_args()

    gpu_ids = tuple(int(x) for x in args.gpus.split(",")) if args.gpus else \
        tuple(range(min(3, torch.cuda.device_count())))
    device = f"cuda:{gpu_ids[0]}"
    layers = [int(x) for x in args.layers.split(",")]
    pools = [int(x) for x in args.pools.split(",")]
    skips = [int(x) for x in args.skips.split(",")]
    prefs = [x for x in args.prefilters.split(",") if x]

    tokens = read_corpus(args.corpus)
    stride = max(1, len(tokens) // args.n_tokens)
    positions = list(range(0, len(tokens), stride))[:args.n_tokens]

    t0 = time.time()
    model, _, _ = load_model(args.model_dir, gpu_ids, mem_gib=args.mem_gib)
    caps = capture(model, tokens, args.k_group, args.tile_width, layers, positions, device)
    del model
    torch.cuda.empty_cache()
    print(f"[pool] captured in {time.time()-t0:.1f}s", flush=True)

    store = {}
    recall_store = {}
    rank_store = {}
    overlap_store = {}
    per_layer = {}
    for L in layers:
        rows, a, wn = caps[L]
        rows = rows.to(device).float()
        a = a.to(device).float()
        wn = wn.to(device).float()
        R, ng, tw = rows.shape
        y2 = rows.sum(1).pow(2).sum(-1)
        gsum = norms2(rows).sum(-1)
        layer_entry = {}

        def put(key, J, recall=None, rank=None, overlap=None, ref_pair=None, jsl=None):
            store.setdefault(key, []).append(J.cpu())
            if recall is not None:
                recall_store.setdefault(key, []).append(recall.cpu())
            if rank is not None:
                rank_store.setdefault(key, []).append(rank.reshape(-1).cpu())
            if overlap is not None:
                overlap_store.setdefault(key, []).append(overlap.cpu())
            if ref_pair is not None:
                store.setdefault(("jp", key), []).append(ref_pair.cpu())
            if jsl is not None:
                store.setdefault(("jsl", key), []).append(jsl.cpu())

        layer_entry["k"] = {}
        for k in skips:
            Sstar = unrestricted_pair(rows, k, args.seed_count, args.swap_iters)
            Jstar = objective(rows, Sstar)
            Jsl = objective(rows, small_l2(rows, k))
            put(("unrestricted", 68, k), Jstar, jsl=Jsl, ref_pair=Jstar)
            rank_store.setdefault(("unrestricted_rank", 68, k), []).append(
                gather_rank(score_rank(norms2(rows)), Sstar).reshape(-1).cpu())
            entry = {"unrestricted": {"j": j_stats(Jstar), "small_l2_mean": float(Jsl.mean())}}
            for kind in prefs:
                scores = pool_scores(kind, rows, a, wn)
                rank = score_rank(scores)
                for C in pools:
                    pool = make_pool(scores, C)
                    idx = restricted_pair(rows, pool, k, args.seed_count, args.swap_iters)
                    J = objective(rows, idx)
                    rec = gather_recall(Sstar, pool, ng)
                    rk = gather_rank(rank, Sstar).float()
                    ov = mask_overlap(idx, Sstar)
                    key = (kind, C, k)
                    put(key, J, recall=rec, rank=rk, overlap=ov, ref_pair=Jstar, jsl=Jsl)
                    entry[f"{kind}:{C}"] = {
                        "j": j_stats(J),
                        "ratio_vs_unrestricted": float(J.mean() / Jstar.mean()),
                        "improvement_vs_small_l2": float(Jsl.mean() / J.mean()),
                        "overlap": float(ov.mean()),
                        "recall": {"mean": float(rec.mean()), "p10": float(rec.quantile(0.10)),
                                   "p50": float(rec.quantile(0.5)),
                                   "p90": float(rec.quantile(0.90)),
                                   "min": float(rec.min())},
                        "rank": {"median": float(rk.median()),
                                 "p90": float(rk.quantile(0.90)),
                                 "p99": float(rk.quantile(0.99))},
                    }
            layer_entry["k"][str(k)] = entry
        per_layer[str(L)] = layer_entry
        print(f"[pool] layer {L} done", flush=True)

    aggregate = {}
    for key, Js in store.items():
        if key[0] in ("jp", "jsl"):
            continue
        kind, C, k = key
        J = torch.cat(Js)
        rec = torch.cat(recall_store[(kind, C, k)]) if (kind, C, k) in recall_store else None
        rk = torch.cat(rank_store[(kind, C, k)]) if (kind, C, k) in rank_store else None
        ov = torch.cat(overlap_store[(kind, C, k)]) if (kind, C, k) in overlap_store else None
        Jstar = torch.cat(store[("jp", (kind, C, k))])
        Jsl = torch.cat(store[("jsl", (kind, C, k))])
        rec_out = None
        if rec is not None:
            rec_out = {"mean": float(rec.mean()), "p10": float(rec.quantile(0.10)),
                       "p50": float(rec.quantile(0.5)), "p90": float(rec.quantile(0.90)),
                       "min": float(rec.min())}
        aggregate.setdefault(kind, {}).setdefault(str(C), {})[str(k)] = {
            "j": j_stats(J),
            "ratio_vs_unrestricted": float(J.mean() / Jstar.mean()),
            "improvement_vs_small_l2": float(Jsl.mean() / J.mean()),
            "overlap": float(ov.mean()) if ov is not None else None,
            "recall": rec_out,
            "rank": None if rk is None else {"median": float(rk.median()),
                                             "p90": float(rk.quantile(0.90)),
                                             "p99": float(rk.quantile(0.99))},
        }

    meta = {"model_dir": args.model_dir, "corpus": args.corpus, "layers": layers,
            "n_tokens": len(positions), "positions": positions, "pools": pools,
            "skips": skips, "prefilters": prefs, "n_groups": 68,
            "seed_count": args.seed_count, "swap_iters": args.swap_iters,
            "tile_width": args.tile_width}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "aggregate": aggregate, "per_layer": per_layer}, f, indent=1)
    print(f"[pool] wrote {args.out} in {time.time()-t0:.1f}s", flush=True)

    for k in skips:
        parts = [f"k={k}({k/68*100:5.2f}%)", f"unrestricted={aggregate['unrestricted']['68'][str(k)]['j']['mean']:.4g}"]
        for kind in prefs:
            for C in pools:
                m = aggregate[kind][str(C)][str(k)]
                parts.append(f"{kind[:4]}C{C}={m['ratio_vs_unrestricted']:.3f}(r{m['recall']['mean']:.2f})")
        print(" ".join(parts), flush=True)


if __name__ == "__main__":
    main()
