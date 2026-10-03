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

from oracle_down import read_corpus  # noqa: E402
from run_oracle import load_model  # noqa: E402
from subset_select import norms2, objective, small_l2  # noqa: E402
from gateA_pool_diag import capture, mask_overlap  # noqa: E402
from prefilter import make_pool, pool_scores, restricted_pair, unrestricted_pair  # noqa: E402
from pool_sketch import SKETCH_METHODS, pool_sketch_select  # noqa: E402
from projection import make_R_all  # noqa: E402


def j_stats(J):
    q = torch.quantile(J, torch.tensor([0.5, 0.9, 0.99], device=J.device))
    return {"mean": float(J.mean()), "median": float(q[0]), "p90": float(q[1]),
            "p99": float(q[2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B")
    ap.add_argument("--corpus", default="artifacts/ffn_prune/corpus.psktok")
    ap.add_argument("--layers", default="0,16,28,32,48,63")
    ap.add_argument("--n-tokens", type=int, default=128)
    ap.add_argument("--k-group", type=int, default=256)
    ap.add_argument("--tile-width", type=int, default=64)
    ap.add_argument("--pools", default="40,48")
    ap.add_argument("--dims", default="16,24,32,48")
    ap.add_argument("--skips", default="16,20,24,27,28")
    ap.add_argument("--prefilters", default="exact_l2,act_wnorm")
    ap.add_argument("--methods", default="residual,pair4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-count", type=int, default=4)
    ap.add_argument("--swap-iters", type=int, default=8)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--mem-gib", type=int, default=27)
    ap.add_argument("--out", default="artifacts/ffn_k256_candidate_selector/sketch_pool_diag.json")
    args = ap.parse_args()

    gpu_ids = tuple(int(x) for x in args.gpus.split(",")) if args.gpus else \
        tuple(range(min(3, torch.cuda.device_count())))
    device = f"cuda:{gpu_ids[0]}"
    layers = [int(x) for x in args.layers.split(",")]
    pools = [int(x) for x in args.pools.split(",")]
    dims = [int(x) for x in args.dims.split(",")]
    skips = [int(x) for x in args.skips.split(",")]
    prefs = [x for x in args.prefilters.split(",") if x]
    methods = [x for x in args.methods.split(",") if x]
    dmax = max(dims)

    tokens = read_corpus(args.corpus)
    stride = max(1, len(tokens) // args.n_tokens)
    positions = list(range(0, len(tokens), stride))[:args.n_tokens]

    t0 = time.time()
    model, _, _ = load_model(args.model_dir, gpu_ids, mem_gib=args.mem_gib)
    caps = capture(model, tokens, args.k_group, args.tile_width, layers, positions, device)
    del model
    torch.cuda.empty_cache()
    print(f"[b] captured in {time.time()-t0:.1f}s", flush=True)

    store = {}
    for L in layers:
        rows, a, wn = caps[L]
        rows = rows.to(device).float()
        a = a.to(device).float()
        wn = wn.to(device).float()
        R, ng, tw = rows.shape
        nt = wn.shape[0]
        Rall = make_R_all(L, nt, dmax, args.seed).to(device)
        for k in skips:
            Sstar = unrestricted_pair(rows, k, 16, args.swap_iters)
            Jstar = objective(rows, Sstar)
            store.setdefault(("unrestricted",), {}).setdefault(("star", k), []).append(Jstar.cpu())
            for kind in prefs:
                scores = pool_scores(kind, rows, a, wn)
                for C in pools:
                    pool = make_pool(scores, C)
                    ref = restricted_pair(rows, pool, k, 16, args.swap_iters)
                    store.setdefault(("ref",), {}).setdefault((kind, C, k), []).append(
                        objective(rows, ref).cpu())
                    ov_star = mask_overlap(ref, Sstar)
                    store.setdefault(("ov",), {}).setdefault(("refstar", kind, C, k), []).append(ov_star.cpu())
                    for d in dims:
                        for m in methods:
                            idx = pool_sketch_select(rows, pool, Rall, d, m, k, args.seed_count)
                            J = objective(rows, idx)
                            store.setdefault((kind, C, d, m), {}).setdefault(k, []).append(J.cpu())
                            store.setdefault(("ov",), {}).setdefault((kind, C, d, m, k),
                                                                     []).append(mask_overlap(idx, ref).cpu())
        print(f"[b] layer {L} done", flush=True)

    aggregate = {}
    star = {k: torch.cat(store[("unrestricted",)][("star", k)]) for k in skips}
    ref = {key: torch.cat(v) for key, v in store[("ref",)].items()}
    for key, byk in store.items():
        if key[0] in ("unrestricted", "ref", "ov"):
            continue
        kind, C, d, m = key
        for k in skips:
            J = torch.cat(byk[k])
            refJ = ref[(kind, C, k)]
            rec = {
                "j": j_stats(J),
                "ratio_vs_pool_exact": float(J.mean() / refJ.mean()),
                "ratio_vs_unrestricted": float(J.mean() / star[k].mean()),
                "overlap_pool_exact": float(torch.cat(
                    store[("ov",)][(kind, C, d, m, k)]).mean()),
                "overlap_unrestricted": float(torch.cat(
                    store[("ov",)][("refstar", kind, C, k)]).mean()),
            }
            aggregate.setdefault(kind, {}).setdefault(str(C), {}).setdefault(str(d), {}).setdefault(
                m, {})[str(k)] = rec

    meta = {"model_dir": args.model_dir, "corpus": args.corpus, "layers": layers,
            "n_tokens": len(positions), "positions": positions, "pools": pools, "dims": dims,
            "skips": skips, "prefilters": prefs, "methods": methods, "seed": args.seed,
            "seed_count": args.seed_count, "n_groups": 68, "dmax": dmax}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "aggregate": aggregate}, f, indent=1)
    print(f"[b] wrote {args.out} in {time.time()-t0:.1f}s", flush=True)

    for k in skips:
        parts = [f"k={k}({k/68*100:5.2f}%)"]
        for kind in prefs:
            for C in pools:
                for d in dims:
                    for m in methods:
                        r = aggregate[kind][str(C)][str(d)][m][str(k)]
                        parts.append(f"{kind[:4]}C{C}d{d}{m}={r['ratio_vs_pool_exact']:.3f}")
        print(" ".join(parts), flush=True)


if __name__ == "__main__":
    main()
